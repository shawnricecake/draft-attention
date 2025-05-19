# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]



import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None

# Xuan: import block sparse attention from mit han lab
from block_sparse_attn import block_sparse_attn_func


class Draft_Attention(nn.Module):
    def __init__(
        self, 
        pool_h: int = 8,
        pool_w: int = 16,
        latent_h: int = 48,
        latent_w: int = 80,
        visual_len: int = 126_720,
        text_len: int = 0,  # we assume the text is at the end of the sequence, which is the case in the hunyuan model
        sparsity_ratio: float = 0.9,
    ):
        super(Draft_Attention, self).__init__()
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.visual_len = visual_len
        self.text_len = text_len
        self.sparsity_ratio = sparsity_ratio
        self.reorg_idx, self.restore_idx = self.generate_reorg_restore_indices(
            pool_h=pool_h,
            pool_w=pool_w,
            latent_h=latent_h,
            latent_w=latent_w,
            visual_len=visual_len,
            text_len=text_len
        )

    def generate_reorg_indices(
        self,
        total_len: int = 126_720,
        part_size: int = 640,
        block_size: int = 80,
        sub_block_size: int = 16
    ) -> list[int]:
        """
        Build a full-length reorder index list of length `total_len` that:
        1. Splits [0..total_len) into parts of size `part_size`.
        2. Within each part, splits into blocks of `block_size`.
        3. Within each block, splits into sub-blocks of `sub_block_size`.
        4. Reorders so that sub-blocks across all blocks are contiguous.

        Returns:
        reorg_idx: List[int] of length total_len, where
                    new_tensor = old_tensor[:, reorg_idx, ...]
        """
        assert total_len % part_size == 0, "total_len must be multiple of part_size"
        assert part_size % block_size == 0, "part_size must be multiple of block_size"
        assert block_size % sub_block_size == 0, "block_size must be multiple of sub_block_size"

        num_parts = total_len // part_size  # 126_720 // 640 = 198
        blocks_per_part = part_size // block_size   # 640 // 80 = 8
        subs_per_block = block_size // sub_block_size   # 80 // 16 = 5

        # build the pattern for one part
        part_pattern = []
        for c in range(subs_per_block):
            for b in range(blocks_per_part):
                start = b * block_size + c * sub_block_size
                part_pattern.extend(range(start, start + sub_block_size))
        assert len(part_pattern) == part_size

        # tile across all parts
        reorg_idx = []
        for p in range(num_parts):
            base = p * part_size
            reorg_idx.extend(base + i for i in part_pattern)

        return reorg_idx

    def generate_reorg_restore_indices(
        self,
        pool_h: int = 8,
        pool_w: int = 16,
        latent_h: int = 48,
        latent_w: int = 80,
        visual_len: int = 126_720,
        text_len: int = 0,    # if there is text at the behind, the text_len should be added to the reorg and restore indices without changing the order
    ) -> tuple[list[int], list[int]]:
        """
        Build the reorg and restore indices in one go.
        """
        part_size = latent_w * pool_h
        block_size = latent_w
        sub_block_size = pool_w

        assert latent_h % pool_h == 0, "latent_h must be multiple of pool_h"
        assert visual_len % part_size == 0, "total_len must be multiple of part_size"
        assert block_size % sub_block_size == 0, "block_size must be multiple of sub_block_size"
        
        reorg_idx = self.generate_reorg_indices(visual_len, part_size, block_size, sub_block_size)

        # invert the mapping
        restore_idx = [0] * visual_len
        for new_pos, orig_pos in enumerate(reorg_idx):
            restore_idx[orig_pos] = new_pos

        # add the text_len to the reorg and restore indices
        if text_len > 0:
            reorg_idx += [i for i in range(visual_len, text_len + visual_len)]
            restore_idx += [i for i in range(visual_len, text_len + visual_len)]
        
        return reorg_idx, restore_idx
    
    def sample_qk_attention_2d(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        frame_h: int,
        frame_w: int,
        pool_h: int,
        pool_w: int,
    ) -> torch.Tensor:
        """
        q, k: [L, H, D] where the first num_frames*frame_h*frame_w tokens are video,
                L_vid = num_frames * frame_h * frame_w.
        frame_h, frame_w: spatial dims of each frame.
        pool_h, pool_w: 2D pooling kernel (and stride) for sampling each frame.

        Returns:
        attn_map: [H, S, S] where
            S = num_frames * ceil(frame_h/pool_h) * ceil(frame_w/pool_w).
        """
        assert len(q.shape) == 3, "q must be of shape [L, H, D], similar for k, which is similar as flash attention input."
        L, H, D = q.shape
        frame_tokens = frame_h * frame_w
        assert L % frame_tokens == 0, "L must be multiple of frame_h*frame_w"
        num_frames = L // frame_tokens

        # 1) Slice out the video part and reshape to frames:
        #    [L, H, D] → [num_frames, frame_h, frame_w, H, D]
        q_vid = q.view(num_frames, frame_h, frame_w, H, D)
        k_vid = k.view(num_frames, frame_h, frame_w, H, D)

        # 2) Permute & merge (num_frames, H*D) into channel dim:
        #    → [num_frames, H*D, frame_h, frame_w]
        q_vid = q_vid.permute(0, 3, 4, 1, 2).reshape(
            num_frames, H * D, frame_h, frame_w
        )
        k_vid = k_vid.permute(0, 3, 4, 1, 2).reshape(
            num_frames, H * D, frame_h, frame_w
        )

        # 3) 2D max‐pool each frame (ceil_mode ensures we cover the edges):
        #    → [num_frames, H*D, S_h, S_w]
        q_pooled = F.avg_pool2d(
            q_vid, kernel_size=(pool_h, pool_w),
            stride=(pool_h, pool_w), ceil_mode=True
        )
        k_pooled = F.avg_pool2d(
            k_vid, kernel_size=(pool_h, pool_w),
            stride=(pool_h, pool_w), ceil_mode=True
        )

        S_h, S_w = q_pooled.shape[-2:]
        S = num_frames * S_h * S_w

        # 4) Un‐merge channel back to [S, H, D]:
        #    → [num_frames, H, D, S_h, S_w] → [S, H, D]
        def unmerge(x):
            x = x.reshape(num_frames, H, D, S_h, S_w)
            return x.permute(0, 3, 4, 1, 2).reshape(S, H, D)

        sampled_q = unmerge(q_pooled)
        sampled_k = unmerge(k_pooled)

        # 5) Compute per‐head scaled dot‐prod attention:
        #    [S, H, D] → [H, S, D]
        q_heads = sampled_q.permute(1, 0, 2)
        k_heads = sampled_k.permute(1, 0, 2)

        # → [H, S, S]
        scores = torch.einsum("hld,hmd->hlm", q_heads, k_heads) / math.sqrt(D)
        attn_map = torch.softmax(scores, dim=-1)

        return attn_map


    def attention_percentile_mask_headwise(self, attn_map: torch.Tensor, r: float) -> torch.BoolTensor:
        """
        Build a mask per head so that each head keeps its top-r fraction of entries as True.

        Args:
        attn_map: Tensor of shape [H, S, S], attention scores (e.g. after softmax).
        r: float in (0,1), fraction of entries *per head* to keep True.

        Returns:
        mask: BoolTensor of shape [H, S, S], where for each head h,
                mask[h].float().mean() ≈ r.
        """
        H, S, _ = attn_map.shape
        mask = torch.zeros_like(attn_map, dtype=torch.bool)

        # process each head independently
        for h in range(H):
            head_scores = attn_map[h]                # [S, S]
            flat = head_scores.flatten()             # [S*S]
            n = flat.numel()
            k = int((1.0 - r) * n)                   # number of smallest to exclude

            if k == 0:
                mask[h] = True
                continue
            if k >= n:
                # nothing to keep
                continue

            # threshold = max of the k smallest scores
            threshold = torch.topk(flat, k, largest=False).values.max()

            # build head mask
            mask[h] = head_scores >= threshold

        return mask

    def forward(
        self,
        q,
        k,
        v,
        attn_mask=None,
        causal=False,
        drop_rate=0.0,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        batch_size=1,   # we set the batch size default to 1 here
        sparsity_ratio=None,
        block_sparse_attention=True, # if not use block sparse attention, use the flash attention
    ):
        if sparsity_ratio is None:
            sparsity_ratio = self.sparsity_ratio
        
        # ======== block_sparse_attention setup ========
        streaming_info = None
        q_len, num_heads, head_dim = q.shape
        # Xuan: in 'head_mask_type': 
        #   torch.tensor([x] * 24, device=q.device, dtype=torch.int32) (PS: in hunyuan, there are 24 heads)
        # mask_type = 0 denotes the dense attention
        # mask_type = -1 denotes the streaming attention
        # mask_type = 1 denotes the blocksparse attention (we use this here)
        head_mask_type = torch.tensor([1] * num_heads, device=q.device, dtype=torch.int32)
        # ======== block_sparse_attention setup ========

        attn = self.sample_qk_attention_2d(
            q[:self.visual_len] if self.text_len > 0 else q,
            k[:self.visual_len] if self.text_len > 0 else k,
            frame_h=self.latent_h,
            frame_w=self.latent_w,
            pool_h=self.pool_h,
            pool_w=self.pool_w,
        )
        m_block_dim = (self.visual_len + attn.shape[1] - 1) // attn.shape[1] # 128
        n_block_dim = (self.visual_len + attn.shape[2] - 1) // attn.shape[2] # 128
        q_block_num = (max_seqlen_q + m_block_dim - 1) // m_block_dim
        k_block_num = (max_seqlen_kv + n_block_dim - 1) // n_block_dim
        base_blockmask_visual = self.attention_percentile_mask_headwise(attn, 1-sparsity_ratio)
        base_blockmask_visual = base_blockmask_visual.unsqueeze(0).repeat(
            cu_seqlens_q.numel() - 1, 1, 1, 1
        ).bool()
        base_blockmask = torch.ones(
            (
                cu_seqlens_q.numel() - 1, 
                num_heads, 
                q_block_num, 
                k_block_num
            ),
            device=q.device,
            dtype=q.dtype,
        ).bool()
        base_blockmask[:, :, :base_blockmask_visual.shape[2], :base_blockmask_visual.shape[3]] = base_blockmask_visual

        if block_sparse_attention:
            # re-organize the q and k for visual alignment  # todo
            q = q[self.reorg_idx, :, :]
            k = k[self.reorg_idx, :, :]
            v = v[self.reorg_idx, :, :]

            x = block_sparse_attn_func(
                q, k, v,
                cu_seqlens_q, cu_seqlens_kv,
                head_mask_type,
                streaming_info,
                base_blockmask,
                max_seqlen_q, max_seqlen_kv,
                drop_rate,
                deterministic=False,
                softmax_scale=None,
                is_causal=causal,
                exact_streaming=False,
                return_attn_probs=False,

                # in 'block_sparse_attention' package, the default block size is 128 as hard code
                #   you can revise the block size at: https://github.com/mit-han-lab/Block-Sparse-Attention/blob/6ec5a27a0cd6bd92ea6296698d64e460c73da27e/block_sparse_attn/block_sparse_attn_interface.py#L402
                #   or you can set it as input for the function 'block_sparse_attn_func()'

                # m_block_dim=m_block_dim,
                # n_block_dim=n_block_dim,
            )

            # re-organize the x to the original order
            x = x[self.restore_idx, :, :]  # todo

            # x with shape [(bxs), a, d]
            x = x.view(
                batch_size, max_seqlen_q, x.shape[-2], x.shape[-1]
            )  # reshape x to [b, s, a, d]

            return x
        else:
            # we use normal pytorch attention here if flash attention is not available
            if flash_attn_varlen_func is None:
                if attn_mask is not None and attn_mask.dtype != torch.bool:
                    attn_mask = attn_mask.to(q.dtype)
                x = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
                )
                return x
            
            # we use flash attention here
            x = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
            )
            # x with shape [(bxs), a, d]
            x = x.view(
                batch_size, max_seqlen_q, x.shape[-2], x.shape[-1]
            )  # reshape x to [b, s, a, d]

            return x




def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
    idx_block=None,
    current_timestep=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # =====================================================================
    # # Dense:    # todo
    # x = flash_attn.flash_attn_varlen_func(
    #     q=q,
    #     k=k,
    #     v=v,
    #     cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
    #         0, dtype=torch.int32).to(q.device, non_blocking=True),
    #     cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
    #         0, dtype=torch.int32).to(q.device, non_blocking=True),
    #     max_seqlen_q=lq,
    #     max_seqlen_k=lk,
    #     dropout_p=dropout_p,
    #     softmax_scale=softmax_scale,
    #     causal=causal,
    #     window_size=window_size,
    #     deterministic=deterministic).unflatten(0, (b, lq))
    
    # # output
    # return x.type(out_dtype)
    # =====================================================================

    # self attention: q, k, v shape: [32256, 40, 128];      [80640, 40, 128] 80640=21x 48x80
    # cross attention: q [32256, 40, 128], k, v [512, 40, 128]

    assert q.shape[0] == 32_256 or 80_640, "Currently, we only support 768*512 and 1280*768 resolution, and we will implement the padding for any-resolution generation in the future."

    # =====================================================================
    # Xuan: sparse
    if q.shape[0] != k.shape[0] or (idx_block is not None and idx_block < 1) or (current_timestep[0] > 925):
        # trick from https://github.com/svg-project/Sparse-VideoGen/blob/079364dc2e4ca6cd0c26c8c45eafeb9fbf51ef8e/svg/models/wan/attention.py#L212
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))
        # output
        return x.type(out_dtype)

    draft_attention = Draft_Attention(
            pool_h=8,
            pool_w=16,
            latent_h=32 if q.shape[0] == 32256 else 48,
            latent_w=48 if q.shape[0] == 32256 else 80,
            visual_len=q.shape[0],
            text_len=0,
            sparsity_ratio=0.75 # todo
        )
    x = draft_attention(
        q,
        k,
        v,
        attn_mask=None,
        causal=causal,
        drop_rate=dropout_p,
        cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
            0, dtype=torch.int32).to(q.device, non_blocking=True),
        cu_seqlens_kv=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
            0, dtype=torch.int32).to(q.device, non_blocking=True),
        max_seqlen_q=lq,
        max_seqlen_kv=lk,
        batch_size=1,
    )
    
    # output
    return x.type(out_dtype)
    # =====================================================================



def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out





# sparse block mask functions
def sample_qk_attention_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    frame_h: int,
    frame_w: int,
    pool_h: int,
    pool_w: int,
) -> torch.Tensor:
    """
    q, k: [L, H, D] where the first num_frames*frame_h*frame_w tokens are video,
            L_vid = num_frames * frame_h * frame_w.
    frame_h, frame_w: spatial dims of each frame.
    pool_h, pool_w: 2D pooling kernel (and stride) for sampling each frame.

    Returns:
    attn_map: [H, S, S] where
        S = num_frames * ceil(frame_h/pool_h) * ceil(frame_w/pool_w).
    """
    L, H, D = q.shape
    frame_tokens = frame_h * frame_w
    assert L % frame_tokens == 0, "L must be multiple of frame_h*frame_w"
    num_frames = L // frame_tokens

    # 1) Slice out the video part and reshape to frames:
    #    [L, H, D] → [num_frames, frame_h, frame_w, H, D]
    q_vid = q.view(num_frames, frame_h, frame_w, H, D)
    k_vid = k.view(num_frames, frame_h, frame_w, H, D)

    # 2) Permute & merge (num_frames, H*D) into channel dim:
    #    → [num_frames, H*D, frame_h, frame_w]
    q_vid = q_vid.permute(0, 3, 4, 1, 2).reshape(
        num_frames, H * D, frame_h, frame_w
    )
    k_vid = k_vid.permute(0, 3, 4, 1, 2).reshape(
        num_frames, H * D, frame_h, frame_w
    )

    # 3) 2D max‐pool each frame (ceil_mode ensures we cover the edges):
    #    → [num_frames, H*D, S_h, S_w]

    # q_pooled = F.max_pool2d(
    #     q_vid, kernel_size=(pool_h, pool_w),
    #     stride=(pool_h, pool_w), ceil_mode=True
    # )
    # k_pooled = F.max_pool2d(
    #     k_vid, kernel_size=(pool_h, pool_w),
    #     stride=(pool_h, pool_w), ceil_mode=True
    # )

    q_pooled = F.avg_pool2d(
        q_vid, kernel_size=(pool_h, pool_w),
        stride=(pool_h, pool_w), ceil_mode=True
    )
    k_pooled = F.avg_pool2d(
        k_vid, kernel_size=(pool_h, pool_w),
        stride=(pool_h, pool_w), ceil_mode=True
    )

    S_h, S_w = q_pooled.shape[-2:]
    S = num_frames * S_h * S_w

    # 4) Un‐merge channel back to [S, H, D]:
    #    → [num_frames, H, D, S_h, S_w] → [S, H, D]
    def unmerge(x):
        x = x.reshape(num_frames, H, D, S_h, S_w)
        return x.permute(0, 3, 4, 1, 2).reshape(S, H, D)

    sampled_q = unmerge(q_pooled)
    sampled_k = unmerge(k_pooled)

    # 5) Compute per‐head scaled dot‐prod attention:
    #    [S, H, D] → [H, S, D]
    q_heads = sampled_q.permute(1, 0, 2)
    k_heads = sampled_k.permute(1, 0, 2)

    # → [H, S, S]
    scores = torch.einsum("hld,hmd->hlm", q_heads, k_heads) / math.sqrt(D)
    attn_map = torch.softmax(scores, dim=-1)

    return attn_map

def attention_percentile_mask_headwise(attn_map: torch.Tensor, r: float) -> torch.BoolTensor:
    """
    Build a mask per head so that each head keeps its top-r fraction of entries as True.

    Args:
    attn_map: Tensor of shape [H, S, S], attention scores (e.g. after softmax).
    r: float in (0,1), fraction of entries *per head* to keep True.

    Returns:
    mask: BoolTensor of shape [H, S, S], where for each head h,
            mask[h].float().mean() ≈ r.
    """
    H, S, _ = attn_map.shape
    mask = torch.zeros_like(attn_map, dtype=torch.bool)

    # process each head independently
    for h in range(H):
        head_scores = attn_map[h]                # [S, S]
        flat = head_scores.flatten()             # [S*S]
        n = flat.numel()
        k = int((1.0 - r) * n)                   # number of smallest to exclude

        if k == 0:
            mask[h] = True
            continue
        if k >= n:
            # nothing to keep
            continue

        # threshold = max of the k smallest scores
        threshold = torch.topk(flat, k, largest=False).values.max()

        # build head mask
        mask[h] = head_scores >= threshold

    return mask
