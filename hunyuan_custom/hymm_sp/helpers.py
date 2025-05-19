import torch
from typing import Union, List
from hymm_sp.modules.posemb_layers import get_1d_rotary_pos_embed, get_meshgrid_nd

from itertools import repeat
import collections.abc


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            x = tuple(x)
            if len(x) == 1:
                x = tuple(repeat(x[0], n))
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)

def get_rope_freq_from_size(latents_size, ndim, target_ndim, args,
                            rope_theta_rescale_factor: Union[float, List[float]]=1.0,
                            rope_interpolation_factor: Union[float, List[float]]=1.0,
                            concat_dict={}):
                            
    if isinstance(args.patch_size, int):
        assert all(s % args.patch_size == 0 for s in latents_size), \
            f"Latent size(last {ndim} dimensions) should be divisible by patch size({args.patch_size}), " \
            f"but got {latents_size}."
        rope_sizes = [s // args.patch_size for s in latents_size]
    elif isinstance(args.patch_size, list):
        assert all(s % args.patch_size[idx] == 0 for idx, s in enumerate(latents_size)), \
            f"Latent size(last {ndim} dimensions) should be divisible by patch size({args.patch_size}), " \
            f"but got {latents_size}."
        rope_sizes = [s // args.patch_size[idx] for idx, s in enumerate(latents_size)]

    if len(rope_sizes) != target_ndim:
        rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
    head_dim = args.hidden_size // args.num_heads
    rope_dim_list = args.rope_dim_list
    if rope_dim_list is None:
        rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
    assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
    freqs_cos, freqs_sin = get_nd_rotary_pos_embed_new(rope_dim_list, 
                                                    rope_sizes, 
                                                    theta=args.rope_theta, 
                                                    use_real=True,
                                                    theta_rescale_factor=rope_theta_rescale_factor,
                                                    interpolation_factor=rope_interpolation_factor,
                                                    concat_dict=concat_dict)
    return freqs_cos, freqs_sin
    
def get_nd_rotary_pos_embed_new(rope_dim_list, start, *args, theta=10000., use_real=False, 
                            theta_rescale_factor: Union[float, List[float]]=1.0,
                            interpolation_factor: Union[float, List[float]]=1.0,
                            concat_dict={}
                            ):

    grid = get_meshgrid_nd(start, *args, dim=len(rope_dim_list))   # [3, W, H, D] / [2, W, H]
    if len(concat_dict)<1:
        pass
    else:
        if concat_dict['mode']=='timecat':
            bias = grid[:,:1].clone()
            bias[0] = concat_dict['bias']*torch.ones_like(bias[0])
            grid = torch.cat([bias, grid], dim=1)
            
        elif concat_dict['mode']=='timecat-w': 
            bias = grid[:,:1].clone()
            bias[0] = concat_dict['bias']*torch.ones_like(bias[0])
            bias[2] += start[-1]    ## ref https://github.com/Yuanshi9815/OminiControl/blob/main/src/generate.py#L178
            grid = torch.cat([bias, grid], dim=1)
    if isinstance(theta_rescale_factor, int) or isinstance(theta_rescale_factor, float):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
    assert len(theta_rescale_factor) == len(rope_dim_list), "len(theta_rescale_factor) should equal to len(rope_dim_list)"

    if isinstance(interpolation_factor, int) or isinstance(interpolation_factor, float):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)
    assert len(interpolation_factor) == len(rope_dim_list), "len(interpolation_factor) should equal to len(rope_dim_list)"

    # use 1/ndim of dimensions to encode grid_axis
    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(rope_dim_list[i], grid[i].reshape(-1), theta, use_real=use_real,
                                      theta_rescale_factor=theta_rescale_factor[i],
                                      interpolation_factor=interpolation_factor[i])    # 2 x [WHD, rope_dim_list[i]]
        
        embs.append(emb)

    if use_real:
        cos = torch.cat([emb[0] for emb in embs], dim=1)    # (WHD, D/2)
        sin = torch.cat([emb[1] for emb in embs], dim=1)    # (WHD, D/2)
        return cos, sin
    else:
        emb = torch.cat(embs, dim=1)    # (WHD, D/2)
        return emb