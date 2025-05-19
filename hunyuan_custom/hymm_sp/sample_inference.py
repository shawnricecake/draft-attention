import math
import time
import torch
import random
from loguru import logger
from hymm_sp.diffusion import load_diffusion_pipeline
from hymm_sp.helpers import get_nd_rotary_pos_embed_new
from hymm_sp.inference import Inference
from hymm_sp.diffusion.schedulers import FlowMatchDiscreteScheduler


def align_to(value, alignment):
    return int(math.ceil(value / alignment) * alignment)

class HunyuanVideoSampler(Inference):
    def __init__(self, args, vae, vae_kwargs, text_encoder, model, text_encoder_2=None, pipeline=None,
                 device=0, logger=None):
        super().__init__(args, vae, vae_kwargs, text_encoder, model, text_encoder_2=text_encoder_2,
                         pipeline=pipeline,  device=device, logger=logger)
        
        self.args = args
        self.pipeline = load_diffusion_pipeline(
            args, 0, self.vae, self.text_encoder, self.text_encoder_2, self.model,
            device=self.device)
        print('load hunyuan model successful... ')

    def get_rotary_pos_embed(self, video_length, height, width, concat_dict={}):
        target_ndim = 3
        ndim = 5 - 2
        if '884' in self.args.vae:
            latents_size = [(video_length-1)//4+1 , height//8, width//8]
        else:
            latents_size = [video_length , height//8, width//8]

        if isinstance(self.model.patch_size, int):
            assert all(s % self.model.patch_size == 0 for s in latents_size), \
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.model.patch_size}), " \
                f"but got {latents_size}."
            rope_sizes = [s // self.model.patch_size for s in latents_size]
        elif isinstance(self.model.patch_size, list):
            assert all(s % self.model.patch_size[idx] == 0 for idx, s in enumerate(latents_size)), \
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.model.patch_size}), " \
                f"but got {latents_size}."
            rope_sizes = [s // self.model.patch_size[idx] for idx, s in enumerate(latents_size)]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
        head_dim = self.model.hidden_size // self.model.num_heads
        rope_dim_list = self.model.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed_new(rope_dim_list, 
                                                    rope_sizes, 
                                                    theta=self.args.rope_theta, 
                                                    use_real=True,
                                                    theta_rescale_factor=1,
                                                    concat_dict=concat_dict)
        return freqs_cos, freqs_sin

    @torch.no_grad()
    def predict(self, 
                prompt, 
                size=(720, 1280),
                video_length=129,
                seed=None,
                negative_prompt=None,
                infer_steps=50,
                guidance_scale=6.0,
                flow_shift=5.0,
                batch_size=1,
                num_videos_per_prompt=1,
                verbose=1,
                output_type="pil",
                **kwargs):
        """
        Predict the image from the given text.

        Args:
            prompt (str or List[str]): The input text.
            kwargs:
                size (int): The (height, width) of the output image/video. Default is (256, 256).
                video_length (int): The frame number of the output video. Default is 1.
                seed (int or List[str]): The random seed for the generation. Default is a random integer.
                negative_prompt (str or List[str]): The negative text prompt. Default is an empty string.
                infer_steps (int): The number of inference steps. Default is 100.
                guidance_scale (float): The guidance scale for the generation. Default is 6.0.
                num_videos_per_prompt (int): The number of videos per prompt. Default is 1.    
                verbose (int): 0 for no log, 1 for all log, 2 for fewer log. Default is 1.
                output_type (str): The output type of the image, can be one of `pil`, `np`, `pt`, `latent`.
                    Default is 'pil'.
        """
        
        out_dict = dict()

        # ---------------------------------
        # Prompt
        # ---------------------------------
        prompt_embeds = kwargs.get("prompt_embeds", None)
        attention_mask = kwargs.get("attention_mask", None)
        negative_prompt_embeds = kwargs.get("negative_prompt_embeds", None)
        negative_attention_mask = kwargs.get("negative_attention_mask", None)
        pixel_value_llava = kwargs.get("pixel_value_llava", None)
        uncond_pixel_value_llava = kwargs.get("uncond_pixel_value_llava", None)
        ref_latents = kwargs.get("ref_latents", None)
        uncond_ref_latents = kwargs.get("uncond_ref_latents", None)
        name = kwargs.get("name", None)
        cpu_offload = kwargs.get("cpu_offload", 0)
        use_deepcache = kwargs.get("use_deepcache", 1)
        denoise_strength = kwargs.get("denoise_strength", 1.0)
        init_latents = kwargs.get("init_latents", None)
        mask = kwargs.get("mask", None)
        if prompt is None:
            # prompt_embeds, attention_mask, negative_prompt_embeds and negative_attention_mask should not be None
            # pipeline will help to check this
            prompt = None
            negative_prompt = None
            batch_size = prompt_embeds.shape[0]
            assert prompt_embeds is not None
        else:
            # prompt_embeds, attention_mask, negative_prompt_embeds and negative_attention_mask should be None
            # pipeline will help to check this
            if isinstance(prompt, str):
                batch_size = 1
                prompt = [prompt]
            elif isinstance(prompt, (list, tuple)):
                batch_size = len(prompt)
            else:
                raise ValueError(f"Prompt must be a string or a list of strings, got {prompt}.")

            if negative_prompt is None:
                negative_prompt = [""] * batch_size
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            
        # ---------------------------------
        # Other arguments
        # ---------------------------------
        scheduler = FlowMatchDiscreteScheduler(shift=flow_shift,
                                                reverse=self.args.flow_reverse,
                                                solver=self.args.flow_solver,
                                                )
        self.pipeline.scheduler = scheduler

        # ---------------------------------
        # Random seed
        # ---------------------------------
        
        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [random.randint(0, 1_000_000) for _ in range(batch_size * num_videos_per_prompt)]
        elif isinstance(seed, int):
            seeds = [seed + i for _ in range(batch_size) for i in range(num_videos_per_prompt)]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [int(seed[i]) + j for i in range(batch_size) for j in range(num_videos_per_prompt)]
            elif len(seed) == batch_size * num_videos_per_prompt:
                seeds = [int(s) for s in seed]
            else:
                raise ValueError(
                    f"Length of seed must be equal to number of prompt(batch_size) or "
                    f"batch_size * num_videos_per_prompt ({batch_size} * {num_videos_per_prompt}), got {seed}."
                )
        else:
            raise ValueError(f"Seed must be an integer, a list of integers, or None, got {seed}.")
        generator = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]
        
        # ---------------------------------
        # Image/Video size and frame
        # ---------------------------------
        size = self.parse_size(size)
        target_height = align_to(size[0], 16)
        target_width = align_to(size[1], 16)
        target_video_length = video_length

        out_dict['size'] = (target_height, target_width)
        out_dict['video_length'] = target_video_length
        out_dict['seeds'] = seeds
        out_dict['negative_prompt'] = negative_prompt
        # ---------------------------------
        # Build RoPE
        # ---------------------------------

        concat_dict = {'mode': 'timecat-w', 'bias': -1} 
        freqs_cos, freqs_sin = self.get_rotary_pos_embed(target_video_length, target_height, target_width, concat_dict)
        
        n_tokens = freqs_cos.shape[0]
        
        # ---------------------------------
        # Inference
        # ---------------------------------
        if verbose == 1:
            debug_str = f"""
                  size: {out_dict['size']}
          video_length: {target_video_length}
                prompt: {prompt}
                name: {name}
            neg_prompt: {negative_prompt}
                  seed: {seed}
           infer_steps: {infer_steps}
      denoise_strength: {denoise_strength}
         use_deepcache: {use_deepcache}
           cpu_offload: {cpu_offload}
 num_images_per_prompt: {num_videos_per_prompt}
        guidance_scale: {guidance_scale}
              n_tokens: {n_tokens}
            flow_shift: {flow_shift}"""
            self.logger.info(debug_str)

        start_time = time.time()
        samples = self.pipeline(prompt=prompt,   
                                name=name,                                 
                                height=target_height,
                                width=target_width,
                                video_length=target_video_length,
                                num_inference_steps=infer_steps,
                                guidance_scale=guidance_scale,
                                negative_prompt=negative_prompt,
                                num_videos_per_prompt=num_videos_per_prompt,
                                generator=generator,
                                prompt_embeds=prompt_embeds,
                                ref_latents=ref_latents,
                                latents=init_latents,
                                denoise_strength=denoise_strength,
                                mask=mask,
                                uncond_ref_latents=uncond_ref_latents,
                                pixel_value_llava=pixel_value_llava,
                                uncond_pixel_value_llava=uncond_pixel_value_llava,
                                ip_cfg_scale=self.args.ip_cfg_scale, 
                                use_deepcache=use_deepcache,
                                attention_mask=attention_mask,
                                negative_prompt_embeds=negative_prompt_embeds,
                                negative_attention_mask=negative_attention_mask,
                                output_type=output_type,
                                freqs_cis=(freqs_cos, freqs_sin),
                                n_tokens=n_tokens,
                                data_type='video' if target_video_length > 1 else 'image',
                                is_progress_bar=True,
                                vae_ver=self.args.vae,
                                enable_tiling=self.args.vae_tiling,
                                cpu_offload=cpu_offload, 
                                )[0]
        if samples is None:
            return None
        out_dict['samples'] = samples
        out_dict["prompts"] = prompt
        gen_time = time.time() - start_time
        logger.info(f"Success, time: {gen_time}")
        return out_dict
    
