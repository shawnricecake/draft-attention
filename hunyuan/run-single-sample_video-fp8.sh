export CUDA_VISIBLE_DEVICES=0

export MODEL_BASE="/mnt/localssd/hunyuan-video/ckpts/"
#dit_weight="/mnt/localssd/hunyuan-video/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"
seed=42

dit_weight="/mnt/localssd/hunyuan-video/fp8/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt"

python3 sample_video.py \
    --model-base $MODEL_BASE \
    --dit-weight $dit_weight \
    --seed $seed \
    --video-size 768 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --use-cpu-offload \
    --use-fp8 \
    --save-path ./z-results
