
export MODEL_BASE="./models"
export PYTHONPATH=./

torchrun --nnodes=1 --nproc_per_node=8 --master_port 29605 hymm_sp/sample_batch.py \
    --input './assets/images/seg_woman_01.png' \
    --pos-prompt "Realistic, High-quality. A woman is drinking coffee at a café." \
    --neg-prompt "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border." \
    --ckpt ${MODEL_BASE}"/hunyuancustom_720P/mp_rank_00_model_states.pt" \
    --video-size 768 1280 \
    --seed 1024 \
    --sample-n-frames 129 \
    --infer-steps 30 \
    --flow-shift-eval-video 13.0 \
    --save-path './results/sp_768p'

