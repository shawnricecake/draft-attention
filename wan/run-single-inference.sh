
ckpt_dir="/mnt/localssd/wan/Wan2.1-T2V-14B"


export CUDA_VISIBLE_DEVICES=0

# --size 1280*768
# --size 768*512

python3 -u generate.py \
        --task t2v-14B \
        --size 768*512 \
        --ckpt_dir $ckpt_dir \
        --prompt "A giant panda is walking."

        # --prompt "warm colors dominate the room, with a focus on the tabby cat sitting contently in the center. the scene captures the fluffy orange tabby cat wearing a tiny virtual reality headset. the setting is a cozy living room, adorned with soft, warm lighting and a modern aesthetic. a plush sofa is visible in the background, along with a few lush potted plants, adding a touch of greenery. the cat's tail flicks curiously, as if engaging with an unseen virtual environment. its paws swipe at the air, indicating a playful and inquisitive nature, as it delves into the digital realm. the atmosphere is both whimsical and futuristic, highlighting the blend of analog and digital experiences."

        # --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."

