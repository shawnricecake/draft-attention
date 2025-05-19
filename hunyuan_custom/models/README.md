# Download Pretrained Models

All models are stored in `HunyuanCustom/models` by default, and the file structure is as follows
```shell
HunyuanCustom
  ├──models
  │  ├──README.md
  │  ├──hunyuancustom_720P
  │  │  ├──mp_rank_00_model_states.pt
  │  │  │──mp_rank_00_model_states_fp8.pt
  │  │  ├──mp_rank_00_model_states_fp8_map.pt
  ├  ├──vae_3d
  │  ├──openai_clip-vit-large-patch14
  │  ├──llava-llama-3-8b-v1_1
  ├──...
```

## Download HunyuanCustom model
To download the HunyuanCustom model, first install the huggingface-cli. (Detailed instructions are available [here](https://huggingface.co/docs/huggingface_hub/guides/cli).)

```shell
python -m pip install "huggingface_hub[cli]"
```

Then download the model using the following commands:

```shell
# Switch to the directory named 'HunyuanCustom'
cd HunyuanCustom
# Use the huggingface-cli tool to download HunyuanCustom model in HunyuanCustom/models dir.
# The download time may vary from 10 minutes to 1 hour depending on network conditions.
huggingface-cli download tencent/HunyuanCustom --local-dir ./
```
