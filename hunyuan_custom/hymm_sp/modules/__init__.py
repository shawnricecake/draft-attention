from .models import HYVideoDiffusionTransformer, HUNYUAN_VIDEO_CONFIG

def load_model(args, in_channels, out_channels, factor_kwargs):
    model = HYVideoDiffusionTransformer(
        args,
        in_channels=in_channels,
        out_channels=out_channels,
        **HUNYUAN_VIDEO_CONFIG[args.model],
        **factor_kwargs,
    )
    return model
