from transformers import PretrainedConfig

class BigCodecConfig(PretrainedConfig):
    model_type = "xcodec"

    def __init__(
        self,
        # 下面这些只是示例超参
        semantic_hidden_size=1024,
        codec_encoder_hidden_size=1024,
        codec_decoder_hidden_size=1024,
        use_vocos=True,
        sample_rate=16000,
        hop_length=320,
        upsample_factors=None,
        upsample_kernel_sizes=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.semantic_hidden_size = semantic_hidden_size
        self.codec_encoder_hidden_size = codec_encoder_hidden_size
        self.codec_decoder_hidden_size = codec_decoder_hidden_size
        self.use_vocos = use_vocos

        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.upsample_factors = list(upsample_factors or [])
        self.upsample_kernel_sizes = list(upsample_kernel_sizes or [])
