# CodecPlus: Audio codec models library
__version__ = "0.1.0"


def load_codec(model_name, hf_id=None, **kwargs):
    """Load a codec model by name

    Args:
        model_name: Codec model name ('wav_tokenizer', 'dac')
        hf_id: HuggingFace repository ID (required for wav_tokenizer)
               Format: 'username/repository-name'
               Examples: 'Vyvo-Research/WavTokenizer-large-speech-320-v2'
        **kwargs: Additional model-specific parameters

    Returns:
        Loaded codec model

    Examples:
        # Download WavTokenizer from HuggingFace
        tokenizer = load_codec('wav_tokenizer',
                              hf_id='Vyvo-Research/WavTokenizer-large-speech-320-v2')

        # Load DAC codec
        dac = load_codec('dac', sample_rate=44100)
    """
    if model_name == "wav_tokenizer":
        from codecplus.codecs.wav_tokenizer.decoder.pretrained import WavTokenizer

        if hf_id is None:
            raise ValueError(
                "WavTokenizer requires 'hf_id' parameter.\n"
                "Examples:\n"
                "  - load_codec('wav_tokenizer', hf_id='Vyvo-Research/WavTokenizer-large-speech-320-v2')\n"
                "  - load_codec('wav_tokenizer', hf_id='username/repo-name')"
            )

        return WavTokenizer.from_pretrained(repo_id=hf_id)

    elif model_name == "dac":
        from codecplus.codecs.dac import DAC
        return DAC(**kwargs)

    else:
        raise ValueError(f"Unknown codec: {model_name}")
