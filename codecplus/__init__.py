# CodecPlus: Audio codec models library
__version__ = "0.1.0"


def load_codec(model_name, hf_id=None, **kwargs):
    """Load a codec model by name

    Args:
        model_name: Codec model name ('wav_tokenizer', 'dac', 'longcat', 'snac', 'mimi')
        hf_id: HuggingFace repository ID (for pretrained models)
        **kwargs: Additional model-specific parameters

    Returns:
        Loaded codec model

    Examples:
        # WavTokenizer from HuggingFace
        tokenizer = load_codec('wav_tokenizer', hf_id='Vyvo-Research/WavTokenizer-large-speech-320-v2')

        # LongCat from HuggingFace
        longcat = load_codec('longcat', hf_id='Vyvo-Research/LongCat-Audio-Codec', variant='24k_4codebooks')

        # DAC (no pretrained yet)
        dac = load_codec('dac', sample_rate=44100)
    """
    if model_name == "wav_tokenizer":
        from codecplus.codecs.wav_tokenizer.decoder.pretrained import WavTokenizer

        if hf_id is None:
            raise ValueError(
                "WavTokenizer requires 'hf_id' parameter.\n"
                "Example: load_codec('wav_tokenizer', hf_id='Vyvo-Research/WavTokenizer-large-speech-320-v2')"
            )
        return WavTokenizer.from_pretrained(repo_id=hf_id)

    elif model_name == "dac":
        from codecplus.codecs.dac import DAC

        if hf_id:
            # Future: DAC from HuggingFace
            raise NotImplementedError("DAC from_pretrained not yet implemented")
        return DAC(**kwargs)

    elif model_name == "longcat":
        from codecplus.codecs.longcat import LongCatCodec

        if hf_id:
            # Load from HuggingFace
            # Extract variant if provided (default to 24k_4codebooks)
            variant = kwargs.pop('variant', kwargs.pop('model_name', '24k_4codebooks'))
            return LongCatCodec.from_pretrained(hf_id=hf_id, model_name=variant, **kwargs)

        # Load from local configs
        required_params = ['encoder_config', 'decoder_config']
        missing = [p for p in required_params if p not in kwargs]
        if missing:
            raise ValueError(
                f"LongCat requires either 'hf_id' or {', '.join(missing)}.\n"
                "Examples:\n"
                "  - load_codec('longcat', hf_id='Vyvo-Research/LongCat-Audio-Codec', variant='24k_4codebooks')\n"
                "  - load_codec('longcat', encoder_config='...', decoder_config='...')"
            )
        return LongCatCodec(**kwargs)

    elif model_name == "snac":
        # Placeholder for SNAC
        if hf_id:
            raise NotImplementedError("SNAC from_pretrained not yet implemented")
        raise NotImplementedError("SNAC codec not yet implemented")

    elif model_name == "mimi":
        # Placeholder for Mimi
        if hf_id:
            raise NotImplementedError("Mimi from_pretrained not yet implemented")
        raise NotImplementedError("Mimi codec not yet implemented")

    else:
        raise ValueError(f"Unknown codec: {model_name}. Available: wav_tokenizer, dac, longcat, snac, mimi")
