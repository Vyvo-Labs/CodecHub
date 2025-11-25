# CodecPlus: Audio codec models library
__version__ = "0.1.0"


def load_codec(model_name, hf_id=None, **kwargs):
    """Load a codec model by name

    Args:
        model_name: Codec model name ('wav_tokenizer', 'dac', 'longcat', 'snac', 'mimi', 'xcodec2', 'higgs_audio', 'tadicodec')
        hf_id: HuggingFace repository ID (for pretrained models)
        **kwargs: Additional model-specific parameters

    Returns:
        Loaded codec model

    Examples:
        # WavTokenizer from HuggingFace
        tokenizer = load_codec('wav_tokenizer', hf_id='Vyvo-Research/WavTokenizer-large-speech-320-v2')

        # LongCat from HuggingFace
        longcat = load_codec('longcat', hf_id='Vyvo-Research/LongCat-Audio-Codec', variant='24k_4codebooks')

        # SNAC from HuggingFace
        snac = load_codec('snac', hf_id='hubertsiuzdak/snac_32khz')

        # Mimi from HuggingFace
        mimi = load_codec('mimi', hf_id='kyutai/mimi')

        # XCodec2 from HuggingFace
        xcodec2 = load_codec('xcodec2', hf_id='NandemoGHS/Anime-XCodec2-44.1kHz-v2')

        # Higgs Audio from HuggingFace
        higgs = load_codec('higgs_audio', hf_id='bosonai/higgs-audio-v2-tokenizer')

        # TaDiCodec from HuggingFace
        tadicodec = load_codec('tadicodec', hf_id='amphion/TaDiCodec')

        # DAC from HuggingFace
        dac = load_codec('dac', hf_id='descript/dac_44khz')
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
            return DAC.from_pretrained(repo_id=hf_id)
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
        from codecplus.codecs.snac import SNAC

        if hf_id:
            # Load from HuggingFace
            device = kwargs.pop('device', None)
            return SNAC.from_pretrained(repo_id=hf_id, device=device)

        raise ValueError(
            "SNAC requires 'hf_id' parameter.\n"
            "Example: load_codec('snac', hf_id='hubertsiuzdak/snac_32khz')\n"
            "Available models:\n"
            "  - 'hubertsiuzdak/snac_24khz' (24kHz, 0.98 kbps, Speech)\n"
            "  - 'hubertsiuzdak/snac_32khz' (32kHz, 1.9 kbps, Music/SFX)\n"
            "  - 'hubertsiuzdak/snac_44khz' (44kHz, 2.6 kbps, Music/SFX)"
        )

    elif model_name == "mimi":
        from codecplus.codecs.mimi import Mimi

        if hf_id:
            # Load from HuggingFace
            return Mimi.from_pretrained(repo_id=hf_id)

        raise ValueError(
            "Mimi requires 'hf_id' parameter.\n"
            "Example: load_codec('mimi', hf_id='kyutai/mimi')\n"
            "Available models:\n"
            "  - 'kyutai/mimi' (24kHz, streaming capable)"
        )

    elif model_name == "xcodec2":
        from codecplus.codecs.xcodec2 import XCodec2

        if hf_id:
            # Load from HuggingFace
            device = kwargs.pop('device', None)
            return XCodec2.from_pretrained(repo_id=hf_id, device=device)

        raise ValueError(
            "XCodec2 requires 'hf_id' parameter.\n"
            "Example: load_codec('xcodec2', hf_id='NandemoGHS/Anime-XCodec2-44.1kHz-v2')\n"
            "Available models:\n"
            "  - 'NandemoGHS/Anime-XCodec2-44.1kHz-v2' (44.1kHz, anime speech)\n"
            "  - 'HKUSTAudio/xcodec2' (16kHz, speech)"
        )

    elif model_name == "higgs_audio":
        from codecplus.codecs.higgs_audio import HiggsAudio

        tokenizer_path = kwargs.pop('tokenizer_path', hf_id)

        if tokenizer_path is None:
            raise ValueError(
                "Higgs Audio requires 'tokenizer_path' or 'hf_id' parameter.\n"
                "Example: load_codec('higgs_audio', hf_id='bosonai/higgs-audio-v2-tokenizer')\n"
                "Or: load_codec('higgs_audio', tokenizer_path='bosonai/higgs-audio-v2-tokenizer')\n"
                "Available models:\n"
                "  - 'bosonai/higgs-audio-v2-tokenizer' (16kHz)"
            )

        return HiggsAudio.from_pretrained(
            tokenizer_path=tokenizer_path,
            **kwargs
        )

    elif model_name == "tadicodec":
        from codecplus.codecs.tadicodec import TaDiCodec

        model_path = kwargs.pop('model_path', hf_id)

        if model_path is None:
            # Default to official HuggingFace model
            model_path = 'amphion/TaDiCodec'

        device = kwargs.pop('device', None)

        return TaDiCodec.from_pretrained(
            model_path=model_path,
            device=device if device else ('cuda' if __import__('torch').cuda.is_available() else 'cpu')
        )

    else:
        raise ValueError(f"Unknown codec: {model_name}. Available: wav_tokenizer, dac, longcat, snac, mimi, xcodec2, higgs_audio, tadicodec")
