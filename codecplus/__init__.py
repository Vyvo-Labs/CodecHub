# CodecPlus: Audio codec models library
__version__ = "0.1.0"


def load_codec(model_name, hf_id=None, **kwargs):
    """Load a codec model by name

    Args:
        model_name: Codec model name ('wav_tokenizer', 'dac', 'mimi', 'snac')
        hf_id: HuggingFace repository ID (required for all models)
               Format: 'username/repository-name'

               WavTokenizer examples:
                 - 'Vyvo-Research/WavTokenizer-large-speech-320-v2'

               DAC available models (from Transformers):
                 - 'descript/dac_16khz' (16kHz)
                 - 'descript/dac_24khz' (24kHz)
                 - 'descript/dac_44khz' (44.1kHz)

               Mimi available models (from Transformers):
                 - 'kyutai/mimi' (24kHz, streaming capable)

               SNAC available models:
                 - 'hubertsiuzdak/snac_24khz' (24kHz, 0.98 kbps, Speech)
                 - 'hubertsiuzdak/snac_32khz' (32kHz, 1.9 kbps, Music/SFX)
                 - 'hubertsiuzdak/snac_44khz' (44kHz, 2.6 kbps, Music/SFX)
        **kwargs: Additional model-specific parameters (reserved for future use)

    Returns:
        Loaded codec model

    Examples:
        # Download WavTokenizer from HuggingFace
        tokenizer = load_codec('wav_tokenizer',
                              hf_id='Vyvo-Research/WavTokenizer-large-speech-320-v2')

        # Download DAC from HuggingFace (uses Transformers library)
        dac = load_codec('dac', hf_id='descript/dac_16khz')
        dac = load_codec('dac', hf_id='descript/dac_44khz')

        # Download Mimi from HuggingFace (uses Transformers library)
        mimi = load_codec('mimi', hf_id='kyutai/mimi')

        # Download SNAC from HuggingFace
        snac = load_codec('snac', hf_id='hubertsiuzdak/snac_32khz')
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

        if hf_id is None:
            raise ValueError(
                "DAC requires 'hf_id' parameter.\n"
                "Available models:\n"
                "  - 'descript/dac_16khz' (16kHz)\n"
                "  - 'descript/dac_24khz' (24kHz)\n"
                "  - 'descript/dac_44khz' (44.1kHz)\n"
                "Examples:\n"
                "  - load_codec('dac', hf_id='descript/dac_16khz')\n"
                "  - load_codec('dac', hf_id='descript/dac_44khz')"
            )

        return DAC.from_pretrained(repo_id=hf_id)

    elif model_name == "mimi":
        from codecplus.codecs.mimi import Mimi

        if hf_id is None:
            raise ValueError(
                "Mimi requires 'hf_id' parameter.\n"
                "Available models:\n"
                "  - 'kyutai/mimi' (24kHz, streaming capable)\n"
                "Examples:\n"
                "  - load_codec('mimi', hf_id='kyutai/mimi')"
            )

        return Mimi.from_pretrained(repo_id=hf_id)

    elif model_name == "snac":
        from codecplus.codecs.snac import SNAC

        if hf_id is None:
            raise ValueError(
                "SNAC requires 'hf_id' parameter.\n"
                "Available models:\n"
                "  - 'hubertsiuzdak/snac_24khz' (24kHz, 0.98 kbps, Speech)\n"
                "  - 'hubertsiuzdak/snac_32khz' (32kHz, 1.9 kbps, Music/SFX)\n"
                "  - 'hubertsiuzdak/snac_44khz' (44kHz, 2.6 kbps, Music/SFX)\n"
                "Examples:\n"
                "  - load_codec('snac', hf_id='hubertsiuzdak/snac_32khz')\n"
                "  - load_codec('snac', hf_id='hubertsiuzdak/snac_24khz')"
            )

        # Extract device from kwargs if provided
        device = kwargs.get('device', None)
        return SNAC.from_pretrained(repo_id=hf_id, device=device)

    else:
        raise ValueError(f"Unknown codec: {model_name}")
