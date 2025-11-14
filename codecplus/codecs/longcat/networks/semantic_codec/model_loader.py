import os
import torch
from typing import Dict
import yaml
from codecplus.codecs.longcat.networks.semantic_codec.LongCatAudioCodec_model import LongCatAudioCodecEncoder, LongCatAudioCodecDecoder



def load_yaml_config(config_path: str)->Dict:
    """
    Loads a YAML configuration file and returns its content.

    Parameters
    ----------
    config_path : str
        The path to the YAML configuration file.

    Returns
    -------
    Dict
        The content of the YAML file as a dictionary.
        
    Raises
    ------
    FileNotFoundError
        If the configuration file is not found at the specified path.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config



def load_encoder(config_path: str, device: torch.device) -> LongCatAudioCodecEncoder:
    """
    Loads, initializes, and returns an Encoder model based on a configuration file.

    Parameters
    ----------
    config_path : str
        The path to the Encoder's YAML configuration file.
    device : torch.device
        The device to load the model onto (e.g., torch.device("cuda")).

    Returns
    -------
    LongCatAudioCodecEncoder
        The loaded Encoder model, set to evaluation mode.
    """
    print(f"\n--- Loading Encoder ---")
    print(f"  - Loading configuration from: {config_path}")

    config = load_yaml_config(config_path)
    args = config['codec_config']
    ckpt_path = args['ckpt_path']
    
    # Initialize the Encoder model
    model = LongCatAudioCodecEncoder(
        encoder_dim=args['encoder_dim'],
        encoder_rates=args['codec_enc_ratios'],
        latent_dim=args['codec_dimension'],
        n_codebooks=args['codec_codebooks'],
        codebook_size=args['codec_codebook_size'],
        codebook_dim=args['codec_codebook_search_dim'],
        input_sample_rate=args['input_sample_rate'],
        semantic_tokenizer_type=args['semantic_tokenizer_type'],
    ).to(device)
    
    # Load pretrained weights
    print(f"  - Loading checkpoint from: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    
    # Set the model to evaluation mode
    model.eval()
    
    print("  - Encoder loaded successfully!")

    return model



def load_decoder(config_path: str, device: torch.device) -> LongCatAudioCodecDecoder:
    """
    Loads, initializes, and returns a Decoder model based on a configuration file.

    Parameters
    ----------
    config_path : str
        The path to the Decoder's YAML configuration file.
    device : torch.device
        The device to load the model onto (e.g., torch.device("cuda")).

    Returns
    -------
    LongCatAudioCodecDecoder
        The loaded Decoder model, set to evaluation mode.
    """
    print(f"\n--- Loading Decoder ---")
    print(f"  - Loading configuration from: {config_path}")

    config = load_yaml_config(config_path)
    args = config['codec_config']
    ckpt_path = args['ckpt_path']
        
    # Initialize the Decoder model
    model = LongCatAudioCodecDecoder(
        latent_dim=args['codec_dimension'],
        decoder_dim=args['decoder_dim'],
        decoder_rates=args['codec_dec_ratios'],
        semantic_dim=args['semantic_dim'],
        decoder_type=args['decoder_type'],
        n_codebooks=args['codec_codebooks'],
        codebook_size=args['codec_codebook_size'],
        codebook_dim=args['codec_codebook_search_dim'],
        semantic_token_nums=args['semantic_token_nums'],
    ).to(device)

    # Load pretrained weights
    print(f"  - Loading checkpoint from: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    
    # Set the model to evaluation mode
    model.eval()

    print(f"  - Decoder loaded successfully!")

    return model