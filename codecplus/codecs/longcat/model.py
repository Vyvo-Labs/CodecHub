"""LongCat Audio Codec implementation for CodecHub"""

import os
import math
from typing import Optional, Tuple, Union, List
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio

from codecplus.codecs.longcat.networks.semantic_codec.LongCatAudioCodec_model import (
    LongCatAudioCodecEncoder,
    LongCatAudioCodecDecoder,
)
from codecplus.utils.download import download_from_hf, get_cache_dir


class LongCatCodec(nn.Module):
    """
    LongCat Audio Codec wrapper for CodecHub.

    A semantic-acoustic neural audio codec that generates semantic and acoustic
    tokens in parallel, enabling high-fidelity audio reconstruction at extremely
    low bitrates (16.6Hz frame rate).

    Features:
    - Dual-path encoding: semantic tokens (content) + acoustic tokens (quality)
    - Multi-rate decoding: supports 16kHz and 24kHz output
    - Streaming capability: low-latency processing
    - Configurable quality: 1-4 acoustic codebooks

    Args:
        encoder_config: Path to encoder YAML config or dict
        decoder_config: Path to decoder YAML config or dict (16k or 24k)
        device: Device to run model on ('cuda' or 'cpu')
        n_acoustic_codebooks: Number of acoustic codebooks (1-4, default: 2)

    Example:
        >>> codec = LongCatCodec(
        ...     encoder_config='configs/encoder.yaml',
        ...     decoder_config='configs/decoder_24k.yaml',
        ...     n_acoustic_codebooks=2
        ... )
        >>> # Encode audio
        >>> semantic_codes, acoustic_codes = codec.encode(audio, sr=16000)
        >>> # Decode audio
        >>> reconstructed = codec.decode(semantic_codes, acoustic_codes)
    """

    def __init__(
        self,
        encoder_config: Union[str, dict],
        decoder_config: Union[str, dict],
        device: Optional[torch.device] = None,
        n_acoustic_codebooks: int = 2,
    ):
        super().__init__()

        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.n_acoustic_codebooks = n_acoustic_codebooks

        # Load configurations
        self.encoder_config = self._load_config(encoder_config)
        self.decoder_config = self._load_config(decoder_config)

        # Initialize encoder and decoder
        self.encoder = self._load_encoder(self.encoder_config)
        self.decoder = self._load_decoder(self.decoder_config)

        # Set to eval mode
        self.encoder.eval()
        self.decoder.eval()

    def _load_config(self, config: Union[str, dict]) -> dict:
        """Load YAML configuration file or return dict as-is"""
        if isinstance(config, dict):
            return config

        if not os.path.exists(config):
            raise FileNotFoundError(f"Configuration file not found: {config}")

        with open(config, 'r') as f:
            return yaml.safe_load(f)

    def _load_encoder(self, config: dict) -> LongCatAudioCodecEncoder:
        """Initialize and load encoder model"""
        args = config['codec_config']
        ckpt_path = args['ckpt_path']

        # Initialize encoder
        model = LongCatAudioCodecEncoder(
            encoder_dim=args['encoder_dim'],
            encoder_rates=args['codec_enc_ratios'],
            latent_dim=args['codec_dimension'],
            n_codebooks=args['codec_codebooks'],
            codebook_size=args['codec_codebook_size'],
            codebook_dim=args['codec_codebook_search_dim'],
            input_sample_rate=args['input_sample_rate'],
            semantic_tokenizer_type=args['semantic_tokenizer_type'],
        ).to(self.device)

        # Load checkpoint if exists
        if os.path.exists(ckpt_path):
            print(f"Loading encoder checkpoint from: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"Warning: Encoder checkpoint not found at {ckpt_path}")
            print(f"  You can download it using: python -m codecplus.codecs.longcat.download --model encoder")

        return model

    def _load_decoder(self, config: dict) -> LongCatAudioCodecDecoder:
        """Initialize and load decoder model"""
        args = config['codec_config']
        ckpt_path = args['ckpt_path']

        # Initialize decoder
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
        ).to(self.device)

        # Load checkpoint if exists
        if os.path.exists(ckpt_path):
            print(f"Loading decoder checkpoint from: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(state_dict, strict=True)
        else:
            print(f"Warning: Decoder checkpoint not found at {ckpt_path}")
            decoder_type = args.get('decoder_type', '24k')
            codebooks = args.get('codec_codebooks', 4)
            print(f"  You can download it using: python -m codecplus.codecs.longcat.download --model decoder_{decoder_type}_{codebooks}codebooks")

        return model

    @property
    def sample_rate(self) -> int:
        """Get decoder output sample rate"""
        return self.decoder.output_rate

    @property
    def input_sample_rate(self) -> int:
        """Get encoder input sample rate"""
        return self.encoder.input_sample_rate

    def encode(
        self,
        audio: torch.Tensor,
        sample_rate: Optional[int] = None,
        n_acoustic_codebooks: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode audio to semantic and acoustic tokens.

        Args:
            audio: Input audio tensor [B, 1, T] or [B, T]
            sample_rate: Sample rate of input audio (default: encoder's input_sample_rate)
            n_acoustic_codebooks: Number of acoustic codebooks to use (default: self.n_acoustic_codebooks)

        Returns:
            Tuple of (semantic_codes, acoustic_codes):
                - semantic_codes: [B, T_codes] semantic tokens
                - acoustic_codes: [B, N_q, T_codes] acoustic tokens or None
        """
        # Ensure 3D tensor [B, 1, T]
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        elif audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)

        # Move to device
        audio = audio.to(self.device)

        # Use default n_acoustic_codebooks if not specified
        if n_acoustic_codebooks is None:
            n_acoustic_codebooks = self.n_acoustic_codebooks

        with torch.no_grad():
            codes = self.encoder(audio, sample_rate, n_acoustic_codebooks=n_acoustic_codebooks)
            semantic_codes, acoustic_codes = codes[0], codes[1]

        return semantic_codes, acoustic_codes

    def decode(
        self,
        semantic_codes: torch.Tensor,
        acoustic_codes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode semantic and acoustic tokens to audio.

        Args:
            semantic_codes: Semantic tokens [B, T_codes]
            acoustic_codes: Acoustic tokens [B, N_q, T_codes] (optional)

        Returns:
            Reconstructed audio waveform [B, 1, T_audio]
        """
        semantic_codes = semantic_codes.to(self.device)
        if acoustic_codes is not None:
            acoustic_codes = acoustic_codes.to(self.device)

        with torch.no_grad():
            audio = self.decoder(semantic_codes, acoustic_codes)

        return audio

    def forward(
        self,
        audio: torch.Tensor,
        sample_rate: Optional[int] = None
    ) -> torch.Tensor:
        """
        Encode and decode audio (full reconstruction).

        Args:
            audio: Input audio tensor [B, 1, T] or [B, T]
            sample_rate: Sample rate of input audio

        Returns:
            Reconstructed audio waveform [B, 1, T_audio]
        """
        semantic_codes, acoustic_codes = self.encode(audio, sample_rate)
        return self.decode(semantic_codes, acoustic_codes)

    @torch.no_grad()
    def encode_file(self, audio_path: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode audio file to tokens.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (semantic_codes, acoustic_codes)
        """
        # Load audio
        wav, sr = torchaudio.load(audio_path)

        # Convert to mono if needed
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        # Add batch dimension
        wav = wav.unsqueeze(0)

        return self.encode(wav, sr)

    @torch.no_grad()
    def decode_to_file(
        self,
        semantic_codes: torch.Tensor,
        acoustic_codes: Optional[torch.Tensor],
        output_path: str
    ):
        """
        Decode tokens and save to file.

        Args:
            semantic_codes: Semantic tokens
            acoustic_codes: Acoustic tokens
            output_path: Path to save reconstructed audio
        """
        audio = self.decode(semantic_codes, acoustic_codes)

        # Remove batch dimension if present
        if audio.dim() == 3:
            audio = audio.squeeze(0)

        torchaudio.save(output_path, audio.cpu(), self.sample_rate)
        print(f"Saved reconstructed audio to: {output_path}")

    @torch.no_grad()
    def reconstruct_file(self, input_path: str, output_path: str):
        """
        Reconstruct audio file (encode + decode).

        Args:
            input_path: Path to input audio file
            output_path: Path to save reconstructed audio
        """
        semantic_codes, acoustic_codes = self.encode_file(input_path)
        self.decode_to_file(semantic_codes, acoustic_codes, output_path)

    def get_codes_with_lengths(
        self,
        audio_batch: torch.Tensor,
        audio_lengths: torch.Tensor,
        n_acoustic_codebooks: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract tokens from batched variable-length audio.

        Args:
            audio_batch: Padded audio batch [B, 1, T]
            audio_lengths: Original lengths of each audio [B]
            n_acoustic_codebooks: Number of acoustic codebooks

        Returns:
            Tuple of (semantic_codes, acoustic_codes, codes_lengths)
        """
        if n_acoustic_codebooks is None:
            n_acoustic_codebooks = self.n_acoustic_codebooks

        audio_batch = audio_batch.to(self.device)
        audio_lengths = audio_lengths.to(self.device)

        # Preprocess
        audio_batch = self.encoder.preprocess(audio_batch, self.input_sample_rate)

        with torch.no_grad():
            # Get semantic codes with lengths
            semantic_codes, codes_lengths = self.encoder.get_semantic_codes_with_lengths(
                audio_batch, audio_lengths
            )

            # Get acoustic codes with lengths
            acoustic_codes, _ = self.encoder.get_acoustic_codes_with_lengths(
                audio_batch, audio_lengths, n_acoustic_codebooks=n_acoustic_codebooks
            )

        return semantic_codes, acoustic_codes, codes_lengths

    @classmethod
    def from_pretrained(
        cls,
        hf_id: str = "Vyvo-Research/LongCat-Audio-Codec",
        model_name: str = "24k_4codebooks",
        device: Optional[torch.device] = None,
        n_acoustic_codebooks: int = 2,
    ) -> "LongCatCodec":
        """
        Load pretrained LongCat codec from HuggingFace

        Args:
            hf_id: HuggingFace repository ID
            model_name: Decoder variant (16k_4codebooks, 24k_2codebooks, 24k_4codebooks, 24k_4codebooks_aug_sft)
            device: Device to run on
            n_acoustic_codebooks: Number of acoustic codebooks (1-4)

        Returns:
            Initialized LongCatCodec model

        Example:
            >>> codec = LongCatCodec.from_pretrained(model_name="24k_4codebooks")
        """
        valid_models = ["16k_4codebooks", "24k_2codebooks", "24k_4codebooks", "24k_4codebooks_aug_sft"]
        if model_name not in valid_models:
            raise ValueError(f"Invalid model_name: {model_name}. Choose from: {valid_models}")

        print(f"\nLoading LongCat pretrained model: {model_name}")
        print("=" * 60)

        # Download encoder checkpoint
        encoder_ckpt = download_from_hf(
            repo_id=hf_id,
            filename="ckpts/LongCatAudioCodec_encoder.pt",
            codec_name="longcat",
        )

        # Download encoder CMVN file
        encoder_cmvn = download_from_hf(
            repo_id=hf_id,
            filename="ckpts/LongCatAudioCodec_encoder_cmvn.npy",
            codec_name="longcat",
        )

        # Download decoder checkpoint
        decoder_ckpt = download_from_hf(
            repo_id=hf_id,
            filename=f"ckpts/LongCatAudioCodec_decoder_{model_name}.pt",
            codec_name="longcat",
        )

        # Get config paths
        module_dir = Path(__file__).parent.parent.parent.parent
        encoder_config_path = module_dir / f"configs/longcat/LongCatAudioCodec_encoder.yaml"
        decoder_config_path = module_dir / f"configs/longcat/LongCatAudioCodec_decoder_{model_name}.yaml"

        # Load and update configs
        with open(encoder_config_path, 'r') as f:
            encoder_config = yaml.safe_load(f)
        encoder_config['codec_config']['ckpt_path'] = encoder_ckpt

        with open(decoder_config_path, 'r') as f:
            decoder_config = yaml.safe_load(f)
        decoder_config['codec_config']['ckpt_path'] = decoder_ckpt

        # Update semantic tokenizer CMVN path
        semantic_config_path = module_dir / "codecplus/codecs/longcat/semantic_tokenizer_general/configs/semantic_tokenizer.infer.yaml"
        if semantic_config_path.exists():
            with open(semantic_config_path, 'r') as f:
                semantic_cfg = yaml.safe_load(f)
            semantic_cfg['feature']['cmvn_file'] = str(encoder_cmvn)
            with open(semantic_config_path, 'w') as f:
                yaml.dump(semantic_cfg, f)

        print("=" * 60)

        return cls(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            device=device,
            n_acoustic_codebooks=n_acoustic_codebooks,
        )


def load_longcat_codec(
    encoder_config: str,
    decoder_config: str,
    device: Optional[torch.device] = None,
    n_acoustic_codebooks: int = 2,
) -> LongCatCodec:
    """
    Helper function to load LongCat codec.

    Args:
        encoder_config: Path to encoder YAML configuration
        decoder_config: Path to decoder YAML configuration
        device: Device to run on
        n_acoustic_codebooks: Number of acoustic codebooks (1-4)

    Returns:
        Initialized LongCatCodec model

    Example:
        >>> codec = load_longcat_codec(
        ...     encoder_config='configs/encoder.yaml',
        ...     decoder_config='configs/decoder_24k.yaml',
        ... )
    """
    return LongCatCodec(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        device=device,
        n_acoustic_codebooks=n_acoustic_codebooks,
    )
