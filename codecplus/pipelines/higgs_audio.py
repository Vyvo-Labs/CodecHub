"""
Higgs Audio Tokenizer codec integration for CodecPlus.

This module provides a wrapper around the Higgs Audio Tokenizer for audio compression
and reconstruction using neural audio codec.
"""

import torch
import torchaudio
import numpy as np
from typing import Optional, Union
from pathlib import Path

from codecplus.boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer


class HiggsAudio:
    """
    Higgs Audio Tokenizer codec for audio encoding and decoding.

    This class wraps the HiggsAudioTokenizer to provide a simple codec interface
    for compressing and reconstructing audio using neural audio compression.

    Examples:
        # Load from HuggingFace
        higgs = HiggsAudio.from_pretrained('bosonai/higgs-audio-v2-tokenizer')

        # Encode audio to VQ codes
        codes = higgs.encode(audio_array, sr=16000)

        # Decode VQ codes back to audio
        reconstructed = higgs.decode(codes)

        # Save reconstructed audio
        higgs.save_audio(reconstructed, 'output.wav')
    """

    def __init__(
        self,
        tokenizer_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize Higgs Audio Tokenizer.

        Args:
            tokenizer_path: Path or HuggingFace model ID for the audio tokenizer
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.tokenizer_path = tokenizer_path
        self.device = device

        # Load the audio tokenizer
        self.tokenizer = load_higgs_audio_tokenizer(tokenizer_path, device=device)
        self.tokenizer.eval()

    @classmethod
    def from_pretrained(
        cls,
        tokenizer_path: str = "bosonai/higgs-audio-v2-tokenizer",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Load a pretrained Higgs Audio Tokenizer.

        Args:
            tokenizer_path: Path or HuggingFace model ID for the audio tokenizer
            device: Device to run the model on ('cuda' or 'cpu')

        Returns:
            HiggsAudio instance
        """
        return cls(tokenizer_path=tokenizer_path, device=device)

    def encode(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        sr: Optional[int] = None,
        loudness_normalize: bool = False,
        loudness_threshold: float = -23.0,
    ) -> torch.Tensor:
        """
        Encode audio to VQ codes.

        Args:
            audio: Audio input - can be:
                - str: Path to audio file
                - np.ndarray: Audio waveform array
                - torch.Tensor: Audio waveform tensor
            sr: Sample rate (required if audio is array/tensor)
            loudness_normalize: Whether to normalize loudness
            loudness_threshold: Target loudness in dB

        Returns:
            torch.Tensor: VQ codes with shape [batch, num_codebooks, time]
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
            if audio.ndim > 1:
                audio = audio.squeeze()

        with torch.no_grad():
            vq_code = self.tokenizer.encode(
                audio,
                sr=sr,
                loudness_normalize=loudness_normalize,
                loudness_threshold=loudness_threshold,
            )

        return vq_code

    def decode(self, vq_code: torch.Tensor) -> np.ndarray:
        """
        Decode VQ codes back to audio.

        Args:
            vq_code: VQ codes with shape [num_codebooks, time] or [batch, num_codebooks, time]

        Returns:
            np.ndarray: Reconstructed audio waveform [batch, channels, samples]
        """
        # Add batch dimension if needed
        if vq_code.ndim == 2:
            vq_code = vq_code.unsqueeze(0)  # [num_codebooks, time] -> [1, num_codebooks, time]

        with torch.no_grad():
            audio = self.tokenizer.decode(vq_code)

        return audio

    def save_audio(self, audio: np.ndarray, path: str, sample_rate: Optional[int] = None):
        """
        Save audio to file.

        Args:
            audio: Audio array to save [batch, channels, samples] or [channels, samples] or [samples]
            path: Output file path (e.g., 'output.wav')
            sample_rate: Audio sampling rate (uses tokenizer's rate if not provided)
        """
        if sample_rate is None:
            sample_rate = self.sampling_rate

        # Handle different audio shapes
        if audio.ndim == 3:
            # [batch, channels, samples] - take first batch
            audio = audio[0]

        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio.float()

        # Ensure audio is 2D [channels, samples]
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        torchaudio.save(path, audio_tensor, sample_rate)

    @property
    def sampling_rate(self) -> int:
        """Get the audio sampling rate."""
        return self.tokenizer.sampling_rate

    @property
    def num_codebooks(self) -> int:
        """Get the number of codebooks."""
        return self.tokenizer.num_codebooks

    @property
    def frame_rate(self) -> int:
        """Get the frame rate (tokens per second)."""
        return self.tokenizer.tps
