"""
TaDiCodec wrapper for CodecPlus library.

This module provides a simple wrapper around TaDiCodecPipline to match
the CodecPlus API conventions.
"""

import torch
import soundfile as sf
import numpy as np
from typing import Optional, Union
from pathlib import Path

from codecplus.codecs.tadicodec.models.tts.tadicodec.inference_tadicodec import TaDiCodecPipline


class TaDiCodec:
    """
    TaDiCodec: Text-aware Diffusion Codec for high-quality audio reconstruction.

    TaDiCodec is a neural audio codec that uses text conditioning to improve
    reconstruction quality. It supports voice cloning with speaker prompts.

    Examples:
        # Load from HuggingFace
        tadicodec = TaDiCodec.from_pretrained('amphion/TaDiCodec')

        # Simple reconstruction
        output = tadicodec.forward(
            audio='input.wav',
            text='Hello world'
        )

        # With speaker prompt
        output = tadicodec.forward(
            audio='target.wav',
            text='Hello world',
            prompt_audio='speaker.wav',
            prompt_text='Reference text'
        )

        # Encode and decode separately
        codes = tadicodec.encode(audio='input.wav', text='text')
        audio = tadicodec.decode(codes, text='text')
    """

    def __init__(self, pipe: TaDiCodecPipline):
        """
        Initialize TaDiCodec wrapper.

        Args:
            pipe: TaDiCodecPipline instance
        """
        self.pipe = pipe
        self.device = pipe.device

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = 'amphion/TaDiCodec',
        device: Optional[torch.device] = None,
    ):
        """
        Load pretrained TaDiCodec from HuggingFace or local path.

        Args:
            model_path: HuggingFace model ID or local path
            device: Device to load model on (default: auto-detect CUDA)

        Returns:
            TaDiCodec instance
        """
        pipe = TaDiCodecPipline.from_pretrained(
            ckpt_dir=model_path,
            device=device
        )
        return cls(pipe)

    def forward(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        text: Optional[str] = None,
        prompt_audio: Optional[Union[str, np.ndarray, torch.Tensor]] = None,
        prompt_text: Optional[str] = None,
        n_timesteps: int = 32,
        cfg_scale: float = 2.0,
    ) -> np.ndarray:
        """
        Reconstruct audio with optional text and speaker conditioning.

        Args:
            audio: Input audio (file path, numpy array, or tensor)
            text: Text content of the audio (optional but improves quality)
            prompt_audio: Speaker reference audio for voice cloning
            prompt_text: Text content of the prompt audio
            n_timesteps: Number of diffusion steps (higher = better quality)
            cfg_scale: Classifier-free guidance scale

        Returns:
            Reconstructed audio as numpy array
        """
        # Convert audio to path if needed
        audio_path = self._prepare_audio(audio, 'target.wav')
        prompt_audio_path = self._prepare_audio(prompt_audio, 'prompt.wav') if prompt_audio is not None else None

        # Extract mel features
        target_mel = self.pipe.extract_mel_feature(audio_path)
        text_input_ids = self.pipe.tokenize_text(text, prompt_text)

        # Handle prompt
        if prompt_audio_path:
            prompt_mel = self.pipe.extract_mel_feature(prompt_audio_path)
            vq_emb, indices = self.pipe.encode(torch.cat([prompt_mel, target_mel], dim=1))
        else:
            vq_emb, indices = self.pipe.encode(target_mel)
            # Use first half or max 150 frames as prompt (whichever is smaller)
            prompt_len = min(150, target_mel.shape[1] // 2)
            prompt_mel = target_mel[:, :prompt_len]

        # Decode with prompt
        rec_mel = self.pipe.decode(
            vq_emb=vq_emb,
            text_token_ids=text_input_ids,
            prompt_mel=prompt_mel,
            n_timesteps=n_timesteps,
            cfg=cfg_scale,
            rescale_cfg=0.75,
        )

        # Concatenate prompt + reconstructed part to get full audio
        full_mel = torch.cat([prompt_mel, rec_mel], dim=1)

        # Vocoder
        rec_audio = self.pipe.vocoder_model(full_mel.transpose(1, 2)).detach().cpu().numpy()[0][0]
        return rec_audio

    def encode(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        text: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Encode audio to discrete codes.

        Args:
            audio: Input audio (file path, numpy array, or tensor)
            text: Text content of the audio (optional)

        Returns:
            Discrete VQ codes
        """
        audio_path = self._prepare_audio(audio, 'encode_input.wav')
        target_mel = self.pipe.extract_mel_feature(audio_path)
        vq_emb, indices = self.pipe.encode(target_mel)
        return indices

    def decode(
        self,
        codes: torch.Tensor,
        text: Optional[str] = None,
        n_timesteps: int = 32,
        cfg_scale: float = 2.0,
    ) -> np.ndarray:
        """
        Decode discrete codes to audio.

        Args:
            codes: Discrete VQ codes
            text: Text content for conditioning (optional)
            n_timesteps: Number of diffusion steps
            cfg_scale: Classifier-free guidance scale

        Returns:
            Reconstructed audio as numpy array
        """
        text_input_ids = self.pipe.tokenize_text(text, None)
        vq_emb = self.pipe.tadicodec.index2vq(codes.long())

        # Decode with empty prompt
        rec_mel = self.pipe.decode(
            vq_emb=vq_emb,
            text_token_ids=text_input_ids,
            prompt_mel=torch.zeros(1, 0, 128, device=self.device),
            n_timesteps=n_timesteps,
            cfg=cfg_scale,
            rescale_cfg=0.75,
        )

        # Vocoder
        rec_audio = self.pipe.vocoder_model(rec_mel.transpose(1, 2)).detach().cpu().numpy()[0][0]
        return rec_audio

    def save_audio(self, audio: np.ndarray, path: str, sr: Optional[int] = None):
        """
        Save audio to file.

        Args:
            audio: Audio numpy array
            path: Output file path
            sr: Sample rate (default: 24000 Hz)
        """
        sr = sr or self.sampling_rate
        sf.write(path, audio, sr)

    def _prepare_audio(self, audio: Union[str, np.ndarray, torch.Tensor], temp_name: str) -> str:
        """Convert audio input to file path."""
        if isinstance(audio, str):
            return audio
        elif isinstance(audio, (np.ndarray, torch.Tensor)):
            # Save to temporary file
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            sf.write(temp_name, audio, self.sampling_rate)
            return temp_name
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")

    @property
    def sampling_rate(self) -> int:
        """Audio sampling rate in Hz."""
        return 24000

    @property
    def frame_rate(self) -> float:
        """Frame rate in Hz."""
        # hop_size = 480, sr = 24000
        # frame_rate = sr / hop_size = 24000 / 480 = 50 Hz
        return 50.0

    @property
    def bitrate(self) -> float:
        """Bitrate in kbps."""
        # 14 codebooks * frame_rate / 1000
        # Assuming 14 VQ codes based on TaDiCodec architecture
        return (14 * self.frame_rate) / 1000  # 0.7 kbps
