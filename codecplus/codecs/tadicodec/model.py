"""
TaDiCodec - Text-aware Diffusion Speech Tokenizer for CodecPlus.

This module provides a wrapper around TaDiCodec for audio compression
and reconstruction using text-aware diffusion-based speech coding.

Reference: https://arxiv.org/abs/2508.16790
"""

import os
import sys
import torch
import torchaudio
import numpy as np
from typing import Optional, Union
from pathlib import Path

# Add TaDiCodec directory to path
TADICODEC_ROOT = Path(__file__).parent.parent.parent.parent / "TaDiCodec"
if str(TADICODEC_ROOT) not in sys.path:
    sys.path.insert(0, str(TADICODEC_ROOT))


class TaDiCodec:
    """
    TaDiCodec wrapper for audio encoding and decoding.

    TaDiCodec is a text-aware diffusion-based speech tokenizer that achieves
    extremely low bitrate (0.0875 kbps at 6.25 Hz frame rate) while maintaining
    high speech quality using text guidance.

    Examples:
        # Load from HuggingFace
        codec = TaDiCodec.from_pretrained('amphion/TaDiCodec')

        # Encode and reconstruct audio with text guidance
        reconstructed = codec.forward(
            audio='input.wav',
            text='The spoken text content',
            prompt_audio='prompt.wav',
            prompt_text='The prompt text'
        )

        # Just encode to get codes
        codes = codec.encode(audio='input.wav', text='The spoken text')

        # Decode from codes
        audio = codec.decode(codes, text='The spoken text')
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize TaDiCodec.

        Args:
            model_path: Path or HuggingFace model ID for TaDiCodec
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.device = torch.device(device)

        # Import TaDiCodec pipeline
        try:
            from models.tts.tadicodec.inference_tadicodec import TaDiCodecPipline
            self._pipeline_class = TaDiCodecPipline
        except ImportError as e:
            raise ImportError(
                f"Failed to import TaDiCodec modules: {e}\n"
                f"Make sure TaDiCodec is available at: {TADICODEC_ROOT}"
            )

        # Load the pipeline
        self.pipeline = self._pipeline_class.from_pretrained(
            ckpt_dir=model_path,
            device=self.device,
            auto_download=True
        )
        self.pipeline.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "amphion/TaDiCodec",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Load a pretrained TaDiCodec model.

        Args:
            model_path: Path or HuggingFace model ID for TaDiCodec
                       Default: 'amphion/TaDiCodec'
                       Also available: 'amphion/TaDiCodec-old'
            device: Device to run the model on ('cuda' or 'cpu')

        Returns:
            TaDiCodec instance

        Examples:
            >>> codec = TaDiCodec.from_pretrained('amphion/TaDiCodec')
            >>> codec = TaDiCodec.from_pretrained('./local/path', device='cpu')
        """
        return cls(model_path=model_path, device=device)

    def encode(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        text: Optional[str] = None,
        prompt_audio: Optional[Union[str, np.ndarray, torch.Tensor]] = None,
        prompt_text: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Encode audio to VQ codes.

        Args:
            audio: Input audio - can be:
                - str: Path to audio file (24kHz)
                - np.ndarray: Audio waveform array
                - torch.Tensor: Audio waveform tensor
            text: Text transcription of the audio (optional but recommended)
            prompt_audio: Prompt audio for speaker conditioning (optional)
            prompt_text: Text transcription of prompt audio (optional)

        Returns:
            torch.Tensor: VQ codes representing the compressed audio
        """
        # Convert audio to path if needed
        audio_path = self._prepare_audio_path(audio, "target_audio.wav")
        prompt_audio_path = None
        if prompt_audio is not None:
            prompt_audio_path = self._prepare_audio_path(prompt_audio, "prompt_audio.wav")

        # Extract mel features
        target_mel = self.pipeline.extract_mel_feature(audio_path)

        if prompt_audio_path is not None:
            prompt_mel = self.pipeline.extract_mel_feature(prompt_audio_path)
            combined_mel = torch.cat([prompt_mel, target_mel], dim=1)
        else:
            combined_mel = target_mel

        # Encode to VQ codes
        with torch.no_grad():
            _, indices = self.pipeline.encode(combined_mel)

        return indices

    def decode(
        self,
        vq_code: torch.Tensor,
        text: Optional[str] = None,
        prompt_audio: Optional[Union[str, np.ndarray, torch.Tensor]] = None,
        prompt_text: Optional[str] = None,
        n_timesteps: int = 32,
        cfg_scale: float = 2.0,
    ) -> np.ndarray:
        """
        Decode VQ codes back to audio.

        Args:
            vq_code: VQ codes from encode()
            text: Text transcription to guide reconstruction (optional but improves quality)
            prompt_audio: Prompt audio for speaker conditioning (optional)
            prompt_text: Text transcription of prompt audio (optional)
            n_timesteps: Number of diffusion timesteps (default: 32)
            cfg_scale: Classifier-free guidance scale (default: 2.0)

        Returns:
            np.ndarray: Reconstructed audio waveform at 24kHz
        """
        # Prepare text input
        text_input_ids = None
        if text is not None:
            text_input_ids = self.pipeline.tokenize_text(text, prompt_text)

        # Prepare prompt mel if provided
        prompt_mel = None
        if prompt_audio is not None:
            prompt_audio_path = self._prepare_audio_path(prompt_audio, "prompt_audio.wav")
            prompt_mel = self.pipeline.extract_mel_feature(prompt_audio_path)

        # Decode using diffusion
        with torch.no_grad():
            rec_mel = self.pipeline.decode(
                indices=vq_code,
                text_token_ids=text_input_ids,
                prompt_mel=prompt_mel,
                n_timesteps=n_timesteps,
                cfg=cfg_scale,
                rescale_cfg=0.75,
            )

            # Vocoder to waveform
            rec_audio = (
                self.pipeline.vocoder_model(rec_mel.transpose(1, 2))
                .detach()
                .cpu()
                .numpy()[0][0]
            )

        return rec_audio

    def forward(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        text: str,
        prompt_audio: Optional[Union[str, np.ndarray, torch.Tensor]] = None,
        prompt_text: Optional[str] = None,
        n_timesteps: int = 32,
        cfg_scale: float = 2.0,
        use_self_prompt: bool = True,
    ) -> np.ndarray:
        """
        Full encode-decode pass (reconstruction).

        This is the main method for audio reconstruction with TaDiCodec.
        It performs encoding to VQ codes and decoding back to audio with
        text-aware diffusion.

        IMPORTANT:
        - TaDiCodec is a TEXT-AWARE codec and REQUIRES text transcription
        - It's designed for TTS/voice conversion with both prompt and target audio
        - For simple reconstruction, set use_self_prompt=True to use the same audio as both
        - Text transcription significantly improves reconstruction quality

        Args:
            audio: Input audio - can be:
                - str: Path to audio file (24kHz)
                - np.ndarray: Audio waveform array
                - torch.Tensor: Audio waveform tensor
            text: Text transcription of the audio (REQUIRED - model needs text guidance)
            prompt_audio: Prompt audio for speaker conditioning (optional, overrides use_self_prompt)
            prompt_text: Text transcription of prompt audio (optional)
            n_timesteps: Number of diffusion timesteps (default: 32, higher = better quality)
            cfg_scale: Classifier-free guidance scale (default: 2.0, higher = stronger text guidance)
            use_self_prompt: If True and no prompt_audio given, use input audio as prompt (default: True)

        Returns:
            np.ndarray: Reconstructed audio waveform at 24kHz

        Examples:
            >>> # Simple reconstruction (uses same audio as prompt)
            >>> audio = codec.forward(
            ...     audio='input.wav',
            ...     text='The spoken text content'  # Required!
            ... )

            >>> # Voice conversion: use different prompt
            >>> audio = codec.forward(
            ...     audio='target.wav',
            ...     text='Hello world',
            ...     prompt_audio='speaker_reference.wav',
            ...     prompt_text='The reference text',
            ...     use_self_prompt=False
            ... )
        """
        # Convert audio to path if needed
        audio_path = self._prepare_audio_path(audio, "target_audio.wav")
        prompt_audio_path = None

        # Handle prompt audio
        if prompt_audio is not None:
            prompt_audio_path = self._prepare_audio_path(prompt_audio, "prompt_audio.wav")
        elif use_self_prompt:
            # Use the same audio as both prompt and target for full reconstruction
            prompt_audio_path = audio_path

        # Use the pipeline's __call__ method
        # TaDiCodec is text-aware by design - it works better with text transcription
        # If no text provided, model will use audio-only mode (lower quality)
        actual_text = text
        actual_prompt_text = None

        if text is not None:
            # When text is provided, use it for both prompt and target if using self-prompt
            actual_prompt_text = prompt_text if prompt_audio is not None else text

        with torch.no_grad():
            rec_audio = self.pipeline(
                text=actual_text,
                speech_path=audio_path,
                prompt_text=actual_prompt_text,
                prompt_speech_path=prompt_audio_path,
                n_timesteps=n_timesteps,
                return_code=False,
                cfg_scale=cfg_scale,
            )

        return rec_audio

    def _prepare_audio_path(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        temp_name: str = "temp_audio.wav"
    ) -> str:
        """
        Convert audio to file path, saving to temp file if needed.

        Args:
            audio: Audio input
            temp_name: Temporary filename to use if saving is needed

        Returns:
            str: Path to audio file
        """
        if isinstance(audio, str):
            # Already a path
            return audio

        # Need to save to temp file
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        if isinstance(audio, np.ndarray):
            # Ensure correct shape
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]  # Add channel dimension
            elif audio.ndim > 2:
                audio = audio.squeeze()
                if audio.ndim == 1:
                    audio = audio[np.newaxis, :]

            # Create temp directory if needed
            temp_dir = Path("/tmp/tadicodec_temp")
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / temp_name

            # Save audio
            audio_tensor = torch.from_numpy(audio).float()
            torchaudio.save(str(temp_path), audio_tensor, 24000)

            return str(temp_path)

        raise ValueError(f"Unsupported audio type: {type(audio)}")

    def save_audio(
        self,
        audio: np.ndarray,
        path: str,
        sample_rate: Optional[int] = None
    ):
        """
        Save audio to file.

        Args:
            audio: Audio array to save
            path: Output file path (e.g., 'output.wav')
            sample_rate: Audio sampling rate (uses 24kHz if not provided)
        """
        if sample_rate is None:
            sample_rate = self.sampling_rate

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
        """Get the audio sampling rate (24kHz for TaDiCodec)."""
        return 24000

    @property
    def frame_rate(self) -> float:
        """Get the frame rate (tokens per second)."""
        return 6.25

    @property
    def bitrate(self) -> float:
        """Get the bitrate in kbps."""
        return 0.0875

    def to(self, device):
        """Move model to specified device."""
        self.device = torch.device(device)
        self.pipeline.tadicodec.to(self.device)
        self.pipeline.vocoder_model.to(self.device)
        self.pipeline.mel_model.to(self.device)
        return self
