import torch
import numpy as np


class XCodec2:
    """XCodec2 neural audio codec implementation

    XCodec2 is a high-quality neural audio codec designed for speech compression.
    It operates at 16kHz sampling rate and provides efficient encoding/decoding.
    """

    def __init__(self):
        self.model = None
        self._device = "cpu"

    @classmethod
    def from_pretrained(cls, repo_id: str, device: str = None) -> "XCodec2":
        """
        Load a pretrained XCodec2 model from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repository ID
                    Available models:
                    - 'NandemoGHS/Anime-XCodec2-44.1kHz-v2' (44.1kHz, anime speech)
                    - 'HKUSTAudio/xcodec2' (16kHz, speech)
            device: Device to load the model on ('cpu', 'cuda', etc.)
                   If None, will use 'cuda' if available, else 'cpu'

        Returns:
            XCodec2: The loaded model instance

        Examples:
            >>> xcodec = XCodec2.from_pretrained('NandemoGHS/Anime-XCodec2-44.1kHz-v2')
            >>> xcodec = XCodec2.from_pretrained('HKUSTAudio/xcodec2', device='cpu')
        """
        try:
            from .modeling_xcodec2 import XCodec2Model
            from .configuration_bigcodec import BigCodecConfig
            from huggingface_hub import hf_hub_download
            from safetensors import safe_open
        except ImportError as e:
            raise ImportError(
                f"Missing required dependencies: {e}\n"
                "Install with: pip install transformers huggingface_hub safetensors"
            )

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Download and load model from HuggingFace
        ckpt_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        ckpt = {}
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                # Handle beta -> bias naming conversion
                ckpt[k.replace(".beta", ".bias")] = f.get_tensor(k)

        # Load config and model
        codec_config = BigCodecConfig.from_pretrained(repo_id)
        model = XCodec2Model.from_pretrained(
            None, config=codec_config, state_dict=ckpt
        )
        model.eval()
        model = model.to(device)

        # Create instance
        instance = cls()
        instance.model = model
        instance._device = device

        return instance

    def encode(self, audio):
        """
        Encode audio to discrete VQ codes.

        Args:
            audio: Audio tensor of shape (T,) or (B, T)
                  Must be 16kHz audio
                  Note: Only supports single input at a time

        Returns:
            VQ codes tensor representing the compressed audio
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Use XCodec2.from_pretrained() to load a model."
            )

        # Convert audio to proper format
        audio = self._prepare_audio_input(audio)

        # Move to model device
        audio = audio.to(self._device)

        # Encode
        with torch.no_grad():
            codes = self.model.encode_code(input_waveform=audio)
            # Normalize codes output to tensor format
            vq_code = self._codes_to_tensor(codes)

        return vq_code

    def _codes_to_tensor(self, codes):
        """
        Normalize the output of xcodec2.encode_code to a tensor with shape (1, 1, N).
        Handles version differences where the return type/shape may vary.
        """
        if isinstance(codes, torch.Tensor):
            return codes.to(self._device)
        try:
            t = torch.as_tensor(codes[0][0], device=self._device)
            return t.unsqueeze(0).unsqueeze(0) if t.ndim == 1 else t
        except Exception:
            return torch.as_tensor(codes, device=self._device)

    def decode(self, vq_code):
        """
        Decode from VQ codes to audio.

        Args:
            vq_code: VQ codes from encode()

        Returns:
            Audio tensor of shape (B, 1, T)
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Use XCodec2.from_pretrained() to load a model."
            )

        # Decode
        with torch.no_grad():
            audio = self.model.decode_code(vq_code)

        return audio

    def forward(self, audio):
        """
        Full encode-decode pass (reconstruction).

        Args:
            audio: Audio tensor of shape (T,) or (B, T)
                  Must be 16kHz audio

        Returns:
            Tuple of (reconstructed_audio, vq_code)
            - reconstructed_audio: Audio tensor of shape (B, 1, T)
            - vq_code: VQ codes
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Use XCodec2.from_pretrained() to load a model."
            )

        # Encode
        vq_code = self.encode(audio)

        # Decode
        recon_audio = self.decode(vq_code)

        return recon_audio, vq_code

    def _prepare_audio_input(self, audio):
        """
        Convert audio to (B, T) format.

        Args:
            audio: Audio tensor of shape (T,) or (B, T)

        Returns:
            Audio tensor of shape (B, T)
        """
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio).float()

        # Handle different input shapes
        if audio.dim() == 1:
            # (T,) -> (1, T)
            audio = audio.unsqueeze(0)
        elif audio.dim() == 2:
            # Already (B, T)
            pass
        elif audio.dim() == 3:
            # (B, C, T) -> (B, T) - convert to mono if needed
            if audio.shape[1] == 1:
                audio = audio.squeeze(1)
            else:
                # Average channels to mono
                audio = audio.mean(dim=1)
        else:
            raise ValueError(
                f"Expected audio with 1, 2, or 3 dimensions, got {audio.dim()}"
            )

        return audio

    @property
    def device(self):
        """Get the device the model is on"""
        return self._device

    @property
    def sampling_rate(self):
        """Get the model's sampling rate (16kHz for XCodec2)"""
        return 16000

    def to(self, device):
        """Move model to specified device"""
        if self.model is not None:
            self.model = self.model.to(device)
            self._device = device
        return self
