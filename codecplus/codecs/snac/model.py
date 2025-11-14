import torch


class SNAC:
    """Multi-Scale Neural Audio Codec (SNAC) implementation

    SNAC compresses audio into discrete codes at a low bitrate using
    multi-scale tokens with variable temporal resolutions.
    """

    def __init__(self):
        self.model = None
        self._device = "cpu"

    @classmethod
    def from_pretrained(cls, repo_id: str, device: str = None) -> "SNAC":
        """
        Load a pretrained SNAC model from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repository ID
                    Available models:
                    - 'hubertsiuzdak/snac_24khz' (24kHz, 0.98 kbps, Speech)
                    - 'hubertsiuzdak/snac_32khz' (32kHz, 1.9 kbps, Music/SFX)
                    - 'hubertsiuzdak/snac_44khz' (44kHz, 2.6 kbps, Music/SFX)
            device: Device to load the model on ('cpu', 'cuda', etc.)
                   If None, will use 'cuda' if available, else 'cpu'

        Returns:
            SNAC: The loaded model instance

        Examples:
            >>> snac = SNAC.from_pretrained('hubertsiuzdak/snac_32khz')
            >>> snac = SNAC.from_pretrained('hubertsiuzdak/snac_24khz', device='cpu')
        """
        try:
            import snac as snac_lib
        except ImportError:
            raise ImportError(
                "The 'snac' package is required to use SNAC models. "
                "Install it with: pip install snac"
            )

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model from HuggingFace
        model = snac_lib.SNAC.from_pretrained(repo_id).eval()
        model = model.to(device)

        # Create instance
        instance = cls()
        instance.model = model
        instance._device = device

        return instance

    def encode(self, audio):
        """
        Encode audio to multi-scale discrete codes.

        Args:
            audio: Audio tensor of shape (T,), (B, T), or (B, 1, T)
                  Will be automatically converted to (B, 1, T) format

        Returns:
            List of code tensors at different temporal resolutions
            Each element in the list represents tokens at a different scale:
            - codes[0]: Coarsest scale (lowest temporal resolution)
            - codes[-1]: Finest scale (highest temporal resolution)

            Example for snac_32khz: 4 RVQ levels with rates [10, 21, 42, 83] Hz
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Use SNAC.from_pretrained() to load a model."
            )

        # Convert audio to proper format (B, 1, T)
        audio = self._prepare_audio_input(audio)

        # Move to model device
        audio = audio.to(self._device)

        # Encode
        with torch.inference_mode():
            codes = self.model.encode(audio)

        return codes

    def decode(self, codes):
        """
        Decode from multi-scale codes to audio.

        Args:
            codes: List of code tensors from encode()
                  Each tensor represents codes at a different temporal resolution

        Returns:
            Audio tensor of shape (B, 1, T)
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Use SNAC.from_pretrained() to load a model."
            )

        # Decode
        with torch.inference_mode():
            audio = self.model.decode(codes)

        return audio

    def forward(self, audio):
        """
        Full encode-decode pass (reconstruction).

        Args:
            audio: Audio tensor of shape (T,), (B, T), or (B, 1, T)

        Returns:
            Tuple of (reconstructed_audio, codes)
            - reconstructed_audio: Audio tensor of shape (B, 1, T)
            - codes: List of code tensors at different temporal resolutions
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Use SNAC.from_pretrained() to load a model."
            )

        # Convert audio to proper format (B, 1, T)
        audio = self._prepare_audio_input(audio)

        # Move to model device
        audio = audio.to(self._device)

        # Forward pass
        with torch.inference_mode():
            audio_hat, codes = self.model(audio)

        return audio_hat, codes

    def _prepare_audio_input(self, audio):
        """
        Convert audio to (B, 1, T) format.

        Args:
            audio: Audio tensor of shape (T,), (B, T), or (B, 1, T)

        Returns:
            Audio tensor of shape (B, 1, T)
        """
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio).float()

        # Handle different input shapes
        if audio.dim() == 1:
            # (T,) -> (1, 1, T)
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            # (B, T) -> (B, 1, T)
            audio = audio.unsqueeze(1)
        elif audio.dim() == 3:
            # Already (B, C, T) or (B, 1, T)
            if audio.shape[1] != 1:
                # If stereo or multi-channel, convert to mono
                audio = audio.mean(dim=1, keepdim=True)
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
        """Get the model's sampling rate"""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Extract sampling rate from model repo_id or config
        # This is a heuristic based on the model name
        model_name = getattr(self.model, 'name_or_path', '')
        if '24khz' in model_name.lower():
            return 24000
        elif '32khz' in model_name.lower():
            return 32000
        elif '44khz' in model_name.lower():
            return 44100
        else:
            # Default fallback
            return 32000

    def to(self, device):
        """Move model to specified device"""
        if self.model is not None:
            self.model = self.model.to(device)
            self._device = device
        return self
