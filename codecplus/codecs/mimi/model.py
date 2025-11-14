import torch


class Mimi:
    """Mimi neural audio codec implementation using Transformers library"""

    def __init__(self):
        self.model = None
        self.feature_extractor = None

    @classmethod
    def from_pretrained(cls, repo_id: str) -> "Mimi":
        """
        Load a pretrained Mimi model from HuggingFace Hub using Transformers.

        Args:
            repo_id: HuggingFace repository ID
                    Available models:
                    - 'kyutai/mimi' (24kHz, streaming capable)

        Returns:
            Mimi: The loaded model instance

        Examples:
            >>> mimi = Mimi.from_pretrained('kyutai/mimi')
        """
        try:
            from transformers import MimiModel, AutoFeatureExtractor
        except ImportError:
            raise ImportError(
                "The 'transformers' package is required to use Mimi models. "
                "Install it with: pip install transformers"
            )

        # Load model and feature extractor from HuggingFace
        model = MimiModel.from_pretrained(repo_id)
        feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)

        # Create instance
        instance = cls()
        instance.model = model
        instance.feature_extractor = feature_extractor
        instance.model.eval()

        return instance

    def encode(self, audio, sample_rate=None, num_quantizers=None):
        """
        Encode audio to discrete codes.

        Args:
            audio: Audio tensor of shape (T,) or (B, T)
            sample_rate: Sample rate of the audio. If None, uses feature_extractor's default.
            num_quantizers: Number of quantizers (codebooks) to use. By default, all quantizers are used.
                          Mimi has 32 quantizers total, with the first quantizer being semantic.

        Returns:
            Tuple of (audio_codes, semantic_codes)
            - audio_codes: Discrete codebook indices of shape (batch_size, num_quantizers, codes_length)
            - semantic_codes: First quantizer codes (semantic tokens) of shape (batch_size, 1, codes_length)
        """
        if self.model is None or self.feature_extractor is None:
            raise RuntimeError(
                "Model not loaded. Use Mimi.from_pretrained() to load a model."
            )

        # Convert to numpy if tensor
        if torch.is_tensor(audio):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio

        # Use feature extractor to prepare inputs
        if sample_rate is None:
            sample_rate = self.feature_extractor.sampling_rate

        inputs = self.feature_extractor(
            raw_audio=audio_np,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )

        # Move to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Encode
        with torch.no_grad():
            encoder_outputs = self.model.encode(
                inputs["input_values"],
                padding_mask=inputs.get("padding_mask"),
                num_quantizers=num_quantizers
            )

        audio_codes = encoder_outputs.audio_codes

        # Extract semantic codes (first quantizer)
        semantic_codes = audio_codes[:, :1, :]

        return audio_codes, semantic_codes

    def decode(self, audio_codes, padding_mask=None):
        """
        Decode from audio codes to audio.

        Args:
            audio_codes: Discrete codebook indices of shape (batch_size, num_quantizers, codes_length)
            padding_mask: Optional padding mask

        Returns:
            Audio tensor of shape (B, C, T) where C is the number of audio channels
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Use Mimi.from_pretrained() to load a model."
            )

        # Decode
        with torch.no_grad():
            decoder_outputs = self.model.decode(
                audio_codes=audio_codes,
                padding_mask=padding_mask
            )

        audio = decoder_outputs.audio_values

        return audio

    def forward(self, audio, sample_rate=None, num_quantizers=None):
        """
        Full encode-decode pass (reconstruction).

        Args:
            audio: Audio tensor of shape (T,) or (B, T)
            sample_rate: Sample rate of the audio. If None, uses feature_extractor's default.
            num_quantizers: Number of quantizers to use. By default, all quantizers are used.

        Returns:
            Reconstructed audio tensor of shape (B, C, T)
        """
        if self.model is None or self.feature_extractor is None:
            raise RuntimeError(
                "Model not loaded. Use Mimi.from_pretrained() to load a model."
            )

        # Convert to numpy if tensor
        if torch.is_tensor(audio):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio

        # Use feature extractor to prepare inputs
        if sample_rate is None:
            sample_rate = self.feature_extractor.sampling_rate

        inputs = self.feature_extractor(
            raw_audio=audio_np,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )

        # Move to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                inputs["input_values"],
                padding_mask=inputs.get("padding_mask"),
                num_quantizers=num_quantizers
            )

        audio = outputs.audio_values

        return audio

    def get_semantic_tokens(self, audio, sample_rate=None):
        """
        Extract semantic tokens from audio (first quantizer only).

        Args:
            audio: Audio tensor of shape (T,) or (B, T)
            sample_rate: Sample rate of the audio. If None, uses feature_extractor's default.

        Returns:
            Semantic tokens of shape (batch_size, 1, codes_length)
        """
        # Encode with only the first quantizer
        audio_codes, semantic_codes = self.encode(audio, sample_rate, num_quantizers=1)
        return semantic_codes

    @property
    def sampling_rate(self):
        """Get the model's sampling rate"""
        if self.feature_extractor is None:
            raise RuntimeError("Feature extractor not loaded")
        return self.feature_extractor.sampling_rate

    @property
    def frame_rate(self):
        """Get the model's frame rate (12 Hz for Mimi)"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model.config.frame_rate
