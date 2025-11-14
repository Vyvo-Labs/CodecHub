import torch


class DAC:
    """Descript Audio Codec implementation using Transformers library"""

    def __init__(self):
        self.model = None
        self.processor = None

    @classmethod
    def from_pretrained(cls, repo_id: str) -> "DAC":
        """
        Load a pretrained DAC model from HuggingFace Hub using Transformers.

        Args:
            repo_id: HuggingFace repository ID
                    Available models:
                    - 'descript/dac_16khz' (16kHz)
                    - 'descript/dac_24khz' (24kHz)
                    - 'descript/dac_44khz' (44.1kHz)

        Returns:
            DAC: The loaded model instance

        Examples:
            >>> dac = DAC.from_pretrained('descript/dac_16khz')
            >>> dac = DAC.from_pretrained('descript/dac_44khz')
        """
        try:
            from transformers import DacModel, AutoProcessor
        except ImportError:
            raise ImportError(
                "The 'transformers' package is required to use DAC models. "
                "Install it with: pip install transformers"
            )

        # Load model and processor from HuggingFace
        model = DacModel.from_pretrained(repo_id)
        processor = AutoProcessor.from_pretrained(repo_id)

        # Create instance
        instance = cls()
        instance.model = model
        instance.processor = processor
        instance.model.eval()

        return instance

    def encode(self, audio, sample_rate=None):
        """
        Encode audio to latent representation and discrete codes.

        Args:
            audio: Audio tensor of shape (T,) or (B, T)
            sample_rate: Sample rate of the audio. If None, uses processor's default.

        Returns:
            Tuple of (quantized_representation, audio_codes)
            - quantized_representation: Continuous latent representation
            - audio_codes: Discrete codebook indices
        """
        if self.model is None or self.processor is None:
            raise RuntimeError(
                "Model not loaded. Use DAC.from_pretrained() to load a model."
            )

        # Convert to numpy if tensor
        if torch.is_tensor(audio):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio

        # Use processor to prepare inputs
        if sample_rate is None:
            sample_rate = self.processor.sampling_rate

        inputs = self.processor(
            raw_audio=audio_np,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )

        # Move to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Encode
        with torch.no_grad():
            encoder_outputs = self.model.encode(inputs["input_values"])

        return encoder_outputs.quantized_representation, encoder_outputs.audio_codes

    def decode(self, quantized_representation=None, audio_codes=None):
        """
        Decode from quantized representation or audio codes to audio.

        Args:
            quantized_representation: Continuous latent representation (optional)
            audio_codes: Discrete codebook indices (optional)
                        If provided, this will be used instead of quantized_representation

        Returns:
            Audio tensor of shape (B, T)
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Use DAC.from_pretrained() to load a model."
            )

        # Decode
        with torch.no_grad():
            decoder_outputs = self.model.decode(
                quantized_representation=quantized_representation,
                audio_codes=audio_codes
            )

        audio = decoder_outputs.audio_values

        # Return shape (B, T) by removing channel dimension if present
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        return audio

    def forward(self, audio, sample_rate=None):
        """
        Full encode-decode pass (reconstruction).

        Args:
            audio: Audio tensor of shape (T,) or (B, T)
            sample_rate: Sample rate of the audio. If None, uses processor's default.

        Returns:
            Reconstructed audio tensor
        """
        if self.model is None or self.processor is None:
            raise RuntimeError(
                "Model not loaded. Use DAC.from_pretrained() to load a model."
            )

        # Convert to numpy if tensor
        if torch.is_tensor(audio):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio

        # Use processor to prepare inputs
        if sample_rate is None:
            sample_rate = self.processor.sampling_rate

        inputs = self.processor(
            raw_audio=audio_np,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )

        # Move to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(inputs["input_values"])

        audio = outputs.audio_values

        # Return shape (B, T) by removing channel dimension if present
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        return audio

    def _extract_features(self, audio):
        # Legacy method - kept for compatibility
        return audio

    def _generate_audio(self, latents):
        # Legacy method - kept for compatibility
        return latents
