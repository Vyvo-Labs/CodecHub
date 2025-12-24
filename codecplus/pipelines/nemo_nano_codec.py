import torch
import librosa
import soundfile as sf
from pathlib import Path
from nemo.collections.tts.models import AudioCodecModel


class NeMoAudioProcessor:
    def __init__(self, model_name='nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps', device='cuda'):
        """
        Initialize the NeMoAudioProcessor

        Args:
            model_name: HuggingFace model name or local path
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.sample_rate = None

    def load_model(self):
        """Load the NeMo Audio Codec model"""
        self.model = AudioCodecModel.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.sample_rate = self.model.sample_rate

    def generate(self, input_audio_path, output_audio_path):
        """
        Generate audio reconstruction

        Args:
            input_audio_path: Path to input audio file
            output_audio_path: Path to save output audio file

        Returns:
            reconstructed_audio: Generated audio tensor
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please call load_model() first.")

        input_path = Path(input_audio_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input audio file not found: {input_audio_path}")

        # Load audio
        audio, _ = librosa.load(str(input_path), sr=self.sample_rate)

        # Prepare tensors
        audio_tensor = torch.from_numpy(audio).unsqueeze(dim=0).to(self.device)
        audio_len = torch.tensor([audio_tensor[0].shape[0]]).to(self.device)

        with torch.no_grad():
            # Encode audio to discrete tokens
            encoded_tokens, encoded_len = self.model.encode(
                audio=audio_tensor,
                audio_len=audio_len
            )

            # Decode tokens back to audio
            reconstructed_audio, _ = self.model.decode(
                tokens=encoded_tokens,
                tokens_len=encoded_len
            )

        # Save reconstructed audio
        output_path = Path(output_audio_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_audio = reconstructed_audio.cpu().detach().numpy().squeeze()
        sf.write(str(output_path), output_audio, self.sample_rate)

        return reconstructed_audio
