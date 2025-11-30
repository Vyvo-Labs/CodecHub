import os
import sys
import json
import librosa
from huggingface_hub import snapshot_download

import torch
import torch.nn as nn
from typing import Optional
import safetensors
from transformers import AutoTokenizer
from utils.util import load_config

from models.tts.tadicodec.inference_tadicodec import TaDiCodecPipline
from models.tts.llm_tts.mgm import MGMT2S

from models.tts.llm_tts.chat_template import gen_chat_prompt_for_tts


class MGMInferencePipeline(nn.Module):
    """
    MGM TTS inference pipeline that integrates TaDiCodec and MGM models
    Uses diffusion-based generation with mask-guided modeling
    """

    def __init__(
        self,
        tadicodec_path: str,
        mgm_path: str,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.mgm_path = mgm_path

        # Load TaDiCodec pipeline
        self.tadicodec = TaDiCodecPipline.from_pretrained(
            ckpt_dir=tadicodec_path, device=device
        )

        # Load tokenizer directly from pretrained
        self.tokenizer = AutoTokenizer.from_pretrained(
            mgm_path,
            trust_remote_code=True,
        )

        config_path = os.path.join(mgm_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        self.cfg = load_config(config_path)

        # Extract MGM config from the loaded config
        mgm_config = self.cfg.model.mgmt2s
        if not mgm_config:
            raise ValueError("MGM config not found in config.json")

        # Load MGM model with config - using the same pattern as llm_infer_eval.py
        self.mgm = MGMT2S(
            hidden_size=mgm_config.hidden_size,
            num_layers=mgm_config.num_layers,
            num_heads=mgm_config.num_heads,
            cfg_scale=mgm_config.cfg_scale,
            cond_codebook_size=mgm_config.cond_codebook_size,
            cond_dim=mgm_config.cond_dim,
            phone_vocab_size=mgm_config.phone_vocab_size,
        )

        # Load model weights
        model_path = os.path.join(mgm_path, "model.safetensors")

        if os.path.exists(model_path):
            safetensors.torch.load_model(self.mgm, model_path, strict=True)
        else:
            # Try loading from the directory directly
            safetensors.torch.load_model(self.mgm, mgm_path, strict=True)

        self.mgm.to(device)
        self.mgm.eval()

    def tensor_to_audio_string(self, tensor):
        """Convert tensor to audio string format"""
        if isinstance(tensor, list) and isinstance(tensor[0], list):
            values = tensor[0]
        else:
            values = tensor[0].tolist() if hasattr(tensor, "tolist") else tensor[0]

        result = "<|start_of_audio|>"
        for value in values:
            result += f"<|audio_{value}|>"
        return result

    def extract_audio_ids(self, text):
        """Extract audio IDs from string containing audio tokens"""
        import re

        pattern = r"<\|audio_(\d+)\|>"
        audio_ids = re.findall(pattern, text)
        return [int(id) for id in audio_ids]

    @classmethod
    def from_pretrained(
        cls,
        tadicodec_path: str = "./ckpt/TaDiCodec",
        mgm_path: str = "./ckpt/TaDiCodec-TTS-MGM",
        device: Optional[torch.device] = None,
        auto_download: bool = True,
    ):
        """
        Create pipeline from pretrained models

        Args:
            tadicodec_path: Path to TaDiCodec model or Hugging Face model ID
            mgm_path: Path to MGM model directory or Hugging Face model ID
            device: Device to run on
            auto_download: Whether to automatically download models from Hugging Face if not found locally

        Returns:
            MGMInferencePipeline instance
        """
        resolved_device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Handle TaDiCodec path
        resolved_tadicodec_path = cls._resolve_model_path(
            tadicodec_path, auto_download=auto_download, model_type="tadicodec"
        )

        # Handle MGM path
        resolved_mgm_path = cls._resolve_model_path(
            mgm_path, auto_download=auto_download, model_type="mgm"
        )

        return cls(
            tadicodec_path=resolved_tadicodec_path,
            mgm_path=resolved_mgm_path,
            device=resolved_device,
        )

    @staticmethod
    def _resolve_model_path(
        model_path: str, auto_download: bool = True, model_type: str = "mgm"
    ) -> str:
        """
        Resolve model path, downloading from Hugging Face if necessary

        Args:
            model_path: Local path or Hugging Face model ID
            auto_download: Whether to auto-download from HF
            model_type: Type of model ("mgm" or "tadicodec")

        Returns:
            Resolved local path
        """
        # If it's already a local path and exists, return as is
        if os.path.exists(model_path):
            return model_path

        # If it looks like a Hugging Face model ID (contains '/')
        if "/" in model_path and auto_download:
            print(f"Downloading {model_type} model from Hugging Face: {model_path}")
            try:
                # Download to cache directory
                cache_dir = os.path.join(
                    os.path.expanduser("~"), ".cache", "huggingface", "hub"
                )
                downloaded_path = snapshot_download(
                    repo_id=model_path,
                    cache_dir=cache_dir,
                    local_dir_use_symlinks=False,
                )
                print(
                    f"Successfully downloaded {model_type} model to: {downloaded_path}"
                )
                return downloaded_path
            except Exception as e:
                print(f"Failed to download {model_type} model from Hugging Face: {e}")
                raise ValueError(
                    f"Could not download {model_type} model from {model_path}"
                )

        # If it's a local path that doesn't exist
        if not os.path.exists(model_path):
            if auto_download:
                raise ValueError(
                    f"Model path does not exist: {model_path}. Set auto_download=True to download from Hugging Face."
                )
            else:
                raise FileNotFoundError(f"Model path does not exist: {model_path}")

        return model_path

    @torch.no_grad()
    def __call__(
        self,
        text: str,
        prompt_text: Optional[str] = None,
        prompt_speech_path: Optional[str] = None,
        n_timesteps_mgm: int = 25,
        n_timesteps: int = 25,
        target_len: Optional[int] = None,
        return_code: bool = False,
    ):
        """
        Perform MGM TTS inference

        Args:
            text: Target text to synthesize
            prompt_text: Prompt text for conditioning
            prompt_speech_path: Path to prompt audio file
            n_timesteps_mgm: Number of diffusion timesteps for MGM
            n_timesteps: Number of diffusion timesteps
            target_len: Target length for audio generation
            return_code: Whether to return audio codes instead of audio

        Returns:
            Generated audio array or audio codes
        """
        # Get prompt audio codes
        if prompt_speech_path:
            prompt_speech_code = self.tadicodec(
                speech_path=prompt_speech_path, return_code=True, text=""
            )
        else:
            raise ValueError("prompt_speech_path is required")

        # Convert prompt codes to tensor
        prompt_codes = torch.tensor(prompt_speech_code).to(self.device)
        prompt_len = prompt_codes.shape[1]

        # Tokenize text for phone conditioning
        input_text = gen_chat_prompt_for_tts(
            prompt_text + " " + text,
            "phi-3" if "phi" in self.cfg.preprocess.tokenizer_path else "qwen2",
        )

        ##### debug #####
        print("input_text: ", input_text)
        ##### debug #####

        text_token_ids = self.tokenizer.encode(input_text)
        text_token_ids = torch.tensor(text_token_ids).unsqueeze(0).to(self.device)

        # Estimate target length based on text length
        frame_rate = getattr(self.cfg.preprocess, "frame_rate", 6.25)

        if target_len is None:
            # If no target_len, estimate based on prompt speech length and text ratio
            prompt_text_len = len(prompt_text.encode("utf-8"))
            target_text_len = len(text.encode("utf-8"))
            prompt_speech_len = librosa.get_duration(filename=prompt_speech_path)
            target_speech_len = prompt_speech_len * target_text_len / prompt_text_len
            target_len = int(target_speech_len * frame_rate)
        else:
            # If target_len is provided, use it directly
            target_len = int(target_len * frame_rate)

        ##### debug #####
        print(f"Prompt length: {prompt_len}, Target length: {target_len}")
        print(f"Text: {text}")
        print(f"Prompt text: {prompt_text}")
        ##### debug #####

        # Generate audio codes using MGM reverse diffusion
        generated_codes = self.mgm.reverse_diffusion(
            prompt=prompt_codes,
            target_len=target_len,
            phone_id=text_token_ids,
            n_timesteps=n_timesteps_mgm,
            cfg=1.5,
            rescale_cfg=0.75,
        )

        print(f"Generated codes shape: {generated_codes.shape}")

        combine_codes = torch.cat([prompt_codes, generated_codes], dim=1)

        if return_code:
            return combine_codes

        # Decode audio using TaDiCodec
        prompt_mel = self.tadicodec.extract_mel_feature(prompt_speech_path)

        text_token_ids = self.tadicodec.tokenize_text(text, prompt_text)
        rec_mel = self.tadicodec.decode(
            indices=combine_codes,
            text_token_ids=text_token_ids,
            prompt_mel=prompt_mel,
            n_timesteps=n_timesteps,
        )

        rec_audio = (
            self.tadicodec.vocoder_model(rec_mel.transpose(1, 2))
            .detach()
            .cpu()
            .numpy()[0][0]
        )

        return rec_audio


# Usage example
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create pipeline
    pipeline = MGMInferencePipeline.from_pretrained(
        tadicodec_path="./ckpt/TaDiCodec",
        mgm_path="./ckpt/TaDiCodec-TTS-MGM",
        device=device,
    )

    # Inference on single sample
    audio = pipeline(
        text="但是 to those who 知道 her well, it was a 标志 of her unwavering 决心 and spirit.",
        prompt_text="In short, we embarked on a mission to make America great again, for all Americans.",
        prompt_speech_path="./use_examples/test_audio/trump_0.wav",
    )

    # Save audio
    import soundfile as sf

    sf.write("./use_examples/test_audio/mgm_tts_output.wav", audio, 24000)
