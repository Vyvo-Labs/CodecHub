def load_audio(file_path, sample_rate=None):
    """Load audio file

    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate (None for original)

    Returns:
        tuple: (audio_tensor, sample_rate)
    """
    import torch
    import soundfile as sf

    # Load audio file using soundfile
    audio, orig_sr = sf.read(file_path, dtype='float32')

    # Convert to torch tensor
    audio = torch.from_numpy(audio)

    # Ensure audio is 2D (C, T) format
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # Add channel dimension
    else:
        # If stereo or multi-channel, transpose to (C, T)
        audio = audio.transpose(0, 1)

    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Resample if target sample rate is specified
    if sample_rate is not None and orig_sr != sample_rate:
        import torchaudio
        resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
        audio = resampler(audio)
        orig_sr = sample_rate

    # Remove channel dimension (B, T) format
    audio = audio.squeeze(0)

    return audio, orig_sr


def save_audio(audio, file_path, sample_rate):
    """Save audio to file

    Args:
        audio: Audio tensor (1D, 2D, or 3D)
        file_path: Output file path
        sample_rate: Sample rate
    """
    import torch
    import soundfile as sf

    # Ensure audio is on CPU and convert to numpy
    if torch.is_tensor(audio):
        if audio.device.type != 'cpu':
            audio = audio.cpu()
        audio = audio.numpy()

    # Handle different dimensions
    if audio.ndim == 3:
        # If (B, C, T), take first batch and transpose to (T, C)
        audio = audio[0].T
    elif audio.ndim == 2:
        # If (C, T), transpose to (T, C)
        audio = audio.T
    elif audio.ndim == 1:
        # If (T,), keep as is
        pass
    else:
        raise ValueError(f"Unexpected audio dimension: {audio.ndim}")

    # Ensure mono audio is (T,) not (T, 1)
    if audio.ndim == 2 and audio.shape[1] == 1:
        audio = audio.squeeze(1)

    # Save audio file using soundfile
    sf.write(file_path, audio, sample_rate)


def resample_audio(audio, orig_sr, target_sr):
    """Resample audio to target sample rate

    Args:
        audio: Audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        array: Resampled audio
    """
    # Implementation would go here
    return audio
