"""
Unified model download utilities for all codecs
"""

from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download, snapshot_download


def get_cache_dir(codec_name: str) -> Path:
    """
    Get cache directory for a specific codec

    Args:
        codec_name: Name of the codec (e.g., 'longcat', 'wav_tokenizer', 'dac')

    Returns:
        Path to cache directory
    """
    cache_dir = Path.home() / ".cache" / "codecplus" / codec_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_from_hf(
    repo_id: str,
    filename: Optional[str] = None,
    codec_name: str = "model",
    cache_dir: Optional[Path] = None,
) -> str:
    """
    Download a model from HuggingFace Hub

    Args:
        repo_id: HuggingFace repository ID (e.g., 'username/repo-name')
        filename: Specific file to download (if None, downloads entire repo)
        codec_name: Name of codec for cache organization
        cache_dir: Custom cache directory (default: ~/.cache/codecplus/{codec_name})

    Returns:
        Path to downloaded file or directory

    Example:
        >>> # Download specific file
        >>> path = download_from_hf(
        ...     repo_id='meituan-longcat/LongCat-Audio-Codec',
        ...     filename='ckpts/LongCatAudioCodec_encoder.pt',
        ...     codec_name='longcat'
        ... )

        >>> # Download entire repository
        >>> path = download_from_hf(
        ...     repo_id='username/model-repo',
        ...     codec_name='mycodec'
        ... )
    """
    if cache_dir is None:
        cache_dir = get_cache_dir(codec_name)

    print(f"Downloading from HuggingFace: {repo_id}")

    try:
        if filename:
            # Download specific file
            print(f"  File: {filename}")
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                resume_download=True,
            )
            print(f"  ✓ Downloaded to: {file_path}")
            return file_path
        else:
            # Download entire repository
            print(f"  Downloading entire repository...")
            repo_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                resume_download=True,
            )
            print(f"  ✓ Downloaded to: {repo_path}")
            return repo_path

    except Exception as e:
        print(f"  ✗ Error downloading: {e}")
        raise


def clear_cache(codec_name: Optional[str] = None):
    """
    Clear cached models

    Args:
        codec_name: Specific codec to clear (if None, clears all)
    """
    import shutil

    if codec_name:
        cache_dir = get_cache_dir(codec_name)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"✓ Cleared cache for {codec_name}")
    else:
        cache_root = Path.home() / ".cache" / "codecplus"
        if cache_root.exists():
            shutil.rmtree(cache_root)
            print("✓ Cleared all codec caches")
