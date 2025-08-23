# WavTokenizer

WavTokenizer is a neural audio codec that converts audio waveforms into discrete tokens and reconstructs them back to audio. It uses an encoder-decoder architecture with vector quantization for efficient audio compression and tokenization.

## Overview

WavTokenizer provides a tokenization approach to audio processing, enabling discrete representation of continuous audio signals. This is particularly useful for applications requiring symbolic audio representations or efficient storage and transmission.

## Architecture

The WavTokenizer consists of several key components:

- **Encoder**: Converts raw audio into latent representations
- **Vector Quantization**: Discretizes continuous latent vectors into tokens
- **Decoder**: Reconstructs audio from quantized tokens
- **Discriminators**: Ensure high-quality audio reconstruction during training

### Key Components

- `encoder/`: Contains the encoding modules
  - `model.py`: Main encoder implementation
  - `modules/`: Neural network building blocks (conv, LSTM, transformer, etc.)
  - `quantization/`: Vector quantization implementations
- `decoder/`: Contains the decoding modules
  - `models.py`: Decoder architecture
  - `discriminators.py`: Adversarial discriminators for training

## Usage

### Basic Example

```python
from codecplus import load_codec
from codecplus.utils import load_audio, save_audio

# Load audio
audio, sr = load_audio('input.wav')

# Initialize WavTokenizer
wav_tokenizer = load_codec('wav_tokenizer', sample_rate=16000)

# Encode audio to tokens
tokens = wav_tokenizer.encode(audio)
print(f"Token shape: {tokens.shape}")

# Decode tokens back to audio
reconstructed_audio = wav_tokenizer.decode(tokens)

# Save reconstructed audio
save_audio(reconstructed_audio, 'output.wav', sr)
```

### Using the Pipeline

```python
from codecplus.pipelines.wav_tokenizer import WavTokenizerPipeline

# Initialize pipeline with pretrained model
pipeline = WavTokenizerPipeline(
    model_path="path/to/model.pth",
    config_path="path/to/config.yaml",
    device="cuda"
)

# Process audio
tokens = pipeline.encode(audio)
reconstructed = pipeline.decode(tokens)
```

## Configuration

WavTokenizer supports various configuration options:

- `sample_rate`: Target sample rate for audio processing
- `frame_size`: Frame size for tokenization
- `codebook_size`: Size of the vector quantization codebook
- `embedding_dim`: Dimension of token embeddings

### Available Configurations

The library includes several pre-configured models in the `configs/` directory:

- `wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml`
- `wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml`
- `wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml`

## Features

- **Efficient Compression**: Reduces audio data size while maintaining quality
- **Discrete Tokens**: Enables symbolic processing of audio
- **Flexible Architecture**: Supports different model sizes and configurations
- **Pre-trained Models**: Ready-to-use models for common audio tasks
- **GPU Acceleration**: CUDA support for fast processing

## Applications

- **Audio Compression**: Efficient storage and transmission
- **Speech Processing**: Tokenization for language models
- **Music Generation**: Discrete audio representations for generative models
- **Audio Analysis**: Feature extraction and representation learning

## Technical Details

### Encoder Architecture
- Convolutional layers for feature extraction
- LSTM/Transformer modules for temporal modeling
- Vector quantization for discretization

### Decoder Architecture  
- Upsampling layers for reconstruction
- Adversarial training with discriminators
- Multi-scale spectral loss for high-quality output

### Training Features
- Multi-scale discriminators
- Spectral losses (STFT, mel-spectrogram)
- Feature matching losses
- Perceptual losses

## Performance

WavTokenizer achieves:
- High-quality audio reconstruction
- Efficient compression ratios
- Fast inference on modern hardware
- Support for various audio formats and sample rates

## Limitations

- Requires GPU for optimal performance
- Model size increases with quality requirements
- May introduce artifacts in very noisy audio
- Training requires large datasets for best results

## References

Based on research in neural audio codecs and vector quantization techniques. The implementation includes optimizations for practical deployment and usage.