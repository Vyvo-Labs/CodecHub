<div align="center">
<h2>
    CodecHub: A Unified Library for Codec Models
</h2>
<div>
    <div align="center">
    <img width="400" alt="CodecHub Logo" src="assets/logo.png" style="max-width: 100%; height: auto;">
</div>
</div>
<div>
    <a href="https://github.com/Vyvo-Labs/VyvoTTS" target="_blank">
        <img src="https://img.shields.io/github/stars/Vyvo-Labs/CodecHub?style=for-the-badge&color=FF6B6B&labelColor=2D3748" alt="GitHub stars">
    </a>
    <a href="https://github.com/Vyvo-Labs/CodecHub/blob/main/LICENSE" target="_blank">
        <img src="https://img.shields.io/badge/License-MIT-4ECDC4?style=for-the-badge&labelColor=2D3748" alt="MIT License">
    </a>
    <a href="https://python.org" target="_blank">
        <img src="https://img.shields.io/badge/Python-3.8+-45B7D1?style=for-the-badge&logo=python&logoColor=white&labelColor=2D3748" alt="Python 3.8+">
    </a>
    <a href="https://huggingface.co/spaces/Vyvo/CodecHub" target="_blank">
        <img src="https://img.shields.io/badge/🤗_Hugging_Face-Spaces-FFD93D?style=for-the-badge&labelColor=2D3748" alt="HuggingFace Spaces">
    </a>
</div>
</div

## Install

```bash
uv venv --python 3.10
uv pip install -e .
```

## Usage

### WavTokenizer Codec

```python
from codecplus import load_codec
from codecplus.utils import load_audio, save_audio

# Load audio
audio, sr = load_audio('input.wav')

# Download and load WavTokenizer from HuggingFace
tokenizer = load_codec(model_name='wav_tokenizer', hf_id='Vyvo-Research/WavTokenizer-large-speech-320-v2')

# Encode and decode
features, discrete_codes = tokenizer.encode(audio)
output = tokenizer.decode(features)

# Save output
save_audio(output, 'output.wav', sr)
```

### DAC Codec

```python
from codecplus import load_codec
from codecplus.utils import load_audio, save_audio

# Load audio and resample to 24kHz to match DAC model
audio, sr = load_audio('input.wav', sample_rate=24000)

# Download and load DAC from HuggingFace using Transformers
# Available models: 'descript/dac_16khz', 'descript/dac_24khz', 'descript/dac_44khz'
dac = load_codec(model_name='dac', hf_id='descript/dac_24khz')

# Encode audio to get quantized representation and discrete codes
quantized_representation, audio_codes = dac.encode(audio, sample_rate=sr)

# Decode from quantized representation
output = dac.decode(quantized_representation=quantized_representation)

# Save output
save_audio(output, 'output_dac.wav', sr)
```

### Mimi Codec

```python
from codecplus import load_codec
from codecplus.utils import load_audio, save_audio

# Load audio and resample to 24kHz to match Mimi model
audio, sr = load_audio('input.wav', sample_rate=24000)

# Download and load Mimi from HuggingFace
mimi = load_codec(model_name='mimi', hf_id='kyutai/mimi')

# Encode audio to get discrete codes (both semantic and acoustic)
audio_codes, semantic_codes = mimi.encode(audio, sample_rate=sr)

# Decode from audio codes
output = mimi.decode(audio_codes=audio_codes)

# Handle output shape (B, C, T)
if output.dim() == 3:
    output_audio = output[0, 0, :]
else:
    output_audio = output.squeeze()

# Save output
save_audio(output_audio, 'output_mimi.wav', sr)

# Extract semantic tokens only (first quantizer)
semantic_tokens = mimi.get_semantic_tokens(audio, sample_rate=sr)
print(f"Semantic tokens shape: {semantic_tokens.shape}")
```

### SNAC Codec

```python
from codecplus import load_codec
from codecplus.utils import load_audio, save_audio

# Load audio and resample to 32kHz to match SNAC model
audio, sr = load_audio('input.wav', sample_rate=32000)

# Download and load SNAC from HuggingFace
# Available models: 'hubertsiuzdak/snac_24khz', 'hubertsiuzdak/snac_32khz', 'hubertsiuzdak/snac_44khz'
snac = load_codec(model_name='snac', hf_id='hubertsiuzdak/snac_32khz')

# Method 1: Encode and decode separately
codes = snac.encode(audio)
output = snac.decode(codes)

# Method 2: Forward pass (encode + decode in one step)
audio_hat, codes = snac.forward(audio)

# Save output (SNAC outputs shape (B, 1, T))
if output.dim() == 3:
    output_audio = output[0, 0, :]
else:
    output_audio = output.squeeze()

save_audio(output_audio, 'output_snac.wav', sr)

# Analyze multi-scale codes
print(f"Number of RVQ levels: {len(codes)}")
for i, code in enumerate(codes):
    print(f"Level {i}: {code.shape}")
```


## 🙏 Acknowledgements

We would like to thank the following projects and teams that made this work possible:

- [WavTokenizer](https://github.com/jishengpeng/WavTokenizer)
- [DAC](https://github.com/descriptinc/descript-audio-codec)
- [Mimi](https://huggingface.co/kyutai/mimi) - Neural audio codec by Kyutai
- [SNAC](https://github.com/hubertsiuzdak/snac) - Multi-Scale Neural Audio Codec

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
