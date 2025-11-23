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

All codecs support loading from HuggingFace with `hf_id` parameter:

```python
from codecplus import load_codec
from codecplus.utils import load_audio, save_audio

# Load audio
audio, sr = load_audio('input.wav')
```

### WavTokenizer

```python
# Load from HuggingFace
tokenizer = load_codec('wav_tokenizer', hf_id='Vyvo-Research/WavTokenizer-large-speech-320-v2')

# Encode and decode
features, discrete_codes = tokenizer.encode(audio)
output = tokenizer.decode(features)
save_audio(output, 'output.wav', sr)
```

### LongCat Audio Codec

```python
# Load from HuggingFace (auto-downloads and caches)
longcat = load_codec('longcat', hf_id='Vyvo-Research/LongCat-Audio-Codec', variant='24k_4codebooks') # `16k_4codebooks`, `24k_2codebooks`, `24k_4codebooks`, `24k_4codebooks_aug_sft`
# Encode to semantic and acoustic tokens
semantic_codes, acoustic_codes = longcat.encode(audio, sr)

# Decode back to audio
output = longcat.decode(semantic_codes, acoustic_codes)
save_audio(output, 'output.wav', longcat.sample_rate)
```

### DAC Codec

```python
# Load DAC (local only for now)
dac = load_codec('dac', sample_rate=44100)
latents = dac.encode(audio)
output = dac.decode(latents)
```

### SNAC (Multi-Scale Neural Audio Codec)

```python
# Load from HuggingFace
snac = load_codec('snac', hf_id='hubertsiuzdak/snac_32khz')

# Encode to multi-scale discrete codes
codes = snac.encode(audio)

# Decode back to audio
output = snac.decode(codes)
save_audio(output, 'output.wav', snac.sampling_rate)
```

### Mimi

```python
# Load from HuggingFace
mimi = load_codec('mimi', hf_id='kyutai/mimi')

# Encode to audio codes and semantic codes
audio_codes, semantic_codes = mimi.encode(audio, sample_rate=sr)

# Decode back to audio
output = mimi.decode(audio_codes)
save_audio(output, 'output.wav', mimi.sampling_rate)

# Get only semantic tokens
semantic_tokens = mimi.get_semantic_tokens(audio, sample_rate=sr)
```

### XCodec2

```python
# Load from HuggingFace
xcodec2 = load_codec('xcodec2', hf_id='NandemoGHS/Anime-XCodec2-44.1kHz-v2')

# Encode to VQ codes (requires 16kHz audio input)
vq_code = xcodec2.encode(audio)

# Decode back to audio
output = xcodec2.decode(vq_code)
save_audio(output, 'output.wav', 44100)  # Output is 44.1kHz for this model
```

### Higgs Audio

```python
# Load from HuggingFace
higgs = load_codec('higgs_audio', hf_id='bosonai/higgs-audio-v2-tokenizer')

# Encode audio to VQ codes
vq_codes = higgs.encode(audio, sr=sr)

# Decode VQ codes back to audio
reconstructed = higgs.decode(vq_codes)

# Save the reconstructed audio
higgs.save_audio(reconstructed, 'output.wav')
```

### TaDiCodec

```python
# Load from HuggingFace
tadicodec = load_codec('tadicodec', hf_id='amphion/TaDiCodec')

# Simple reconstruction (audio input -> audio output)
reconstructed = tadicodec.forward(
    audio='input.wav',
    text='The spoken text content'  # Optional but improves quality
)

# With speaker prompt for voice cloning
reconstructed = tadicodec.forward(
    audio='target.wav',
    text='Hello world',
    prompt_audio='speaker_reference.wav',
    prompt_text='The reference text'
)

# Encode and decode separately
codes = tadicodec.encode(audio='input.wav', text='The text')
audio = tadicodec.decode(codes, text='The text')

# Save output
tadicodec.save_audio(reconstructed, 'output.wav')

# Model properties
print(f"Sampling rate: {tadicodec.sampling_rate} Hz")  # 24000 Hz
print(f"Frame rate: {tadicodec.frame_rate} Hz")       # 6.25 Hz
print(f"Bitrate: {tadicodec.bitrate} kbps")           # 0.0875 kbps
```

## 🙏 Acknowledgements

We would like to thank the following projects and teams that made this work possible:

- [WavTokenizer](https://github.com/jishengpeng/WavTokenizer)
- [DAC](https://github.com/descriptinc/descript-audio-codec)
- [LongCat Audio Codec](https://github.com/meituan-longcat/LongCat-Audio-Codec)
- [SNAC](https://github.com/hubertsiuzdak/snac)
- [Mimi](https://github.com/kyutai-labs/mimi)
- [XCodec2](https://github.com/HKUSTAudio/xcodec2)
- [Higgs Audio](https://github.com/bosonai/higgs-audio)
- [TaDiCodec](https://github.com/AmphionTeam/Diffusion-Speech-Tokenizer)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
