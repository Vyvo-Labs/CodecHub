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

```python
from codecplus import load_codec
from codecplus.utils import load_audio, save_audio

# Load audio
audio, sr = load_audio('input.wav')

# WavTokenizer
tokenizer = load_codec('wav_tokenizer')
tokens = tokenizer.encode(audio)
output = tokenizer.decode(tokens)

# DAC
dac = load_codec('dac')
latents = dac.encode(audio)
output = dac.decode(latents)

# Save output
save_audio(output, 'output.wav', sr)
```


## 🙏 Acknowledgements

We would like to thank the following projects and teams that made this work possible:

- [WavTokenizer](https://github.com/jishengpeng/WavTokenizer)
- [DAC](https://github.com/descriptinc/descript-audio-codec)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
