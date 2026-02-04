# Fiji Hindi Live Speech Translator

Real-time bidirectional speech translation between English and Fiji Hindi.

Completes the translation arc: Project 37 → 38 → 43 → 47

## Architecture
```
English Speech → [Whisper API] → [Custom Transformer] → [TTS] → Fiji Hindi Speech
```

## Features
- Real-time speech recognition (Whisper)
- Custom Transformer translation (73% accuracy, 2,399 pairs)
- Natural speech synthesis (gTTS)
- Runs locally on M4 Pro

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

## Tech Stack
- Whisper API (speech recognition)
- Custom Transformer from Project 43 (translation)
- gTTS (text-to-speech)
- TensorFlow 2.x

## Performance
- Translation: 73% accuracy
- Latency: <2s end-to-end
- Training data: 2,399 Fiji Hindi pairs

## Author
**Rxkz** - [GitHub](https://github.com/rxkz)

Part of [DL Practice Week](https://github.com/Rxkz/Deep-Learning-Practice-Projects)