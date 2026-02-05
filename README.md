## Fiji Hindi Speech Translator

Sequential speech translation between English and Fiji Hindi using a custom-trained Transformer model.

**Part of Deep Learning Practice Week - Completes the translation arc: Project 37 → 38 → 43 → 47**

---

## Project Overview

A speech translation system that enables conversation between English and Fiji Hindi speakers. The system combines state-of-the-art speech recognition (Whisper) with a custom-trained Transformer model for translation, and text-to-speech synthesis for natural audio output.

This project demonstrates the strategic use of pre-trained models (Whisper, Edge-TTS) alongside custom-built models (Transformer), showcasing when to build vs. when to leverage existing solutions—a critical skill for AI engineers.

---
## Note on Translation Accuracy:
The translator achieves 73% exact match accuracy due to training on a limited dataset of 2,399 Fiji Hindi pairs. While single words translate with 90%+ accuracy, complex sentences face generalization challenges. Data augmentation was employed to expand from the initial 1,000 pairs to 2,399, but further improvement to 85%+ accuracy would require scaling to 10,000+ training examples. This limitation demonstrates the fundamental relationship between data quantity and model performance—the Transformer architecture is sound, but constrained by available training data for this low-resource language.

---

##  Architecture
```
English Speech Input
       ↓
[Whisper API] ——————→ English Text
       ↓
[Custom Transformer] → Fiji Hindi Text
       ↓
[Edge-TTS] ——————————→ Fiji Hindi Speech
       ↓
Audio Output
```

**Key Components:**
- **Speech-to-Text:** OpenAI Whisper (base model)
- **Translation:** Custom Transformer trained from scratch (Project 43, 73% accuracy)
- **Text-to-Speech:** Microsoft Edge-TTS (Hindi voice with better quality than gTTS)

---

## Related Projects

This project completes a translation learning arc across the Deep Learning Practice Week:

- **Project 37:** Basic text translation using RNN sequence-to-sequence
- **Project 38:** Improved translation with attention mechanism  
- **Project 43:** Transformer-based translation (English ↔ Fiji Hindi, 73% accuracy, 2,399 pairs)
- **Project 47:** Speech translation system (THIS PROJECT)

Full project series: [Deep Learning Practice Projects](https://github.com/Rxkz/Deep-Learning-Practice-Projects)

---

##  Features

- Sequential speech translation (speak English → hear Fiji Hindi)
- Custom Transformer model (73% accuracy on 2,399 training pairs)
- User-controlled translation loop with command interface
- Configurable recording duration (3/5/10 seconds)
- Runs locally on M4 Pro with Metal GPU acceleration
- High-quality speech synthesis using Edge-TTS
- Interactive mode with comprehensive error handling

---

## Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Speech Recognition | OpenAI Whisper (base) | English speech → text |
| Translation | Custom Transformer (4 layers, 8 heads, 256d) | English text → Fiji Hindi text |
| Text-to-Speech | Microsoft Edge-TTS | Fiji Hindi text → speech |
| Audio Playback | Pygame | Cross-platform audio playback |
| ML Framework | TensorFlow 2.15 | Model training & inference |
| GPU Acceleration | Metal (Apple Silicon) | Fast local inference |
| Audio I/O | sounddevice, soundfile | Recording & playback |

---

## Model Performance

| Metric | Value |
|--------|-------|
| Translation Accuracy | 73% (exact match) |
| Partial Match | 80%+ |
| Training Data | 2,399 Fiji Hindi pairs |
| Model Parameters | ~8-10M |
| Inference Time | <200ms (translation only) |
| End-to-End Latency | ~2-3s (speech→speech) |

**Performance Notes:**
- Single words: 90%+ accuracy 
- Short phrases (2-4 words): 80%+ accuracy 
- Full sentences (5+ words): 73% accuracy (limited by training data)

---

##  Installation

### Prerequisites
- macOS with Apple Silicon (M1/M2/M3/M4) or compatible system
- Python 3.9+
- 8GB+ RAM (24GB recommended)
- Microphone and speakers

### Setup
```bash
# Clone repository
git clone https://github.com/Rxkz/Fiji-Hindi-Speech-Translator.Ai.git
cd Fiji-Hindi-Speech-Translator.Ai

# Install dependencies
pip3 install -r requirements.txt

# Verify installation (optional)
python3 check_installation.py
```

### Dependencies
```txt
# Core ML/DL
tensorflow>=2.15.0
tensorflow-metal>=1.1.0

# Speech Recognition
openai-whisper
soundfile
librosa

# Text-to-Speech (switched from gTTS to Edge-TTS)
edge-tts>=6.1.0
pygame>=2.5.0

# Audio I/O
sounddevice

# Utils
numpy
pandas
tqdm
```

---

##  Usage

### Quick Start
```bash
python3 main.py
```

### Mode Selection

When you run the application, you'll be prompted to select a mode:

**1. Interactive Mode (Recommended)**
- User-controlled translation with command interface
- Commands: 'go' (5s), 'long' (10s), 'quit'
- Best for demos and testing

**2. Single Test Mode**
- One translation cycle then exit
- Quick functionality test

### Example Session
```
======================================================================
PROJECT 47: SPEECH TRANSLATION SYSTEM
English → Fiji Hindi Translation
======================================================================

[System initializes Whisper, Transformer, Edge-TTS...]

System ready for translation

SELECT MODE
======================================================================
1. Interactive Mode (translate with command control)
2. Single Test (one translation then exit)
======================================================================

Select mode (1/2): 1

======================================================================
INTERACTIVE MODE
======================================================================
Commands:
  'go' or 'g'     - Start translation (5 seconds)
  'long' or 'l'   - Long recording (10 seconds)
  'quit' or 'q'   - Exit
======================================================================

Command: go

Translating for 5 seconds...
Recording English speech...
[Speak: "Hello, how are you?"]

English text: Hello, how are you?
Translating to Fiji Hindi...
Fiji Hindi text: namaste, tum kaisan ho?
Speaking Fiji Hindi...
[Plays audio: "namaste, tum kaisan ho?"]

Command: _
```

---

## 📁 Project Structure
```
Fiji-Hindi-Speech-Translator/
├── data/
│   ├── project_43_fiji_hindi_transformer.weights.h5  # Trained model
│   ├── fiji_hindi_data.txt                           # Training data (2,399 pairs)
│   └── test_audio/                                   # Test audio files
├── models/                                            # (Reserved for future)
├── results/                                           # Output files
├── transformer.py                                     # Transformer architecture (from Project 43)
├── translator.py                                      # Translation wrapper
├── speech_to_text.py                                  # Whisper integration
├── text_to_speech.py                                  # Edge-TTS implementation
├── main.py                                            # Main application
├── requirements.txt                                   # Dependencies
├── README.md                                          # This file
└── .gitignore                                         # Git ignore rules
```

---

## Issues Encountered & Solutions

### 1. **Data Limitation - Translation Quality**

**Issue:** Initial training with limited data (2,399 pairs) resulted in 73% accuracy. Complex sentences struggled with novel compositions.

**Analysis:** The model learned individual words well (90%+ accuracy) but lacked enough sentence patterns for full generalization. Example: "give me water" translated as "hamke paani pasand hai" (I like water) instead of "hamke paani deo" (give me water).

**Solution:** 
- Accepted 73% as sufficient for demo/proof-of-concept
- Documented that scaling to 10K+ pairs would achieve 85%+ accuracy
- Focused on completing the end-to-end pipeline
- Plan to retrain on expanded dataset in future (upload to Colab for 200+ epochs)

**Learning:** Data quantity is the limiting factor, not architecture. The Transformer design is sound.

---

### 2. **TTS Audio Corruption**

**Issue:** Audio playback was corrupted/broken during continuous translation loop. Audio files were being deleted while still playing.

**Root Cause:** 
```python
# Original problematic code:
audio_file = self.speak(text, play=True)
time.sleep(0.5)  # Too short
os.remove(audio_file)  # Deleted while playing
```

**Solution:** Changed cleanup strategy to delete PREVIOUS file instead of current file:
```python
class TextToSpeech:
    def __init__(self):
        self.last_audio_file = None  # Track previous file
    
    def speak_and_cleanup(self, text):
        # Clean up PREVIOUS file (safe - already finished playing)
        if self.last_audio_file and os.path.exists(self.last_audio_file):
            os.remove(self.last_audio_file)
        
        # Play NEW file
        audio_file = self.speak(text)
        self.last_audio_file = audio_file  # Store for next cleanup
```

**Result:** Audio playback now clear and uninterrupted.

---

### 3. **First Letter Cutoff in TTS**

**Issue:** First letter of Fiji Hindi phrases was cut off during playback. Example: "kaise ho" sounded like "aise ho".

**Root Cause:** TTS engines sometimes trim leading silence too aggressively.

**Solution:** Added padding to text before synthesis:
```python
padded_text = f", ... {text} "  # Ellipsis creates leading buffer
tts = gTTS(text=padded_text, lang=self.lang)
```

**Result:** Full phrases now play correctly with all letters intact.

---

### 4. **gTTS vs Edge-TTS**

**Initial Choice:** Started with gTTS (Google Text-to-Speech)

-  Free
-  Limited voice quality
-  Robotic pronunciation
-  No voice customization

**Final Choice:** Switched to Edge-TTS (Microsoft Edge Text-to-Speech)
- Higher quality, more natural voices
- Better Hindi pronunciation
- Free (uses Microsoft Edge TTS Locally Ran)
- Multiple voice options
- Offline capability (after initial model download)

**Implementation Change:**
- Kept original gTTS code in git history
- Migrated to edge-tts for production
- Added pygame for reliable audio playback

---

### 5. **Whisper Model Size Selection**

**Testing:** Tried both 'base' and 'medium' Whisper models

**Comparison:**
- **Base Model:** Fast inference (~1-2s), good accuracy for clear speech
- **Medium Model:** Slower inference (~3-4s), slightly better accuracy

**Decision:** Settled on 'base' model
- Faster response time for better user experience
- Accuracy difference minimal for clear recordings
- M4 Pro handles 'base' model instantly

**Tradeoff:** Sacrificed 2-3% accuracy for 2x faster response time.

---

### 6. **Continuous Loop vs User-Controlled**

**Initial Design:** Auto-loop that continuously recorded and translated (Ctrl+C to exit)

**Problems:**
- Audio corruption when files deleted too quickly
- No pause for user to process translation
- Difficult to exit cleanly

**Final Design:** Command-based interactive mode
```
Commands:
  'go' - Translate (5 seconds)
  'long' - Translate (10 seconds)
  'quit' - Exit cleanly
```

**Benefits:**
- User controls pacing
- Clean exit without Ctrl+C
- Better demo/presentation format

---

##Testing Individual Components

### Test Translator Only
```bash
python3 translator.py
```

### Test Speech Recognition Only
```bash
python3 speech_to_text.py
```

### Test Text-to-Speech Only
```bash
python3 text_to_speech.py
```

---

##Use Cases

1. **Language Learning:** Practice Fiji Hindi pronunciation and comprehension
2. **Cultural Preservation:** Enable communication in Fiji Hindi for diaspora communities
3. **Travel Aid:** Translation assistance for travelers in Fiji
4. **Accessibility:** Voice-based translation for literacy barriers
5. **Research:** Demonstrate low-resource language translation capabilities

---

##Known Limitations

- **Translation Quality:** 73% accuracy due to limited training data (2,399 pairs)
- **Vocabulary:** Best performance on common words and short phrases seen in training
- **Novel Sentences:** May struggle with sentence structures not in training data
- **Background Noise:** Recognition accuracy affected by noisy environments
- **Hindi Voice Approximation:** No native Fiji Hindi TTS available (using Hindi as proxy)
- **Sequential Processing:** Not real-time streaming (requires recording → process → play cycle)



---

## Learning Outcomes

This project demonstrates:

**Technical Skills:**
-  Integration of multiple AI systems (speech, NLP, synthesis)
-  Strategic use of pre-trained vs. custom models
-  Pipeline design with error handling and state management
-  Edge deployment on Apple Silicon (Metal GPU acceleration)
-  Audio processing and real-time system constraints

**Engineering Practices:**
-  Problem identification and debugging (audio corruption, TTS cutoff)
-  Performance optimization (model size selection, cleanup strategy)
-  User experience design (command interface vs auto-loop)
-  Tradeoff analysis (accuracy vs latency, data vs architecture)

**AI/ML Concepts:**
-  Understanding data-model performance relationships
-  Low-resource language challenges
- Transfer learning and model integration
- Evaluation metrics for translation quality

---

##  Author

**Rxkz (Foxsin)**
- GitHub: [@Rxkz](https://github.com/Rxkz)
- AI/ML Engineering Student @ Yoobee Colleges, Auckland, NZ


**Portfolio:** This project is part of a 50-project Deep Learning practice series demonstrating hands-on implementation of modern AI techniques.

---

##  Acknowledgments

- OpenAI for Whisper speech recognition model
- Microsoft for Edge-TTS text-to-speech service
- Stanford University & Coursera for Deep Learning Specialization
- Fiji Hindi language community for cultural context
- Main project series: [Deep Learning Practice Projects](https://github.com/Rxkz/Deep-Learning-Practice-Projects)

---

## License

MIT License - See LICENSE file for details

---

##  Links

- **Main Portfolio:** [Deep Learning Practice Projects](https://github.com/Rxkz/Deep-Learning-Practice-Projects)
- **Project 43 (Transformer Training):** [Project 43 Documentation]
- **LinkedIn:** [Your LinkedIn Profile]

---

**Built  for  AI learning journey**

*Project 47 of 50 - Deep Learning Practice Week*  
*Demonstrating end-to-end speech translation with custom Transformers*