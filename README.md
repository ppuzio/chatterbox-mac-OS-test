# Voice Cloning with Chatterbox TTS

A voice cloning system using [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) - Resemble AI's state-of-the-art open-source text-to-speech model. This project is optimized for Mac (Apple Silicon) and handles long text processing with minimal concatenation artifacts.

## Features

- **High-quality voice cloning** using reference audio samples
- **Long text processing** that bypasses the 1000-token limit
- **Multiple processing strategies** for optimal audio quality
- **Mac Apple Silicon optimized** with automatic compatibility patches
- **Expressive speech control** with emotion and intensity settings
- **Batch processing** capabilities
- **Minimal concatenation artifacts** through intelligent chunking

## 🔧 Installation

### Prerequisites

- Mac with Apple Silicon (M1/M2/M4) recommended
- Python 3.9 or higher
- At least 8GB RAM (16GB+ recommended for faster processing)

### Install Dependencies

```bash
pip install chatterbox-tts
```

This will automatically install all required dependencies including:

- torch and torchaudio (with Mac compatibility)
- transformers
- diffusers
- librosa
- numpy
- and other ML libraries

## 📁 Project Structure

```
your-repo/
├── voice_samples/
│   ├── sample_1.wav          # Your reference voice (required)
│   ├── sample_2.wav          # Additional samples (optional)
│   └── text.py               # Your texts to convert
├── process_optimal_chunks.py  # Main processing script (recommended)
├── process_smooth_audio.py   # Alternative with crossfading
├── voice_cloner.py           # Interactive voice cloner
└── README.md
```

## 🎤 Voice Samples Setup

### Reference Audio Requirements

Place your reference voice samples in the `voice_samples/` directory:

- **Primary sample**: `sample_1.wav` (required)
- **Additional samples**: `sample_2.wav`, `sample_3.wav`, etc. (optional)

**Audio specifications:**

- Format: WAV (recommended) or MP3
- Length: 10-30 seconds optimal
- Quality: Clear speech, minimal background noise
- Content: Natural conversational speech works best

### Text Input Format

Create `voice_samples/text.py` with your texts:

```python
TEXT = """
Your first text goes here. This can be multiple sentences and paragraphs.
The system will process this into natural-sounding speech using your voice.
"""

TEXT2 = """
Your second text here. You can have multiple text variables.
Each will be processed separately.
"""

TEXT3 = """
Third text example. Keep each text focused on one topic or section.
"""
```

## 🚀 Usage

### Quick Start

1. **Add your voice sample** to `voice_samples/sample_1.wav`
2. **Add your texts** to `voice_samples/text.py`
3. **Run the optimal processor** (recommended):

```bash
python process_optimal_chunks.py
```

This will:

- Automatically find the optimal chunk size for your system
- Process all texts in `text.py`
- Output high-quality audio files to `optimal_audio_output/`

### Alternative Processing Methods

#### Interactive Voice Cloner

```bash
python voice_cloner.py --presets -r voice_samples/sample_1.wav
```

#### Single Text Processing

```bash
python voice_cloner.py "Your text here" -r voice_samples/sample_1.wav -o output.wav
```

#### Batch Processing

Create a text file with one sentence per line, then:

```bash
python voice_cloner.py --batch sentences.txt -r voice_samples/sample_1.wav
```

### Advanced Options

#### Emotion Control

```bash
python voice_cloner.py "Excited text!" -r voice_samples/sample_1.wav -e 0.8 -c 0.2
```

- `-e` / `--exaggeration`: 0.0-1.0 (higher = more expressive)
- `-c` / `--cfg-weight`: 0.0-1.0 (lower = better pacing for expression)
- `-t` / `--temperature`: 0.0-1.0 (higher = more variation)

## 📊 Processing Strategies

### 1. Optimal Chunks (Recommended)

**Script**: `process_optimal_chunks.py`
**Output**: `optimal_audio_output/`

- Finds maximum safe text length per generation
- Minimizes concatenation points
- Best quality with minimal artifacts
- 2-4 audio segments per text typically

### 2. Smooth Audio Processing

**Script**: `process_smooth_audio.py`
**Output**: `smooth_audio_output/`

- Uses crossfading between segments
- More processing overhead
- May introduce some artifacts
- Good for very long texts

### 3. Basic Long Audio

**Script**: `process_long_texts.py`
**Output**: `long_audio_output/`

- Simple concatenation with gaps
- Fast processing
- Most obvious seams
- Good for testing

## 🍎 Mac Compatibility Notes

This project includes automatic Mac compatibility patches for:

- **CUDA device mapping issues** - automatically maps to CPU
- **MPS acceleration** - uses Apple Silicon GPU when available
- **Memory optimization** - configured for Mac RAM management
- **PyTorch compatibility** - handles Mac-specific serialization

No manual configuration needed - the scripts automatically detect and adapt to your Mac environment.

## 📈 Performance Tips

### For Faster Processing

- Use shorter texts (under 1000 characters per TEXT variable)
- Close other intensive applications
- Use the optimal chunks method
- Process one text at a time for very long content

### For Better Quality

- Use high-quality reference audio (clear, no background noise)
- Keep reference samples 10-30 seconds long
- Use the expressive settings you prefer: `exaggeration=0.7, cfg_weight=0.3`
- Let longer texts process completely (can take 10-30 minutes)

## 🎵 Output Files

Generated audio files will be saved to:

- `optimal_audio_output/text_1_optimal.wav`
- `optimal_audio_output/text_2_optimal.wav`
- `optimal_audio_output/text_3_optimal.wav`

**Audio specifications:**

- Format: WAV
- Sample rate: 24kHz
- Quality: High-fidelity voice cloning
- Watermarked: Includes Resemble AI's Perth watermark

## 🐛 Troubleshooting

### Common Issues

**"CUDA device error"**

- Already handled by compatibility patches
- Script will automatically use CPU

**"Model loading failed"**

- Check internet connection (downloads ~2GB on first run)
- Ensure sufficient disk space
- Try restarting the script

**"Reference audio not found"**

- Verify `voice_samples/sample_1.wav` exists
- Check file permissions
- Ensure audio format is supported (WAV/MP3)

**"Processing stuck"**

- Very long texts can take 30+ minutes
- Check process status: `ps aux | grep python`
- Monitor output directories for progress

### Performance Issues

- Close other applications to free RAM
- Use shorter text chunks
- Process texts individually
- Restart Python session between long runs

## 🔍 Advanced Usage

### Custom Settings

Edit the settings in processing scripts:

```python
settings = {
    "exaggeration": 0.7,    # 0.0-1.0, higher = more expressive
    "cfg_weight": 0.3,      # 0.0-1.0, lower = better expression pacing
    "temperature": 0.8      # 0.0-1.0, higher = more variation
}
```

### Multiple Voice Samples

To use different reference voices:

```bash
python voice_cloner.py "Your text" -r voice_samples/sample_2.wav
```

### Preset Emotional Styles

Generate samples with different emotional presets:

```bash
python voice_cloner.py --presets -r voice_samples/sample_1.wav
```

## 📚 Credits

- [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) by Resemble AI
- Built with PyTorch, Transformers, and other open-source libraries
- Optimized for Mac Apple Silicon development environment

## 📄 License

This project uses Chatterbox TTS under the MIT License. This implementation is also released under the MIT License. Generated audio includes watermarking as per Resemble AI's responsible AI practices.
