#!/usr/bin/env python3
"""
Process long texts with smooth audio transitions - no obvious gaps
"""

import os
import sys
import torch
import torchaudio as ta
import numpy as np
from datetime import datetime

# Mac compatibility patch for Chatterbox TTS
original_torch_load = torch.load

def patched_torch_load(f, map_location=None, pickle_module=None, **kwargs):
    """Patched torch.load that defaults to CPU map_location on Mac"""
    if map_location is None:
        map_location = torch.device('cpu')
    return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)

torch.load = patched_torch_load

# Import after patching
from chatterbox.tts import ChatterboxTTS

# Import the texts
sys.path.append('voice_samples')
from text import TEXT, TEXT2, TEXT3

def setup_model():
    """Initialize the Chatterbox TTS model"""
    print("🔄 Loading Chatterbox TTS model...")
    try:
        model = ChatterboxTTS.from_pretrained(device="cpu")
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

def clean_text(text):
    """Clean and prepare text for TTS"""
    # Remove extra whitespace and newlines
    text = ' '.join(text.split())
    # Remove any problematic characters
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    return text.strip()

def detect_silence_threshold(audio, percentile=10):
    """Automatically detect silence threshold based on audio"""
    audio_abs = torch.abs(audio)
    threshold = torch.quantile(audio_abs, percentile / 100.0)
    return threshold.item()

def trim_silence(audio, threshold=None):
    """Trim silence from beginning and end of audio"""
    if audio.numel() == 0:
        return audio
    
    # Auto-detect threshold if not provided
    if threshold is None:
        threshold = detect_silence_threshold(audio, percentile=5)
    
    # Find non-silent regions
    audio_abs = torch.abs(audio.squeeze())
    non_silent = audio_abs > threshold
    
    if not torch.any(non_silent):
        # If everything is silent, return a very short piece
        return audio[:, :int(0.1 * 24000)]  # 0.1 seconds
    
    # Find first and last non-silent samples
    non_silent_indices = torch.where(non_silent)[0]
    start_idx = non_silent_indices[0].item()
    end_idx = non_silent_indices[-1].item()
    
    # Add small padding to avoid cutting off speech
    padding = int(0.05 * 24000)  # 50ms padding
    start_idx = max(0, start_idx - padding)
    end_idx = min(audio.shape[1], end_idx + padding)
    
    return audio[:, start_idx:end_idx]

def crossfade_audio(audio1, audio2, fade_length=0.1, sample_rate=24000):
    """Crossfade between two audio segments"""
    fade_samples = int(fade_length * sample_rate)
    
    if audio1.shape[1] < fade_samples or audio2.shape[1] < fade_samples:
        # If audio too short for crossfade, just concatenate with minimal gap
        tiny_gap = torch.zeros(1, int(0.05 * sample_rate))  # 50ms gap
        return torch.cat([audio1, tiny_gap, audio2], dim=1)
    
    # Extract fade regions
    audio1_fade_out = audio1[:, -fade_samples:]
    audio2_fade_in = audio2[:, :fade_samples]
    
    # Create fade curves
    fade_out_curve = torch.linspace(1, 0, fade_samples).unsqueeze(0)
    fade_in_curve = torch.linspace(0, 1, fade_samples).unsqueeze(0)
    
    # Apply fades
    audio1_faded = audio1_fade_out * fade_out_curve
    audio2_faded = audio2_fade_in * fade_in_curve
    
    # Mix the faded regions
    crossfaded_region = audio1_faded + audio2_faded
    
    # Combine: audio1 (without fade region) + crossfaded region + audio2 (without fade region)
    result = torch.cat([
        audio1[:, :-fade_samples],
        crossfaded_region,
        audio2[:, fade_samples:]
    ], dim=1)
    
    return result

def smart_text_split(text, max_chars=120):
    """Split text more intelligently to avoid awkward breaks"""
    # First split by sentences
    import re
    sentences = re.split(r'([.!?]+)', text)
    
    # Recombine sentences with their punctuation
    complete_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = sentences[i] + sentences[i + 1]
            complete_sentences.append(sentence.strip())
    
    # Further split long sentences at natural breakpoints
    chunks = []
    for sentence in complete_sentences:
        if len(sentence) <= max_chars:
            chunks.append(sentence)
        else:
            # Split at commas, semicolons, or conjunctions
            parts = re.split(r'(,\s+|;\s+|\s+and\s+|\s+but\s+|\s+or\s+|\s+so\s+|\s+because\s+)', sentence)
            
            current_chunk = ""
            for part in parts:
                if len(current_chunk + part) <= max_chars:
                    current_chunk += part
                else:
                    if current_chunk.strip():
                        # Ensure chunk ends with punctuation
                        if not current_chunk.strip().endswith(('.', '!', '?', ',', ';')):
                            current_chunk += ","
                        chunks.append(current_chunk.strip())
                    current_chunk = part
            
            if current_chunk.strip():
                # Ensure final chunk ends with punctuation
                if not current_chunk.strip().endswith(('.', '!', '?')):
                    current_chunk += "."
                chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if chunk.strip()]

def process_text_to_smooth_audio(model, text, output_file, reference_audio_path):
    """Process text with smooth audio transitions"""
    print(f"\n🎭 Processing smooth audio for: {output_file}")
    
    # Clean the text
    cleaned_text = clean_text(text)
    print(f"📝 Original text length: {len(cleaned_text)} characters")
    
    # Smart text splitting
    chunks = smart_text_split(cleaned_text, max_chars=120)
    print(f"🔄 Processing {len(chunks)} optimally split chunks...")
    
    # Settings for expressive speech
    settings = {
        "exaggeration": 0.7,
        "cfg_weight": 0.3,
        "temperature": 0.8
    }
    
    audio_segments = []
    successful_chunks = 0
    
    for i, chunk in enumerate(chunks):
        print(f"🎙️  [{i+1}/{len(chunks)}] {chunk[:60]}{'...' if len(chunk) > 60 else ''}")
        
        try:
            wav = model.generate(
                chunk,
                audio_prompt_path=reference_audio_path,
                **settings
            )
            
            # Trim silence from this chunk
            trimmed_wav = trim_silence(wav)
            audio_segments.append(trimmed_wav)
            successful_chunks += 1
            
        except Exception as e:
            print(f"❌ Error generating chunk {i+1}: {e}")
            # Add minimal silence for failed chunks
            silence = torch.zeros(1, int(0.2 * model.sr))
            audio_segments.append(silence)
    
    if len(audio_segments) == 0:
        print("❌ No audio generated")
        return None
    
    if len(audio_segments) == 1:
        final_audio = audio_segments[0]
    else:
        # Smoothly concatenate with crossfading
        print("🔗 Creating smooth audio transitions...")
        final_audio = audio_segments[0]
        
        for i, segment in enumerate(audio_segments[1:], 1):
            # Determine pause length based on context
            prev_chunk = chunks[i-1] if i-1 < len(chunks) else ""
            current_chunk = chunks[i] if i < len(chunks) else ""
            
            # Shorter pauses within sentences, longer between sentences
            if prev_chunk.endswith(('.', '!', '?')) or current_chunk[0].isupper():
                fade_length = 0.15  # Longer pause between sentences
            else:
                fade_length = 0.08  # Shorter pause within sentences
            
            final_audio = crossfade_audio(final_audio, segment, fade_length, model.sr)
    
    # Save the result
    ta.save(output_file, final_audio, model.sr)
    
    duration = final_audio.shape[1] / model.sr
    print(f"✅ Saved: {output_file}")
    print(f"⏱️  Duration: {duration:.1f} seconds")
    print(f"📊 Successfully processed {successful_chunks}/{len(chunks)} chunks")
    print(f"🎵 Smooth transitions applied between segments")
    
    return output_file

def main():
    # Setup
    model = setup_model()
    reference_audio_path = "voice_samples/sample_1.wav"
    
    if not os.path.exists(reference_audio_path):
        print(f"❌ Reference audio not found: {reference_audio_path}")
        return
    
    # Create output directory
    output_dir = "smooth_audio_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each text
    texts_to_process = [
        (TEXT, "text_1_smooth.wav"),
        (TEXT2, "text_2_smooth.wav"), 
        (TEXT3, "text_3_smooth.wav")
    ]
    
    all_output_files = []
    
    for text_content, output_name in texts_to_process:
        output_file = os.path.join(output_dir, output_name)
        result = process_text_to_smooth_audio(model, text_content, output_file, reference_audio_path)
        if result:
            all_output_files.append(result)
    
    print(f"\n🎉 Smooth audio processing complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎵 Generated {len(all_output_files)} smooth audio files:")
    for file in all_output_files:
        print(f"   📄 {file}")
    print(f"\n✨ These files should have much more natural transitions!")

if __name__ == "__main__":
    main() 