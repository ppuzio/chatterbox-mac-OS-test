#!/usr/bin/env python3
"""
Process long texts by maximizing chunk length to minimize concatenation artifacts
"""

import os
import sys
import torch
import torchaudio as ta
import re
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

def estimate_tokens(text):
    """Rough estimation of tokens - approximately 4 characters per token"""
    return len(text) // 4

def find_optimal_chunk_size(model, reference_audio_path, test_sentences):
    """Find the maximum text length that doesn't get truncated"""
    print("🔍 Finding optimal chunk size...")
    
    settings = {"exaggeration": 0.7, "cfg_weight": 0.3, "temperature": 0.8}
    
    # Test with increasing lengths
    test_lengths = [200, 400, 600, 800, 1000, 1200, 1500, 2000]
    max_safe_length = 200
    
    base_text = " ".join(test_sentences)
    
    for length in test_lengths:
        if len(base_text) < length:
            test_text = (base_text + " ") * ((length // len(base_text)) + 1)
        else:
            test_text = base_text
            
        test_chunk = test_text[:length]
        if not test_chunk.endswith(('.', '!', '?')):
            # Find the last sentence ending
            last_period = max(test_chunk.rfind('.'), test_chunk.rfind('!'), test_chunk.rfind('?'))
            if last_period > length // 2:  # Don't cut too much
                test_chunk = test_chunk[:last_period + 1]
        
        print(f"   Testing {len(test_chunk)} chars (~{estimate_tokens(test_chunk)} tokens)...")
        
        try:
            # Generate with the test chunk
            wav1 = model.generate(test_chunk, audio_prompt_path=reference_audio_path, **settings)
            duration1 = wav1.shape[1] / model.sr
            
            # Generate with the same text repeated to see if it gets longer
            double_text = test_chunk + " " + test_chunk
            wav2 = model.generate(double_text, audio_prompt_path=reference_audio_path, **settings)
            duration2 = wav2.shape[1] / model.sr
            
            # If doubling the text doesn't significantly increase duration, we hit the limit
            ratio = duration2 / duration1
            print(f"     Duration: {duration1:.1f}s, Double: {duration2:.1f}s, Ratio: {ratio:.2f}")
            
            if ratio < 1.5:  # If doubling text doesn't at least 1.5x the duration, we're hitting limits
                print(f"   ⚠️ Approaching token limit at {len(test_chunk)} characters")
                break
            else:
                max_safe_length = len(test_chunk)
                print(f"   ✅ {len(test_chunk)} characters is safe")
                
        except Exception as e:
            print(f"   ❌ Error at {len(test_chunk)} characters: {e}")
            break
    
    print(f"📏 Optimal chunk size: {max_safe_length} characters (~{estimate_tokens(str(max_safe_length))} tokens)")
    return max_safe_length

def split_text_optimally(text, max_chunk_size):
    """Split text into the longest possible chunks without exceeding the limit"""
    # Split into sentences first
    sentences = re.split(r'([.!?]+)', text)
    
    # Recombine sentences with their punctuation
    complete_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = sentences[i] + sentences[i + 1]
            if sentence.strip():
                complete_sentences.append(sentence.strip())
    
    # Combine sentences into optimal chunks
    chunks = []
    current_chunk = ""
    
    for sentence in complete_sentences:
        # Check if adding this sentence would exceed the limit
        if len(current_chunk + " " + sentence) <= max_chunk_size:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            # Save current chunk if it has content
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Start new chunk with current sentence
            if len(sentence) <= max_chunk_size:
                current_chunk = sentence
            else:
                # If single sentence is too long, split it at commas
                parts = sentence.split(', ')
                temp_chunk = ""
                for part in parts:
                    if len(temp_chunk + ", " + part) <= max_chunk_size:
                        if temp_chunk:
                            temp_chunk += ", " + part
                        else:
                            temp_chunk = part
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip() + ".")
                        temp_chunk = part
                current_chunk = temp_chunk
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def process_text_with_optimal_chunks(model, text, output_file, reference_audio_path, max_chunk_size):
    """Process text using optimal chunk sizes"""
    print(f"\n🎭 Processing with optimal chunks: {output_file}")
    
    # Clean the text
    cleaned_text = clean_text(text)
    print(f"📝 Original text length: {len(cleaned_text)} characters")
    
    # Split into optimal chunks
    chunks = split_text_optimally(cleaned_text, max_chunk_size)
    print(f"🔄 Split into {len(chunks)} optimal chunks (avg {sum(len(c) for c in chunks)//len(chunks)} chars each)")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"   Chunk {i}: {len(chunk)} chars - {chunk[:60]}{'...' if len(chunk) > 60 else ''}")
    
    # Settings for expressive speech
    settings = {
        "exaggeration": 0.7,
        "cfg_weight": 0.3,
        "temperature": 0.8
    }
    
    audio_segments = []
    successful_chunks = 0
    
    for i, chunk in enumerate(chunks):
        print(f"\n🎙️  [{i+1}/{len(chunks)}] Generating {len(chunk)} chars...")
        
        try:
            wav = model.generate(
                chunk,
                audio_prompt_path=reference_audio_path,
                **settings
            )
            
            duration = wav.shape[1] / model.sr
            print(f"     Generated: {duration:.1f} seconds")
            
            audio_segments.append(wav)
            successful_chunks += 1
            
        except Exception as e:
            print(f"❌ Error generating chunk {i+1}: {e}")
            # Add short silence for failed chunks
            silence = torch.zeros(1, int(0.5 * model.sr))
            audio_segments.append(silence)
    
    if len(audio_segments) == 0:
        print("❌ No audio generated")
        return None
    
    if len(audio_segments) == 1:
        final_audio = audio_segments[0]
    else:
        # Simple concatenation with minimal, natural pauses
        print("🔗 Concatenating with minimal gaps...")
        
        # Use very short, natural pauses
        short_pause = torch.zeros(1, int(0.15 * model.sr))  # 150ms pause
        
        final_audio = audio_segments[0]
        for segment in audio_segments[1:]:
            final_audio = torch.cat([final_audio, short_pause, segment], dim=1)
    
    # Save the result
    ta.save(output_file, final_audio, model.sr)
    
    duration = final_audio.shape[1] / model.sr
    print(f"✅ Saved: {output_file}")
    print(f"⏱️  Total duration: {duration:.1f} seconds")
    print(f"📊 Successfully processed {successful_chunks}/{len(chunks)} chunks")
    print(f"🎵 Using {len(chunks)} concatenation points (minimized)")
    
    return output_file

def main():
    # Setup
    model = setup_model()
    reference_audio_path = "voice_samples/sample_1.wav"
    
    if not os.path.exists(reference_audio_path):
        print(f"❌ Reference audio not found: {reference_audio_path}")
        return
    
    # Find optimal chunk size using sample sentences
    # Try to use actual content from text.py, fall back to generic sentences
    try:
        # Extract some sentences from the actual texts for testing
        all_texts = [TEXT, TEXT2, TEXT3]
        test_sentences = []
        
        for text in all_texts:
            if text and text.strip():
                # Split into sentences and take first few
                import re
                sentences = re.split(r'[.!?]+', clean_text(text))
                sentences = [s.strip() + "." for s in sentences if s.strip()]
                test_sentences.extend(sentences[:2])  # Take first 2 sentences from each text
                
        # If we don't have enough sentences, use fallback
        if len(test_sentences) < 3:
            test_sentences = [
                "This is a sample sentence for testing the voice cloning system.",
                "The quick brown fox jumps over the lazy dog, demonstrating clear speech.",
                "Hello, this is a test of the text-to-speech processing capabilities.",
                "We are testing the optimal chunk size for generating natural speech."
            ]
    except:
        # Fallback to generic test sentences
        test_sentences = [
            "This is a sample sentence for testing the voice cloning system.",
            "The quick brown fox jumps over the lazy dog, demonstrating clear speech.",
            "Hello, this is a test of the text-to-speech processing capabilities.",
            "We are testing the optimal chunk size for generating natural speech."
        ]
    
    optimal_size = find_optimal_chunk_size(model, reference_audio_path, test_sentences)
    
    # Create output directory
    output_dir = "optimal_audio_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each text with optimal chunking
    texts_to_process = [
        (TEXT, "text_1_optimal.wav"),
        (TEXT2, "text_2_optimal.wav"), 
        (TEXT3, "text_3_optimal.wav")
    ]
    
    all_output_files = []
    
    for text_content, output_name in texts_to_process:
        output_file = os.path.join(output_dir, output_name)
        result = process_text_with_optimal_chunks(
            model, text_content, output_file, reference_audio_path, optimal_size
        )
        if result:
            all_output_files.append(result)
    
    print(f"\n🎉 Optimal chunk processing complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎵 Generated {len(all_output_files)} optimized audio files:")
    for file in all_output_files:
        print(f"   📄 {file}")
    print(f"\n✨ These files should have minimal concatenation artifacts!")

if __name__ == "__main__":
    main() 