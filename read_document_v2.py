#!/usr/bin/env python3
"""
OpenVoice V2 - Much better quality!
"""
import os
import torch
from melo.api import TTS
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import soundfile as sf

# ============== SETUP ==============
VOICE_FILE = "my_voice.wav"
TEXT_CONTENT = """The Art of Clean Code seminar was memorable because it approached the topic of clean code in a practical and interactive way rather than as a purely theoretical concept. I arrived late to the event, so I made a conscious effort to pay close attention once I entered the room."""

OUTPUT_FILE = "outputs/my_voice_reading_v2.wav"

# ============== DON'T CHANGE BELOW ==============
def main():
    if not os.path.exists(VOICE_FILE):
        print(f"Error: {VOICE_FILE} not found!")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load V2 converter
    ckpt_converter = "checkpoints_v2/checkpoints_v2/converter"
    print("\nLoading V2 converter...")
    tone_color_converter = ToneColorConverter(f"{ckpt_converter}/config.json", device=device)
    tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")
    
    # Get your voice embedding
    print("Extracting your voice embedding...")
    target_se, audio_name = se_extractor.get_se(
        VOICE_FILE, 
        tone_color_converter, 
        target_dir="processed_v2", 
        vad=True
    )
    
    # Use MeloTTS for base speaker
    print("\nLoading MeloTTS...")
    model = TTS(language="EN", device=device)
    speaker_ids = model.hps.data.spk2id
    speaker_key = list(speaker_ids.keys())[0]
    speaker_id = speaker_ids[speaker_key]
    
    # Load base speaker embedding
    source_se = torch.load("checkpoints_v2/checkpoints_v2/base_speakers/ses/en-default.pth", map_location=device)
    
    # Generate speech
    os.makedirs("outputs", exist_ok=True)
    temp_path = "outputs/temp_v2.wav"
    SPEED = 0.85  # 0.80-0.90 range
    
    print(f"Generating speech at speed {SPEED}...")
    model.tts_to_file(text=TEXT_CONTENT, speaker_id=speaker_id, speed=SPEED, output_path=temp_path)
    
    # Convert to your voice - lower tau = more like your voice
    print("Converting to your voice...")
    tone_color_converter.convert(
        audio_src_path=temp_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=OUTPUT_FILE,
        tau=0.05,  # Very low = strong voice cloning
        message="@MyShell"
    )
    
    os.remove(temp_path)
    
    print(f"\nDone! Saved to: {OUTPUT_FILE}")
    print("Play with: ffplay outputs/my_voice_reading_v2.wav")

if __name__ == "__main__":
    main()