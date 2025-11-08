import torch
from transformers import pipeline, WhisperConfig, WhisperModel
import pandas as pd
import os

# Check if CUDA is available, otherwise use CPU
device = 0 if torch.cuda.is_available() else -1
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Using device: {'CUDA' if device == 0 else 'CPU'}")

try:
    # Create the pipeline with proper device handling
    asr_pipeline = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        dtype=dtype,
        device=device
    )

except Exception as e:
    print(f"Error occurred: {e}")
    print("This might be due to model loading or other issues.")

# Create model configuration and model instance
configuration = WhisperConfig()
model = WhisperModel(configuration)

# Process audio files in the specified directory
audio_folder = "data/db_extracted_audio"
audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".3gp")]

# Store transcriptions
transcriptions = []

# Iterate over audio files and transcribe
for audio_file in audio_files:
    audio_path = os.path.join(audio_folder, audio_file)
    print(f"Processing: {audio_path}")

    try:
        result = asr_pipeline(audio_path)
        text = result["text"]
        print(f"→ Transcription: {text}\n")
        transcriptions.append({"file": audio_file, "transcription": text})
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")

# Save results to a CSV file
pd.DataFrame(transcriptions).to_csv("transcriptions.csv", index=False)