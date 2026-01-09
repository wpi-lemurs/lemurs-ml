import torch
from transformers import pipeline, WhisperConfig, WhisperModel
import pandas as pd
from src.config import DATA_DIR, DB_AUDIO_DIR

# Use centralized data directories
data_dir = DATA_DIR
audio_folder = DB_AUDIO_DIR

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
audio_files = [f for f in audio_folder.iterdir() if f.suffix == ".3gp"]

# Store transcriptions
transcriptions = []

# Iterate over audio files and transcribe
for audio_file in audio_files:
    print(f"Processing: {audio_file}")

    try:
        result = asr_pipeline(str(audio_file))
        text = result["text"]
        print(f"→ Transcription: {text}\n")
        transcriptions.append({"file": audio_file.name, "transcription": text})
    except Exception as e:
        print(f"Error processing {audio_file.name}: {e}")

# Save results to a CSV file in data directory
output_path = data_dir / "transcriptions.csv"
pd.DataFrame(transcriptions).to_csv(output_path, index=False)
print(f"Transcriptions saved to: {output_path}")

