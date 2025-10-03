import torch
from transformers import pipeline, WhisperConfig, WhisperModel
import pandas as pd

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

screentime_data = pd.read_csv('screentime_data.csv')
screentime_app_data = pd.read_csv('screentime_app_data.csv')
