"""
Central configuration module for the LEMURS ML project.
Defines project-wide paths to ensure consistency across all modules.
"""
from pathlib import Path

# Project root directory (this file is in src/, so go up one level)
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DB_AUDIO_DIR = DATA_DIR / "db_extracted_audio"

# Ensure data directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Other configuration constants can be added here
# For example:
# MODEL_DIR = PROJECT_ROOT / "models"
# LOGS_DIR = PROJECT_ROOT / "logs"

