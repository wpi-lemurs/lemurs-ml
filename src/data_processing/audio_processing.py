"""
Audio processing module for extracting and processing audio files from the database.
Contains reusable functions for audio extraction, validation, and conversion.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import io

from src.database_service import DatabaseService
from src.config import DB_AUDIO_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_audio_format(audio_data: bytes) -> str:
    """
    Detect the format of audio data by examining file signatures.

    Args:
        audio_data: Binary audio data

    Returns:
        Detected format ('3gp', 'wav', 'mp3', 'amr', 'm4a', 'aac', 'caf', 'unknown')
    """
    if not audio_data or len(audio_data) < 10:
        return 'unknown'

    # Check for 3GP format (Android)
    if b'ftyp3gp' in audio_data[:20]:
        return '3gp'

    # Check for M4A format (iOS) - MPEG-4 Audio
    # M4A files use the same container as MP4 but with audio-specific ftyp brands
    if b'ftyp' in audio_data[:12]:
        # Check for M4A specific brands
        if (b'M4A ' in audio_data[:20] or b'M4B ' in audio_data[:20] or
            b'mp42' in audio_data[:20] or b'isom' in audio_data[:20]):
            return 'm4a'

    # Check for CAF format (iOS Core Audio Format)
    if audio_data.startswith(b'caff'):
        return 'caf'

    # Check for AAC format (iOS/Android)
    # AAC can have ADTS headers
    if len(audio_data) >= 2:
        # ADTS sync word: 0xFFF (12 bits) at start of frame
        if audio_data[0] == 0xFF and (audio_data[1] & 0xF0) == 0xF0:
            return 'aac'

    # Check for WAV format
    if audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:20]:
        return 'wav'

    # Check for MP3 format
    if audio_data.startswith(b'ID3') or audio_data.startswith(b'\xff\xfb'):
        return 'mp3'

    # Check for AMR format (common in mobile recordings)
    if audio_data.startswith(b'#!AMR'):
        return 'amr'

    return 'unknown'


def validate_audio_data(audio_data: bytes, format_type: str) -> Tuple[bool, str]:
    """
    Validate audio data based on format and file signatures.

    Args:
        audio_data: Binary audio data
        format_type: Detected audio format

    Returns:
        Tuple of (is_valid: bool, reason: str)
    """
    if not audio_data:
        return False, "No audio data"

    if len(audio_data) < 1000:
        return False, f"Audio data too small ({len(audio_data)} bytes)"

    if format_type == '3gp':
        if b'ftyp3gp' not in audio_data[:20]:
            return False, "Invalid 3GP signature"

    elif format_type == 'm4a':
        if b'ftyp' not in audio_data[:12]:
            return False, "Invalid M4A signature"
        # Additional validation: check for presence of audio-related atoms
        if not (b'M4A ' in audio_data[:50] or b'mp42' in audio_data[:50] or
                b'isom' in audio_data[:50] or b'mdat' in audio_data[:100]):
            return False, "Missing M4A audio markers"

    elif format_type == 'caf':
        if not audio_data.startswith(b'caff'):
            return False, "Invalid CAF signature"
        # CAF files should have version info after signature
        if len(audio_data) < 8:
            return False, "CAF file too short"

    elif format_type == 'aac':
        if len(audio_data) < 2:
            return False, "AAC data too short"
        # Validate ADTS sync word
        if not (audio_data[0] == 0xFF and (audio_data[1] & 0xF0) == 0xF0):
            return False, "Invalid AAC ADTS header"

    elif format_type == 'wav':
        if not audio_data.startswith(b'RIFF'):
            return False, "Invalid WAV signature"
        if b'WAVE' not in audio_data[:20]:
            return False, "Missing WAVE header"

    elif format_type == 'mp3':
        if not (audio_data.startswith(b'ID3') or audio_data.startswith(b'\xff\xfb')):
            return False, "Invalid MP3 signature"

    return True, f"Valid {format_type.upper()} file ({len(audio_data):,} bytes)"


def generate_audio_filename(record: Dict, format_type: str) -> str:
    """
    Generate a descriptive filename based on record metadata.

    Args:
        record: Database record dictionary with audio metadata
        format_type: Audio format for file extension

    Returns:
        Generated filename string
    """
    audio_id = record.get('id', 'unknown')
    survey_id = record.get('survey_response_id', 'unknown')
    question_id = record.get('audio_question_id', 'unknown')

    # Format timestamp
    timestamp = record.get('timestamp') or record.get('created_at')
    if timestamp:
        try:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = timestamp
            time_str = dt.strftime('%Y%m%d_%H%M%S')
        except:
            time_str = 'unknown_time'
    else:
        time_str = 'unknown_time'

    # Choose extension
    ext = format_type if format_type != 'unknown' else 'bin'

    return f"db_audio_{audio_id}_survey_{survey_id}_q_{question_id}_{time_str}.{ext}"


def save_audio_file(audio_data: bytes, output_path: Path) -> bool:
    """
    Save audio data to a file.

    Args:
        audio_data: Binary audio data
        output_path: Output file path (as Path object or string)

    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            f.write(audio_data)

        logger.info(f"Successfully saved: {output_path} ({len(audio_data):,} bytes)")
        return True
    except Exception as e:
        logger.error(f"Error saving audio file {output_path}: {e}")
        return False


def convert_to_wav(audio_data: bytes, original_path: Path) -> Optional[Path]:
    """
    Try to convert audio to WAV format using pydub if available.

    Args:
        audio_data: Original audio data
        original_path: Path to original file

    Returns:
        Path to WAV file if conversion successful, None otherwise
    """
    try:
        from pydub import AudioSegment

        # Try to load the audio
        audio_io = io.BytesIO(audio_data)
        audio = AudioSegment.from_file(audio_io)

        # Generate WAV filename
        wav_path = original_path.with_suffix('.wav')
        if wav_path == original_path:
            wav_path = original_path.parent / f"{original_path.stem}_converted.wav"

        # Export as WAV
        audio.export(str(wav_path), format="wav")
        logger.info(f"Successfully converted to WAV: {wav_path}")
        return wav_path
    except ImportError:
        logger.info("pydub not available - skipping WAV conversion")
        return None
    except Exception as e:
        logger.warning(f"Could not convert to WAV: {e}")
        return None


def extract_audio_from_database(
    table_name: str = "audio_response",
    limit: Optional[int] = None,
    convert_wav: bool = False,
    output_dir: Optional[Path] = None,
    db_service: Optional[DatabaseService] = None
) -> List[Path]:
    """
    Extract audio files from the database.

    This is the main audio extraction function that:
    1. Connects to the database
    2. Retrieves audio records
    3. Validates and processes each audio file
    4. Saves files to the output directory
    5. Optionally converts to WAV format

    Args:
        table_name: Database table name containing audio data (default: "audio_response")
        limit: Optional limit on number of records to process
        convert_wav: Whether to also create WAV versions of the files
        output_dir: Directory to save files (default: DB_AUDIO_DIR from config)
        db_service: Optional DatabaseService instance (creates new one if None)

    Returns:
        List of Path objects for successfully created files

    Example:
        >>> # Extract all audio files
        >>> files = extract_audio_from_database()
        >>> print(f"Extracted {len(files)} audio files")

        >>> # Extract with limit and WAV conversion
        >>> files = extract_audio_from_database(limit=10, convert_wav=True)
    """
    # Use default output directory if not specified
    if output_dir is None:
        output_dir = DB_AUDIO_DIR
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create database service if not provided
    if db_service is None:
        db_service = DatabaseService(output_dir=str(output_dir))

    # Connect to database
    if not db_service.connect():
        logger.error("Failed to connect to database")
        return []

    try:
        # Get audio records from database
        records = db_service.get_audio_records(table_name, limit)

        if not records:
            logger.warning("No audio records found in database")
            return []

        logger.info(f"Processing {len(records)} audio records...")
        successful_files = []

        for i, record in enumerate(records, 1):
            try:
                audio_data = record['audio_data']

                # Handle different data types (bytes, memoryview, etc.)
                if isinstance(audio_data, memoryview):
                    audio_data = audio_data.tobytes()
                elif isinstance(audio_data, str):
                    audio_data = audio_data.encode('latin-1')

                # Detect format
                format_type = detect_audio_format(audio_data)
                logger.info(f"Processing record {i}/{len(records)}: ID={record['id']}, Format={format_type}")

                # Validate
                is_valid, reason = validate_audio_data(audio_data, format_type)
                if not is_valid:
                    logger.warning(f"Skipping record {record['id']}: {reason}")
                    continue

                # Generate filename and save
                filename = generate_audio_filename(record, format_type)
                output_path = output_dir / filename

                if save_audio_file(audio_data, output_path):
                    successful_files.append(output_path)

                    # Try to convert to WAV if requested
                    # Include iOS formats (m4a, aac, caf) and Android formats (3gp, amr)
                    if convert_wav and format_type in ['3gp', 'amr', 'm4a', 'aac', 'caf']:
                        wav_path = convert_to_wav(audio_data, output_path)
                        if wav_path:
                            successful_files.append(wav_path)

            except Exception as e:
                logger.error(f"Error processing record {record.get('id', 'unknown')}: {e}")
                continue

        logger.info(f"Successfully extracted {len(successful_files)} audio files to {output_dir}")
        return successful_files

    finally:
        # Always disconnect
        db_service.disconnect()

if __name__ == "__main__":
    # Extract audio files
    print("\nExtracting audio files...")
    extracted_files = extract_audio_from_database(
        limit=None,  # No limit - extract all
        convert_wav=False  # Set to True if you want WAV conversion
    )

    print(f"\nExtraction complete!")
    print(f"Total files extracted: {len(extracted_files)}")