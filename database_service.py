"""
PostgreSQL Audio Extractor - Directly connects to database to extract audio files
"""
import pandas as pd
import psycopg2
import os
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
db_config = {
    'host': os.getenv("LEMURS_POSTGRES_HOST", "host"),
    'port': int(os.getenv("LEMURS_POSTGRES_PORT", 5432)),
    'dbname': os.getenv("LEMURS_POSTGRES_DB", 'your_database'),
    'user': os.getenv("LEMURS_POSTGRES_USER", 'your_username'),
    'password': os.getenv("LEMURS_POSTGRES_PASSWORD", 'your_password'),
}

class DatabaseService:
    """Class to extract data directly from PostgreSQL database"""
    def __init__(self,
                 host: str = "localhost",
                 port: int = 5432,
                 dbname: str = "your_database",
                 user: str = "your_username",
                 password: str = "your_password",
                 output_dir: str = "data/db_extracted_audio"):
        """
        Initialize the PostgreSQL audio extractor

        Args:
            host: Database host
            port: Database port
            dbname: Database name
            user: Database username
            password: Database password
            output_dir: Directory to save extracted audio files
        """
        self.host = os.getenv("LEMURS_POSTGRES_HOST", host)
        self.port = int(os.getenv("LEMURS_POSTGRES_PORT", port))
        self.dbname = os.getenv("LEMURS_POSTGRES_DB", dbname)
        self.user = os.getenv("LEMURS_POSTGRES_USER", user)
        self.password = os.getenv("LEMURS_POSTGRES_PASSWORD", password)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.connection = None

    def connect(self) -> bool:
        """Connect to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.dbname,
                user=self.user,
                password=self.password
            )
            logger.info(f"Successfully connected to PostgreSQL database at {self.host}:{self.port}/{self.dbname}")
            return True
        except psycopg2.Error as e:
            logger.error(f"Error connecting to PostgreSQL database: {e}")
            return False

    def disconnect(self):
        """Disconnect from database"""
        if self.connection:
            self.connection.close()
            logger.info("Disconnected from PostgreSQL database")

    def get_audio_records(self, table_name: str = "audio_response",
                         limit: Optional[int] = None) -> List[Dict]:
        """
        Fetch audio records from database

        Args:
            table_name: Name of the audio table
            limit: Optional limit on number of records

        Returns:
            List of audio record dictionaries
        """
        try:
            cursor = self.connection.cursor()

            # Build query
            query = f"""
                SELECT id, survey_response_id, audio_question_id, 
                       audio_data, timestamp, created_at
                FROM {table_name}
                WHERE audio_data IS NOT NULL 
                AND length(audio_data) > 1000
                ORDER BY id
            """

            if limit:
                query += f" LIMIT {limit}"

            cursor.execute(query)

            # Fetch all records
            records = cursor.fetchall()

            # Convert to list of dictionaries
            columns = ['id', 'survey_response_id', 'audio_question_id',
                      'audio_data', 'timestamp', 'created_at']

            audio_records = []
            for record in records:
                audio_dict = dict(zip(columns, record))
                audio_records.append(audio_dict)

            cursor.close()
            logger.info(f"Retrieved {len(audio_records)} audio records from database")

            return audio_records

        except psycopg2.Error as e:
            logger.error(f"Error fetching audio records: {e}")
            return []

    def detect_audio_format(self, audio_data: bytes) -> str:
        """
        Detect the format of audio data

        Args:
            audio_data: Binary audio data

        Returns:
            Detected format ('3gp', 'wav', 'mp3', 'unknown')
        """
        if not audio_data or len(audio_data) < 10:
            return 'unknown'

        # Check for 3GP format
        if b'ftyp3gp' in audio_data[:20]:
            return '3gp'

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

    def validate_audio_data(self, audio_data: bytes, format_type: str) -> tuple[bool, str]:
        """
        Validate audio data based on format

        Args:
            audio_data: Binary audio data
            format_type: Detected audio format

        Returns:
            Tuple of (is_valid, reason)
        """
        if not audio_data:
            return False, "No audio data"

        if len(audio_data) < 1000:
            return False, f"Audio data too small ({len(audio_data)} bytes)"

        if format_type == '3gp':
            if b'ftyp3gp' not in audio_data[:20]:
                return False, "Invalid 3GP signature"

        elif format_type == 'wav':
            if not audio_data.startswith(b'RIFF'):
                return False, "Invalid WAV signature"
            if b'WAVE' not in audio_data[:20]:
                return False, "Missing WAVE header"

        elif format_type == 'mp3':
            if not (audio_data.startswith(b'ID3') or audio_data.startswith(b'\xff\xfb')):
                return False, "Invalid MP3 signature"

        return True, f"Valid {format_type.upper()} file ({len(audio_data):,} bytes)"

    def generate_filename(self, record: Dict, format_type: str) -> str:
        """
        Generate filename based on record metadata

        Args:
            record: Database record dictionary
            format_type: Audio format for extension

        Returns:
            Generated filename
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
        if format_type == 'unknown':
            ext = 'bin'
        else:
            ext = format_type

        return f"db_audio_{audio_id}_survey_{survey_id}_q_{question_id}_{time_str}.{ext}"

    def save_audio_file(self, audio_data: bytes, output_path: str) -> bool:
        """
        Save audio data to file

        Args:
            audio_data: Binary audio data
            output_path: Output file path

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_path, 'wb') as f:
                f.write(audio_data)
            logger.info(f"Successfully saved: {output_path} ({len(audio_data):,} bytes)")
            return True
        except Exception as e:
            logger.error(f"Error saving audio file {output_path}: {e}")
            return False

    def convert_to_wav_if_possible(self, audio_data: bytes, original_path: str) -> Optional[str]:
        """
        Try to convert audio to WAV format using pydub if available

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
            wav_path = original_path.replace('.3gp', '_converted.wav').replace('.amr', '_converted.wav')

            # Export as WAV
            audio.export(wav_path, format="wav")

            logger.info(f"Successfully converted to WAV: {wav_path}")
            return wav_path

        except ImportError:
            logger.info("pydub not available - skipping WAV conversion")
            return None
        except Exception as e:
            logger.warning(f"Could not convert to WAV: {e}")
            return None

    def extract_all_audio(self, table_name: str = "audio_response",
                         limit: Optional[int] = None,
                         convert_to_wav: bool = True) -> List[str]:
        """
        Extract all audio files from database

        Args:
            table_name: Database table name
            limit: Optional limit on records to process
            convert_to_wav: Whether to also create WAV versions

        Returns:
            List of successfully created file paths
        """
        if not self.connect():
            return []

        try:
            # Get audio records
            records = self.get_audio_records(table_name, limit)

            if not records:
                logger.warning("No audio records found in database")
                return []

            successful_files = []

            for i, record in enumerate(records):
                try:
                    audio_data = record['audio_data']

                    # Handle different data types (bytes, memoryview, etc.)
                    if isinstance(audio_data, memoryview):
                        audio_data = audio_data.tobytes()
                    elif isinstance(audio_data, str):
                        # If it's a string, try to decode as binary
                        audio_data = audio_data.encode('latin-1')

                    # Detect format
                    format_type = self.detect_audio_format(audio_data)
                    logger.info(f"Processing record {i+1}/{len(records)}: ID={record['id']}, Format={format_type}")

                    # Validate
                    is_valid, reason = self.validate_audio_data(audio_data, format_type)
                    if not is_valid:
                        logger.warning(f"Skipping record {record['id']}: {reason}")
                        continue

                    # Generate filename and save
                    filename = self.generate_filename(record, format_type)
                    output_path = str(self.output_dir / filename)

                    if self.save_audio_file(audio_data, output_path):
                        successful_files.append(output_path)

                        # Try to convert to WAV if requested
                        if convert_to_wav and format_type in ['3gp', 'amr']:
                            wav_path = self.convert_to_wav_if_possible(audio_data, output_path)
                            if wav_path:
                                successful_files.append(wav_path)

                except Exception as e:
                    logger.error(f"Error processing record {record.get('id', 'unknown')}: {e}")
                    continue

            logger.info(f"Successfully extracted {len(successful_files)} audio files")
            return successful_files

        finally:
            self.disconnect()

    def extract_from_database(self, table_name: str) -> pd.DataFrame:
        """
        Screentime data extraction method
        """
        # Connect if not already connected
        if not self.connection:
            if not self.connect():
                raise Exception("Failed to connect to database")

        try:
            cursor = self.connection.cursor()

            # Build query
            query = f"""
                       SELECT *
                       FROM {table_name}
                       ORDER BY id
                   """

            cursor.execute(query)

            # Fetch all records
            records = cursor.fetchall()

            # Get column names
            columns = [desc[0] for desc in cursor.description]

            # Convert to dataframe
            df = pd.DataFrame(records, columns=columns)
            cursor.close()
            logger.info(f"Retrieved {len(df)} records from {table_name}")
            return df
        except Exception as e:
            logger.error(f"Error extracting data from {table_name}: {e}")
            raise


def main():

    print("PostgreSQL Audio Extractor")
    print("=" * 50)

    extractor = DatabaseService(**db_config)

    print("Starting audio extraction from PostgreSQL...")
    extracted_files = extractor.extract_all_audio(
        table_name="audio_response",  # Update table name if different
        limit=None,                   # Remove limit to process all records
        convert_to_wav=False          # Also create WAV versions
    )

    print(f"\nExtraction complete!")
    print(f"Total files extracted: {len(extracted_files)}")

    if extracted_files:
        print("\nExtracted files:")
        for file_path in extracted_files:
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            print(f"  - {file_path} ({file_size:,} bytes)")

if __name__ == "__main__":
    main()
