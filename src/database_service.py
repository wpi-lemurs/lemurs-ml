"""
Database Service - Handles PostgreSQL database connections and data extraction.

For audio extraction functionality, use src.data_processing.audio_processing module.
"""
import pandas as pd
import psycopg2
import os
from pathlib import Path
import logging
from typing import Dict, List, Optional
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
    """
    Class for database operations and data extraction.

    For audio extraction, use src.data_processing.audio_processing.extract_audio_from_database()
    """
    def __init__(self,
                 host: str = "localhost",
                 port: int = 5432,
                 dbname: str = "your_database",
                 user: str = "your_username",
                 password: str = "your_password",
                 output_dir: str = None):
        """
        Initialize the database service.

        Args:
            host: Database host
            port: Database port
            dbname: Database name
            user: Database username
            password: Database password
            output_dir: Directory for audio files (default: DB_AUDIO_DIR from config)
                       Note: For audio extraction, use audio_processing module instead
        """
        self.host = os.getenv("LEMURS_POSTGRES_HOST", host)
        self.port = int(os.getenv("LEMURS_POSTGRES_PORT", port))
        self.dbname = os.getenv("LEMURS_POSTGRES_DB", dbname)
        self.user = os.getenv("LEMURS_POSTGRES_USER", user)
        self.password = os.getenv("LEMURS_POSTGRES_PASSWORD", password)

        # Use centralized config for audio directory
        if output_dir is None:
            from src.config import DB_AUDIO_DIR
            self.output_dir = DB_AUDIO_DIR
        else:
            self.output_dir = Path(output_dir)

        # Directory is already created in config.py, but ensure it exists
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
        Fetch audio records from database.

        This method is primarily used by src.data_processing.audio_processing module.

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


    def extract_from_database(self, table_name: str) -> pd.DataFrame:
        """
        Extract data from a database table into a pandas DataFrame.

        This is the main method for extracting non-audio data (screentime, health metrics, etc.)

        Args:
            table_name: Name of the database table to extract

        Returns:
            DataFrame containing all records from the table

        Raises:
            Exception: If database connection fails or query execution fails
        """
        # Connect if not already connected
        if not self.connection or self.connection.closed:
            if not self.connect():
                raise Exception("Failed to connect to database")
        
        try:
            cursor = self.connection.cursor()

            # First check if 'id' column exists
            check_query = f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}' AND column_name = 'id'
            """
            cursor.execute(check_query)
            has_id = cursor.fetchone() is not None
            cursor.close()

            # Build query with ORDER BY only if id column exists
            if has_id:
                query = f"""
                           SELECT * 
                           FROM {table_name}
                           ORDER BY id
                           """
            else:
                query = f"""
                           SELECT * 
                           FROM {table_name}
                           """

            df = pd.read_sql(query, self.connection)
            logger.info(f"Retrieved {len(df)} records from {table_name}")
            return df

        except Exception as e:
            logger.error(f"Error extracting data from {table_name}: {e}")
            raise
