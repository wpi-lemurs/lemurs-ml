# Lemurs Machine Learning Repository

This repository provides tools for extracting data for machine learning workflows for the LEMURS MQP

## Requirements
Please create a virtual environment using the command:
``` bash
python -m venv venv
```

Activate the virtual environment:
- On Windows:
``` bash
venv\Scripts\activate
```
- On macOS/Linux:
``` bash
source venv/bin/activate
```

Install the required packages:
``` bash
pip install -r requirements.txt
```

## Environment Configuration
Create a `.env` file in the project root with your database credentials:

``` bash
LEMURS_POSTGRES_HOST=your_db_host
LEMURS_POSTGRES_PORT=5432
LEMURS_POSTGRES_DB=your_database_name
LEMURS_POSTGRES_USER=your_db_user
LEMURS_POSTGRES_PASSWORD=your_db_password
```

You can get these from anyone on the lemurs team.

## Usage

### Extract Audio from PostgreSQL
Run the extraction script (database_service.py) to pull audio files from the database:
Extracted audio files will be saved in the `db_extracted_audio/` directory. WAV conversions (if enabled) will also be saved there.


