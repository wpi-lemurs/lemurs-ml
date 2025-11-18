'''
NOTE: sdgx is NOT compatible with python 3.13+. must use a lower version.
'''

from sdgx.data_connectors.csv_connector import CsvConnector
from sdgx.models.ml.single_table.ctgan import CTGANSynthesizerModel
from sdgx.synthesizer import Synthesizer
from sdgx.data_loader import DataLoader
from sdgx.data_processors.formatters.datetime import DatetimeFormatter
from health_data_analysis import unique_steps
from PHQ9_categorization import df as phq9
import pandas as pd

import warnings #warning are annoying

from src.database_service import DatabaseService

warnings.filterwarnings("ignore", category=FutureWarning, module="sdgx")


def generate_synthetic_data(df, num_samples, name):
    '''
    Generates synthetic data based on a dataframe.
    INPUT MUST BE REPRESENTATIVE OF REAL DATA.
    :param df: dataframe of original data
    :param num_samples: number of samples to generate
    :param name: name of the data (e.g., 'steps')
    '''
    df = df.copy()

    # convert any datetime columns to timestamp
    datetime_formats = {}
    format = '%Y-%m-%d %H:%M:%S.%f'

    # identify datetime columns
    for col in df.columns:
       # case 1: already datetime dtype
       if pd.api.types.is_datetime64_any_dtype(df[col]):
           datetime_formats[col] = format
           continue
        # case 2: try converting string/object columns to datetime
       if df[col].dtype == 'object':
           try:
               pd.to_datetime(df[col], errors='raise')
               datetime_formats[col] = format
           except Exception:
               pass  # not a datetime column

    # convert datetime columns to timestamp
    formatter = DatetimeFormatter()
    new_df = formatter.convert_datetime_columns(datetime_formats.keys(), datetime_formats, df)

    # convert df to csv
    csv_path = "new_df.csv"
    new_df.to_csv(csv_path, index=False)

    # create data connector for csv file
    data_connector = CsvConnector(path=csv_path)

    # initialize data loader
    dataloader = DataLoader(data_connector)

    # Access data
    dataloader.load_all()  # This will read all data from csv, and cache it.
    dataloader.load_all()  # This will read all data from cache.

    # Fit synthesizer, use CTGAN model
    synthesizer = Synthesizer(model=CTGANSynthesizerModel(epochs=1), data_connector=data_connector)
    synthesizer.fit()

    # Sample
    sampled_data = synthesizer.sample(num_samples)
    # convert timestamps back to datetime
    for col in datetime_formats.keys():
        sampled_data[col] = pd.to_datetime(sampled_data[col], unit='s')

    # export as csv
    sampled_data.to_csv(f"synthetic_{name}.csv", index=False)

# Create db service instance
service = DatabaseService()
# Extract all health data from database
steps_data = service.extract_from_database("step")

# Steps
# user testing generally had 10-200 steps/day. for 50 users across 4 weeks, this comes out to around 10,000 rows
generate_synthetic_data(unique_steps, num_samples=10000, name='steps')

# PHQ9
# each user should have 4. for 50 users, this would be 200 rows
generate_synthetic_data(phq9, num_samples=200, name='phq9')