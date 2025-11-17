'''
NOTE: sdgx is NOT compatible with python 3.13+. must use a lower version.
'''

from sdgx.data_connectors.csv_connector import CsvConnector
from sdgx.models.ml.single_table.ctgan import CTGANSynthesizerModel
from sdgx.synthesizer import Synthesizer
from sdgx.data_loader import DataLoader
from sdgx.data_processors.formatters.datetime import DatetimeFormatter
from steps_analysis import unique_steps
import pandas as pd

def generate_synthetic_data(df, num_samples):
    '''
    Generates synthetic data based on a dataframe.
    INPUT MUST BE REPRESENTATIVE OF REAL DATA.
    :param df: dataframe of original data
    :param num_samples: number of samples to generate
    :return: new dataframe of synthetic data
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

    return df

# Steps
synthetic_steps = generate_synthetic_data(unique_steps, num_samples=400)
print(synthetic_steps)