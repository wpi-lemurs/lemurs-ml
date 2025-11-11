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

unique_steps = unique_steps.copy()

# convert datetime columns to timestamp
formatter = DatetimeFormatter()
datetime_formats = {
    'start_timestamp': '%Y-%m-%d %H:%M:%S.%f',
    'end_timestamp': '%Y-%m-%d %H:%M:%S.%f',
    'recorded_date': '%Y-%m-%d %H:%M:%S.%f'
}
new_df = formatter.convert_datetime_columns(['start_timestamp', 'end_timestamp', 'recorded_date'],datetime_formats,unique_steps)

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

# Initialize synthesizer, use CTGAN model
synthesizer = Synthesizer(
   model=CTGANSynthesizerModel(epochs=1),  # For quick demo
   data_connector=data_connector,
)

# Fit the model
synthesizer.fit()

# Sample
sampled_data = synthesizer.sample(400)
# convert back to datetime
sampled_data['start_timestamp'] = pd.to_datetime(sampled_data['start_timestamp'], unit='s')
sampled_data['end_timestamp'] = pd.to_datetime(sampled_data['end_timestamp'], unit='s')
sampled_data['recorded_date'] = pd.to_datetime(sampled_data['recorded_date'], unit='s')
print(sampled_data)

# save sampled_data to csv file
sampled_data.to_csv("synthetic_steps.csv", index=False)

