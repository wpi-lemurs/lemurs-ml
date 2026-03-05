import pandas as pd
import matplotlib.pyplot as plt

'''
We asked study participants to update the app on 2025-12-15 in order to (theoretically) improve data collection.
This script visualizes the impact of the update on daily steps for users who had data at/after the update date. 
'''

# Load the CSV table
df = pd.read_csv('C:/Users/emast/Downloads/MQP/lemurs-ml/data/daily_health_and_phq9_data.csv', parse_dates=['date'])

# Find users with at least one row on/after the update date

UPDATE_DATE = pd.to_datetime('2025-12-15')

users_after_update = df.loc[df['date'] >= UPDATE_DATE, 'app_user_id'].unique()

# Prepare complete date index for the whole dataset
all_dates = pd.date_range(df['date'].min(), df['date'].max())

plt.figure(figsize=(12, 6))
for user_id in users_after_update:
    user_df = df[df['app_user_id'] == user_id][['date', 'daily_steps']]
    # Reindex by all_dates so gaps/missing days show up
    user_df = user_df.set_index('date').reindex(all_dates)
    # Optionally, to show gaps as zero, uncomment:
    user_df['daily_steps'] = user_df['daily_steps'].fillna(0)
    plt.plot(all_dates, user_df['daily_steps'], marker='o', label=f'User {user_id}')

plt.axvline(UPDATE_DATE, color='red', linestyle='--', label='Update (2025-12-15)')

plt.xlabel('Date')
plt.ylabel('Daily Steps')
plt.title('Daily Steps per User (show users who have data at/after update)')
plt.legend()
plt.tight_layout()
plt.show()