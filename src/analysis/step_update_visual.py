from src.database_service import DatabaseService
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# extract data from database
service = DatabaseService()
steps_data = service.extract_from_database("step")

# constants
UPDATE_DATE = pd.to_datetime("2025-12-15")
USERS_WITH_UPDATE = [16,17,18,19]

# ----------------
# DAILY DATA
# ----------------
# prepare daily aggregated step data by start time
steps_data['date'] = pd.to_datetime(steps_data['start_timestamp']).dt.date
daily_steps = (steps_data.groupby(['app_user_id', 'date'])['steps'].sum().reset_index(name='daily_steps'))

daily_steps['date'] = pd.to_datetime(daily_steps['date'])

# plot the users that updated the app
all_dates = pd.date_range(daily_steps['date'].min(), daily_steps['date'].max())

plt.figure(figsize=(12, 6))
for user_id in USERS_WITH_UPDATE:
    user_df = daily_steps[daily_steps['app_user_id'] == user_id][['date', 'daily_steps']].set_index('date')
    user_df = user_df.reindex(all_dates)
    plt.plot(all_dates, user_df['daily_steps'], marker='o', label=f'User {user_id}')

plt.axvline(UPDATE_DATE, color='red', linestyle='--', label='Update (2025-12-15)')
plt.xlabel('Date')
plt.ylabel('Daily Steps')
plt.title('Daily Steps per User with Updated App')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join('analysis_outputs', "daily_steps_plot.png"))

# ----------------
# HOURLY DATA
# ----------------

# prepare hourly aggregated step data by start time
steps_data['date_by_hour'] = pd.to_datetime(steps_data['start_timestamp']).dt.floor('h')

start = UPDATE_DATE - pd.Timedelta(days=7)
end = UPDATE_DATE + pd.Timedelta(days=7)
steps_window = steps_data[(steps_data['date_by_hour'] >= start) & (steps_data['date_by_hour'] < end + pd.Timedelta(days=1))]

hourly_steps = (steps_window.groupby(['app_user_id', 'date_by_hour'])['steps'].sum().reset_index(name='hourly_steps'))
all_hours = pd.date_range(start=start, end=end + pd.Timedelta(days=1), freq='h')

plt.figure(figsize=(14, 6))
for user_id in USERS_WITH_UPDATE:
    user_df = hourly_steps[hourly_steps['app_user_id'] == user_id].set_index('date_by_hour')
    user_df = user_df.reindex(all_hours)
    plt.plot(all_hours, user_df['hourly_steps'], marker='o', linestyle='-', label=f'User {user_id}')

plt.axvline(UPDATE_DATE, color='red', linestyle='--', label='Update (2025-12-15)')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.xlabel('Hour')
plt.ylabel('Steps')
plt.title('Hourly Steps per User with Updated App (7 days before and after update)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join('analysis_outputs', "hourly_steps_plot.png"))