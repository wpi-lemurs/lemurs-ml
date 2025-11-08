from database_service import DatabaseService
import pandas as pd

# Create db service instance
service = DatabaseService()
# Extract steps data from database
steps_data = service.extract_from_database("step")

# Remove duplicate rows
unique_steps = steps_data.drop_duplicates(subset='start_timestamp')

def weekly_avg_steps(df, start_col='start_timestamp', steps_col=None, week_anchor='MON', fill_missing=False):
    """
    Parse timestamps with milliseconds, aggregate to daily totals, then compute average steps-per-day per week.
    """
    df = df.copy()
    # strict parse for timestamps like 2025-09-24 10:45:43.221
    df[start_col] = pd.to_datetime(df[start_col], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    # drop rows that failed to parse
    df = df.dropna(subset=[start_col])

    if steps_col is None:
        numeric = df.select_dtypes(include='number').columns.tolist()
        if not numeric:
            raise ValueError("No numeric column found; set `steps_col` explicitly.")
        steps_col = numeric[0]

    df = df.set_index(start_col)
    daily = df[steps_col].resample('D').sum()

    if fill_missing:
        daily = daily.fillna(0)

    freq = f'W-{week_anchor}'
    weekly = daily.resample(freq, label='left', closed='left').mean()
    weekly = weekly.rename('avg_daily_steps').reset_index().rename(columns={start_col: 'week_start'})

    return weekly

print(weekly_avg_steps(unique_steps))