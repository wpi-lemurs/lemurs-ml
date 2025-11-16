from database_service import DatabaseService
import pandas as pd

# Create db service instance
service = DatabaseService()
# Extract all health data from database
steps_data = service.extract_from_database("step")
speed_data = service.extract_from_database("speed")
distance_data = service.extract_from_database("distance")
calorie_data = service.extract_from_database("calorie")

# Remove duplicate rows
unique_steps = steps_data.drop_duplicates(subset='start_timestamp')
unique_speed_data = speed_data.drop_duplicates(subset='start_timestamp')
unique_distance_data = distance_data.drop_duplicates(subset='start_timestamp')
unique_calorie_data = calorie_data.drop_duplicates(subset='start_timestamp')

def weekly_avg_health_data(df, start_col='start_timestamp', target_col=None, week_anchor='MON', fill_missing=False, new_col_name='avg_daily_steps', user_id=-1):
    """
    Parse timestamps with milliseconds, optionally filter by user ID, aggregate to daily totals,
    then compute average per-day over weekly chunks.

    Parameters:
    - df: pandas.DataFrame containing at least the timestamp column and a numeric target column.
    - start_col: name of timestamp column (default 'start_timestamp').
    - target_col: name of numeric column to aggregate; if None, the function selects the first numeric column.
    - week_anchor: weekday anchor for weekly resampling (e.g. 'MON', 'SUN').
    - fill_missing: if True, fill missing daily values with 0 before weekly averaging.
    - new_col_name: name for the resulting average column.
    - user_id: filter rows to this user_id; if -1, do not filter (i.e., include all users).

    Returns:
    - pandas.DataFrame with columns ['week_start', new_col_name]
    """
    if df is None:
        raise ValueError("df must be a pandas DataFrame")

    df = df.copy()

    # If requested, filter by user_id. -1 means include all users.
    if user_id != -1:
        if 'user_id' not in df.columns:
            raise KeyError("user_id column not found in DataFrame; cannot filter by user_id")
        df = df[df['user_id'] == user_id]

    # strict parse for timestamps like 2025-09-24 10:45:43.221
    df[start_col] = pd.to_datetime(df[start_col], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    # drop rows that failed to parse
    df = df.dropna(subset=[start_col])

    if target_col is None:
        numeric = df.select_dtypes(include='number').columns.tolist()
        # prefer common column names if present
        for preferred in ['steps', 'value', 'count']:
            if preferred in numeric:
                target_col = preferred
                break
        if target_col is None:
            if not numeric:
                raise ValueError("No numeric column found; set `target_col` explicitly.")
            target_col = numeric[0]

    # set datetime index and aggregate to daily totals
    df = df.set_index(start_col)
    daily = df[target_col].resample('D').sum()

    if fill_missing:
        daily = daily.fillna(0)

    freq = f'W-{week_anchor}'
    # resample by week and compute mean daily value for the week
    weekly = daily.resample(freq, label='left', closed='left').mean()
    weekly = weekly.rename(new_col_name).reset_index().rename(columns={start_col: 'week_start'})

    return weekly


# convenience wrappers that require an explicit DataFrame to avoid DB access on import
def weekly_avg_steps(df=unique_steps, **kwargs):
    return weekly_avg_health_data(df, new_col_name='avg_daily_steps', **kwargs)


def weekly_avg_speed(df=unique_speed_data, **kwargs):
    return weekly_avg_health_data(df, new_col_name='avg_daily_speed', **kwargs)


def weekly_avg_distance(df=unique_distance_data, **kwargs):
    return weekly_avg_health_data(df, new_col_name='avg_daily_distance', **kwargs)


def weekly_avg_calorie(df=unique_calorie_data, **kwargs):
    return weekly_avg_health_data(df, new_col_name='avg_daily_calories', **kwargs)

print(weekly_avg_steps().head())
print(weekly_avg_speed().head())
print(weekly_avg_steps().head())
print(weekly_avg_calorie().head())