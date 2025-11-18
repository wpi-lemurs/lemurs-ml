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
# unique_steps = steps_data.drop_duplicates(subset='start_timestamp')
# unique_speed_data = speed_data.drop_duplicates(subset='start_timestamp')
# unique_distance_data = distance_data.drop_duplicates(subset='start_timestamp')
# unique_calorie_data = calorie_data.drop_duplicates(subset='start_timestamp')

def weekly_avg_health_data(df, start_col='start_timestamp', target_col=None, week_anchor='MON', fill_missing=False, new_col_name='avg_daily_steps', app_user_id=-1):
    """
    Parse timestamps with milliseconds, optionally filter by user ID, aggregate to daily totals,
    then compute average per-day over weekly chunks. Returns one row per user per week.

    Parameters:
    - df: pandas.DataFrame containing at least the timestamp column and a numeric target column.
    - start_col: name of timestamp column (default 'start_timestamp').
    - target_col: name of numeric column to aggregate; if None, the function selects the first numeric column.
    - week_anchor: weekday anchor for weekly resampling (e.g. 'MON', 'SUN').
    - fill_missing: if True, fill missing daily values with 0 before weekly averaging.
    - new_col_name: name for the resulting average column.
    - app_user_id: filter rows to this app_user_id; if -1, process all users separately (i.e., include all users).

    Returns:
    - pandas.DataFrame with columns ['app_user_id', 'week_start', new_col_name]
      Each row represents one user's average daily value for one week.
    """
    if df is None:
        raise ValueError("df must be a pandas DataFrame")

    df = df.drop_duplicates(subset='start_timestamp').copy()

    # Check if app_user_id column exists
    has_app_user_id = 'app_user_id' in df.columns

    # If requested, filter by app_user_id. -1 means process all users.
    if app_user_id != -1:
        if not has_app_user_id:
            raise KeyError("app_user_id column not found in DataFrame; cannot filter by app_user_id")
        df = df[df['app_user_id'] == app_user_id]

    # strict parse for timestamps like 2025-09-24 10:45:43.221
    df[start_col] = pd.to_datetime(df[start_col], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    # drop rows that failed to parse
    df = df.dropna(subset=[start_col])

    if target_col is None:
        numeric = df.select_dtypes(include='number').columns.tolist()
        # Remove app_user_id from numeric columns if present
        if 'app_user_id' in numeric:
            numeric.remove('app_user_id')
        # prefer common column names if present
        for preferred in ['steps', 'value', 'count']:
            if preferred in numeric:
                target_col = preferred
                break
        if target_col is None:
            if not numeric:
                raise ValueError("No numeric column found; set `target_col` explicitly.")
            target_col = numeric[0]

    # If we have app_user_id column, group by user
    if has_app_user_id:
        results = []
        for uid, user_df in df.groupby('app_user_id'):
            # set datetime index and aggregate to daily totals
            user_df = user_df.set_index(start_col)
            daily = user_df[target_col].resample('D').sum()

            if fill_missing:
                daily = daily.fillna(0)

            freq = f'W-{week_anchor}'
            # resample by week and compute mean daily value for the week
            weekly = daily.resample(freq, label='left', closed='left').mean()
            weekly_df = weekly.reset_index()
            weekly_df.columns = ['week_start', new_col_name]
            weekly_df['app_user_id'] = uid
            results.append(weekly_df)

        if results:
            final_df = pd.concat(results, ignore_index=True)
            # Reorder columns to have app_user_id first
            final_df = final_df[['app_user_id', 'week_start', new_col_name]]
        else:
            # Return empty DataFrame with correct columns
            final_df = pd.DataFrame(columns=['app_user_id', 'week_start', new_col_name])
    else:
        # No app_user_id column, process all data together
        df = df.set_index(start_col)
        daily = df[target_col].resample('D').sum()

        if fill_missing:
            daily = daily.fillna(0)

        freq = f'W-{week_anchor}'
        # resample by week and compute mean daily value for the week
        weekly = daily.resample(freq, label='left', closed='left').mean()
        final_df = weekly.reset_index()
        final_df.columns = ['week_start', new_col_name]

    return final_df


# convenience wrappers that require an explicit DataFrame to avoid DB access on import
def weekly_avg_steps(df=steps_data, **kwargs):
    return weekly_avg_health_data(df, new_col_name='avg_daily_steps', **kwargs)


def weekly_avg_speed(df=speed_data, **kwargs):
    return weekly_avg_health_data(df, new_col_name='avg_daily_speed', **kwargs)


def weekly_avg_distance(df=distance_data, **kwargs):
    return weekly_avg_health_data(df, new_col_name='avg_daily_distance', **kwargs)


def weekly_avg_calorie(df=calorie_data, **kwargs):
    return weekly_avg_health_data(df, new_col_name='avg_daily_calories', **kwargs)