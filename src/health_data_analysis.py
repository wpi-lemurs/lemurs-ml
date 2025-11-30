from src.database_service import DatabaseService
import pandas as pd
import os

# Create db service instance
service = DatabaseService()
# Extract all health data from database
steps_data = service.extract_from_database("step")
speed_data = service.extract_from_database("speed")
distance_data = service.extract_from_database("distance")
calorie_data = service.extract_from_database("calorie")

# Synthetic data (comment this out when not using)
# steps_data = pd.read_csv('../data/synthetic/synthetic_step_data.csv')

# Remove duplicate rows
# unique_steps = steps_data.drop_duplicates(subset='start_timestamp')
# unique_speed_data = speed_data.drop_duplicates(subset='start_timestamp')
# unique_distance_data = distance_data.drop_duplicates(subset='start_timestamp')
# unique_calorie_data = calorie_data.drop_duplicates(subset='start_timestamp')

def _process_health_dataframe(df, agg_func, time_unit, start_col='start_timestamp', value_col=None, app_user_id=-1, date_range=None):
    """
    Helper function to process a single health metric dataframe by aggregating to a specified time unit.

    Parameters:
    - df: DataFrame with health data
    - agg_func: aggregation function ('sum', 'mean', etc.)
    - time_unit: 'D' for daily, 'H' for hourly
    - start_col: name of timestamp column
    - value_col: name of numeric column to aggregate; if None, auto-detect
    - app_user_id: filter rows to this app_user_id; if -1, include all users
    - date_range: tuple of (start_date, end_date) to filter data. Dates can be strings or datetime objects.
                  If None, no date filtering is applied.

    Returns:
    - DataFrame with aggregated data by time unit, or None if input is empty/invalid
    """
    if df is None or df.empty:
        return None

    df = df.drop_duplicates(subset='start_timestamp').copy()

    # Check if app_user_id column exists
    has_app_user_id = 'app_user_id' in df.columns

    # Filter by app_user_id if needed
    if app_user_id != -1:
        if not has_app_user_id:
            raise KeyError("app_user_id column not found in DataFrame; cannot filter by app_user_id")
        df = df[df['app_user_id'] == app_user_id]

    # Parse timestamps
    df[start_col] = pd.to_datetime(df[start_col], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    df = df.dropna(subset=[start_col])

    # Filter by date range if specified
    if date_range is not None:
        if len(date_range) != 2:
            raise ValueError("date_range must be a tuple of (start_date, end_date)")
        start_date, end_date = date_range
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df = df[(df[start_col] >= start_date) & (df[start_col] <= end_date)]

    if df.empty:
        return None

    # Determine value column if not specified
    if value_col is None:
        numeric = df.select_dtypes(include='number').columns.tolist()
        if 'app_user_id' in numeric:
            numeric.remove('app_user_id')
        for preferred in ['steps', 'speed', 'distance', 'calories', 'value', 'count']:
            if preferred in numeric:
                value_col = preferred
                break
        if value_col is None:
            if not numeric:
                return None
            value_col = numeric[0]

    # Extract time grouping key based on time_unit
    if time_unit == 'H':
        df['time_key'] = df[start_col].dt.floor('h')
        group_col = 'time_key'
    elif time_unit == 'D':
        df['time_key'] = df[start_col].dt.date
        group_col = 'time_key'
    else:
        raise ValueError(f"Unsupported time_unit: {time_unit}. Use 'D' for daily or 'H' for hourly.")

    # Group by user (if exists) and time, then aggregate
    if has_app_user_id:
        aggregated = df.groupby(['app_user_id', group_col])[value_col].agg(agg_func).reset_index()
    else:
        aggregated = df.groupby(group_col)[value_col].agg(agg_func).reset_index()

    return aggregated


def _apply_null_handling(df, health_cols, method='linear', has_app_user_id=True):
    """
    Apply null value handling to health data columns using specified method.
    Handling is done per user to avoid filling across different users.

    Parameters:
    - df: DataFrame with health data
    - health_cols: list of column names to process
    - method: null handling method - 'linear' for linear interpolation, 'fill' for forward/backward filling, None for no handling
    - has_app_user_id: whether the dataframe has an app_user_id column

    Returns:
    - DataFrame with null values handled according to method
    """
    if df is None or df.empty:
        return df

    if method is None or method.lower() == 'none':
        return df

    df = df.copy()

    if has_app_user_id:
        # Process separately for each user to avoid handling across users
        processed_dfs = []
        for user_id, user_df in df.groupby('app_user_id'):
            for col in health_cols:
                if col in user_df.columns:
                    if method == 'linear':
                        # Linear interpolation for each user
                        user_df[col] = user_df[col].interpolate(method='linear', limit_direction='both')
                    elif method == 'fill':
                        # Forward then backward fill for each user
                        user_df[col] = user_df[col].ffill().bfill()
            processed_dfs.append(user_df)

        if processed_dfs:
            df = pd.concat(processed_dfs, ignore_index=True)
    else:
        # Process all data together if no user_id
        for col in health_cols:
            if col in df.columns:
                if method == 'linear':
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
                elif method == 'fill':
                    df[col] = df[col].ffill().bfill()

    return df


def weekly_avg_health_data(df, start_col='start_timestamp', target_col=None, week_anchor='MON', fill_missing=False, new_col_name='avg_daily_steps', app_user_id=-1, date_range=None):
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
    - date_range: tuple of (start_date, end_date) to filter data. Example: ('2025-01-01', '2025-12-31')

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

    # Filter by date range if specified
    if date_range is not None:
        if len(date_range) != 2:
            raise ValueError("date_range must be a tuple of (start_date, end_date)")
        start_date, end_date = date_range
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df = df[(df[start_col] >= start_date) & (df[start_col] <= end_date)]

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


def daily_health_with_week(steps_df=steps_data, speed_df=speed_data, distance_df=distance_data,
                           calorie_df=calorie_data, start_col='start_timestamp',
                           week_anchor='MON', app_user_id=-1, null_method=None, date_range=None):
    """
    Calculate daily totals for steps, distance, calories and daily average for speed,
    while keeping the week_start date column for each day.

    Parameters:
    - steps_df: DataFrame with steps data
    - speed_df: DataFrame with speed data
    - distance_df: DataFrame with distance data
    - calorie_df: DataFrame with calorie data
    - start_col: name of timestamp column (default 'start_timestamp')
    - week_anchor: weekday anchor for weekly grouping (e.g. 'MON', 'SUN')
    - app_user_id: filter rows to this app_user_id; if -1, include all users
    - null_method: method for handling null values - 'linear' for linear interpolation, 'fill' for forward/backward filling, None for no handling
    - date_range: tuple of (start_date, end_date) to filter data. Example: ('2025-01-01', '2025-12-31')

    Returns:
    - pandas.DataFrame with columns ['app_user_id', 'date', 'week_start', 'day_index',
      'daily_steps', 'daily_distance', 'daily_calories', 'daily_avg_speed']
      Each row represents one user's daily health metrics with associated week.
      day_index is 0 for the first day of each week, 1 for the second, etc.
    """

    # Process each health metric using the shared helper function
    daily_steps = _process_health_dataframe(steps_df, 'sum', 'D', start_col, app_user_id=app_user_id, date_range=date_range)
    daily_speed = _process_health_dataframe(speed_df, 'mean', 'D', start_col, app_user_id=app_user_id, date_range=date_range)
    daily_distance = _process_health_dataframe(distance_df, 'sum', 'D', start_col, app_user_id=app_user_id, date_range=date_range)
    daily_calories = _process_health_dataframe(calorie_df, 'sum', 'D', start_col, app_user_id=app_user_id, date_range=date_range)

    # Start with the first non-None dataframe
    result = None
    for df, col_name in [(daily_steps, 'daily_steps'),
                          (daily_distance, 'daily_distance'),
                          (daily_calories, 'daily_calories'),
                          (daily_speed, 'daily_avg_speed')]:
        if df is not None:
            # Rename time_key to date and value column to col_name
            df = df.rename(columns={'time_key': 'date', df.columns[-1]: col_name})
            if result is None:
                result = df
            else:
                # Merge on app_user_id and date, or just date if no app_user_id
                merge_cols = ['app_user_id', 'date'] if 'app_user_id' in result.columns else ['date']
                result = result.merge(df, on=merge_cols, how='outer')

    if result is None:
        return pd.DataFrame(columns=['app_user_id', 'date', 'week_start', 'day_index',
                                    'daily_steps', 'daily_distance', 'daily_calories', 'daily_avg_speed'])

    # Convert date back to datetime for week calculation
    result['date'] = pd.to_datetime(result['date'])

    # Calculate week_start for each date
    # Week starts on the specified anchor day
    freq = f'W-{week_anchor}'
    result['week_start'] = result['date'].dt.to_period(freq).dt.start_time

    # Calculate day index relative to each user's week (0-6)
    # Day 0 is the first day of the week (based on week_anchor)
    if 'app_user_id' in result.columns:
        # For each user and week, calculate day index
        result['day_index'] = (result['date'] - result['week_start']).dt.days
    else:
        # If no user ID, just calculate day index for all data
        result['day_index'] = (result['date'] - result['week_start']).dt.days

    # Reorder columns
    if 'app_user_id' in result.columns:
        cols = ['app_user_id', 'date', 'week_start', 'day_index']
    else:
        cols = ['date', 'week_start', 'day_index']

    for col in ['daily_steps', 'daily_distance', 'daily_calories', 'daily_avg_speed']:
        if col in result.columns:
            cols.append(col)

    result = result[cols]

    # Sort by app_user_id (if exists) and date
    sort_cols = ['app_user_id', 'date'] if 'app_user_id' in result.columns else ['date']
    result = result.sort_values(sort_cols).reset_index(drop=True)

    # Apply null handling if requested
    if null_method is not None:
        health_cols = ['daily_steps', 'daily_distance', 'daily_calories', 'daily_avg_speed']
        has_app_user_id = 'app_user_id' in result.columns
        result = _apply_null_handling(result, health_cols, null_method, has_app_user_id)

    return result

def hourly_health_data(steps_df=steps_data, speed_df=speed_data, distance_df=distance_data, cal_df=calorie_data, start_col='start_timestamp', week_anchor='MON', app_user_id=-1, null_method=None, date_range=None):
    """
    Calculate the hourly total steps, distance, calories and average speed for each user
    :param steps_df: DataFrame with steps data
    :param speed_df: DataFrame with speed data
    :param distance_df: DataFrame with distance data
    :param cal_df: DataFrame with calorie data
    :param start_col: name of timestamp column (default 'start_timestamp')
    :param week_anchor: weekday anchor for weekly grouping (e.g. 'MON', 'SUN')
    :param app_user_id: filter rows to this app_user_id; if -1, include all users
    :param null_method: method for handling null values - 'linear' for linear interpolation, 'fill' for forward/backward filling, None for no handling
    :param date_range: tuple of (start_date, end_date) to filter data. Example: ('2025-01-01', '2025-12-31')
    :return: pandas df with columns ['app_user_id', 'datetime', 'date', 'week_start', 'day_index', 'hour_index',
      'hourly_steps', 'hourly_distance', 'hourly_calories', 'hourly_avg_speed']
      Each row represents one user's hourly health metrics with the associated day and week.
      day_index is 0 for the first day of each week, 1 for the second, etc. hour_index is 0 for the first hour of the
      day where a user has data, 1 for the second, etc.
    """

    # Process each health metric using the shared helper function
    hourly_steps = _process_health_dataframe(steps_df, 'sum', 'H', start_col, app_user_id=app_user_id, date_range=date_range)
    hourly_speed = _process_health_dataframe(speed_df, 'mean', 'H', start_col, app_user_id=app_user_id, date_range=date_range)
    hourly_distance = _process_health_dataframe(distance_df, 'sum', 'H', start_col, app_user_id=app_user_id, date_range=date_range)
    hourly_calories = _process_health_dataframe(cal_df, 'sum', 'H', start_col, app_user_id=app_user_id, date_range=date_range)


    # Start with the first non-None dataframe
    result = None
    for df, col_name in [(hourly_steps, 'hourly_steps'),
                          (hourly_distance, 'hourly_distance'),
                          (hourly_calories, 'hourly_calories'),
                          (hourly_speed, 'hourly_avg_speed')]:
        if df is not None:
            # Rename time_key to datetime and value column to col_name
            df = df.rename(columns={'time_key': 'datetime', df.columns[-1]: col_name})
            if result is None:
                result = df
            else:
                # Merge on app_user_id and datetime, or just datetime if no app_user_id
                merge_cols = ['app_user_id', 'datetime'] if 'app_user_id' in result.columns else ['datetime']
                result = result.merge(df, on=merge_cols, how='outer')

    if result is None:
        return pd.DataFrame(columns=['app_user_id', 'datetime', 'date', 'week_start', 'day_index', 'hour_index',
                                    'hourly_steps', 'hourly_distance', 'hourly_calories', 'hourly_avg_speed'])

    # Extract date from datetime
    result['date'] = result['datetime'].dt.date
    result['date'] = pd.to_datetime(result['date'])

    # Calculate week_start for each date
    freq = f'W-{week_anchor}'
    result['week_start'] = result['date'].dt.to_period(freq).dt.start_time

    # Calculate day index relative to each user's week (0-6)
    if 'app_user_id' in result.columns:
        result['day_index'] = (result['date'] - result['week_start']).dt.days
    else:
        result['day_index'] = (result['date'] - result['week_start']).dt.days

    # Calculate hour index relative to each user's day
    # For each user and day, calculate hour index starting from 0
    if 'app_user_id' in result.columns:
        # Group by user and date, then assign hour index
        result = result.sort_values(['app_user_id', 'datetime'])
        result['hour_index'] = result.groupby(['app_user_id', 'date']).cumcount()
    else:
        result = result.sort_values(['datetime'])
        result['hour_index'] = result.groupby(['date']).cumcount()

    # Reorder columns
    if 'app_user_id' in result.columns:
        cols = ['app_user_id', 'datetime', 'date', 'week_start', 'day_index', 'hour_index']
    else:
        cols = ['datetime', 'date', 'week_start', 'day_index', 'hour_index']

    for col in ['hourly_steps', 'hourly_distance', 'hourly_calories', 'hourly_avg_speed']:
        if col in result.columns:
            cols.append(col)

    result = result[cols]

    # Sort by app_user_id (if exists) and datetime
    sort_cols = ['app_user_id', 'datetime'] if 'app_user_id' in result.columns else ['datetime']
    result = result.sort_values(sort_cols).reset_index(drop=True)

    # Apply null handling if requested
    if null_method is not None:
        health_cols = ['hourly_steps', 'hourly_distance', 'hourly_calories', 'hourly_avg_speed']
        has_app_user_id = 'app_user_id' in result.columns
        result = _apply_null_handling(result, health_cols, null_method, has_app_user_id)

    return result