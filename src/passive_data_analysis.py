from src.database_service import DatabaseService
import pandas as pd

# Create db service instance
service = DatabaseService()
# Extract all health data from database
steps_data = service.extract_from_database("step")
speed_data = service.extract_from_database("speed")
distance_data = service.extract_from_database("distance")
calorie_data = service.extract_from_database("calorie")

# Synthetic data (comment this out when not using)
# steps_data = pd.read_csv('../data/synthetic/synthetic_step_data.csv')

# Extract screentime
screentime_data = service.extract_from_database("screentime")

# Remove duplicate rows
# unique_steps = steps_data.drop_duplicates(subset='start_timestamp')
# unique_speed_data = speed_data.drop_duplicates(subset='start_timestamp')
# unique_distance_data = distance_data.drop_duplicates(subset='start_timestamp')
# unique_calorie_data = calorie_data.drop_duplicates(subset='start_timestamp')

def _process_passive_data_dataframe(df, agg_func, time_unit, start_col='start_timestamp', value_col=None, app_user_id=-1, date_range=None):
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

    df = df.drop_duplicates(subset=start_col).copy()

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



def _add_week_metadata(df, date_col, week_anchor='MON'):
    """
    Add week_start and day_index columns to a dataframe based on a date column.

    Parameters:
    - df: DataFrame with date information
    - date_col: name of the date/datetime column
    - week_anchor: weekday anchor for weekly grouping (e.g. 'MON', 'SUN')

    Returns:
    - DataFrame with added 'week_start' and 'day_index' columns
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # Ensure date column is datetime
    if date_col in df.columns:
        if df[date_col].dtype == 'object':
            df[date_col] = pd.to_datetime(df[date_col])

    # Calculate week_start for each date
    freq = f'W-{week_anchor}'
    df['week_start'] = df[date_col].dt.to_period(freq).dt.start_time

    # Calculate day index relative to week (0-6)
    df['day_index'] = (df[date_col] - df['week_start']).dt.days

    return df


def _add_hour_index(df, datetime_col, date_col):
    """
    Add hour_index column to a dataframe, representing the hour within each day.
    Hour index starts from 0 for each day (per user if app_user_id exists).

    Parameters:
    - df: DataFrame with datetime information
    - datetime_col: name of the datetime column
    - date_col: name of the date column

    Returns:
    - DataFrame with added 'hour_index' column
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # Calculate hour index relative to each user's day
    if 'app_user_id' in df.columns:
        df = df.sort_values(['app_user_id', datetime_col])
        df['hour_index'] = df.groupby(['app_user_id', date_col]).cumcount()
    else:
        df = df.sort_values([datetime_col])
        df['hour_index'] = df.groupby([date_col]).cumcount()

    return df


def _apply_fill_method(df, health_cols, fill_method, has_app_user_id=True):
    """
    Apply the specified fill method to health data columns.

    Parameters:
    - df: DataFrame with health data
    - health_cols: list of column names to fill
    - fill_method: 'interpolate', 'ffill_bfill', or None
    - has_app_user_id: whether the dataframe has an app_user_id column

    Returns:
    - DataFrame with filled values
    """
    if not fill_method or df is None or df.empty:
        return df

    df = df.copy()

    if fill_method == 'interpolate':
        # Use linear interpolation
        if has_app_user_id:
            interpolated_dfs = []
            for user_id, user_df in df.groupby('app_user_id'):
                for col in health_cols:
                    if col in user_df.columns:
                        user_df[col] = user_df[col].interpolate(method='linear', limit_direction='both')
                interpolated_dfs.append(user_df)
            if interpolated_dfs:
                df = pd.concat(interpolated_dfs, ignore_index=True)
        else:
            for col in health_cols:
                if col in df.columns:
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')

    elif fill_method == 'ffill_bfill':
        # Use forward fill then backward fill
        if has_app_user_id:
            filled_dfs = []
            for user_id, user_df in df.groupby('app_user_id'):
                for col in health_cols:
                    if col in user_df.columns:
                        user_df[col] = user_df[col].ffill().bfill()
                filled_dfs.append(user_df)
            if filled_dfs:
                df = pd.concat(filled_dfs, ignore_index=True)
        else:
            for col in health_cols:
                if col in df.columns:
                    df[col] = df[col].ffill().bfill()

    else:
        raise ValueError(f"Invalid fill_method: {fill_method}. Use None, 'interpolate', or 'ffill_bfill'.")

    return df


def _merge_health_dataframes(dataframes_with_names, merge_key_cols):
    """
    Merge multiple health dataframes together using outer join.

    Parameters:
    - dataframes_with_names: list of tuples (df, column_name) where df is a processed dataframe
      and column_name is the name to give to the value column
    - merge_key_cols: list of column names to merge on (e.g., ['app_user_id', 'date'])

    Returns:
    - Merged DataFrame or None if all input dataframes are None
    """
    result = None

    for df, col_name in dataframes_with_names:
        if df is not None:
            # Rename the value column (last column) to col_name
            df = df.rename(columns={df.columns[-1]: col_name})

            if result is None:
                result = df
            else:
                result = result.merge(df, on=merge_key_cols, how='outer')

    return result


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
                           week_anchor='MON', app_user_id=-1, fill_method=None, date_range=None):
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
    - fill_method: method to fill null values. Options:
        - None: leave null values as is
        - 'interpolate': apply linear interpolation
        - 'ffill_bfill': apply forward fill then backward fill
    - date_range: tuple of (start_date, end_date) to filter data. Example: ('2025-01-01', '2025-12-31')

    Returns:
    - pandas.DataFrame with columns ['app_user_id', 'date', 'week_start', 'day_index',
      'daily_steps', 'daily_distance', 'daily_calories', 'daily_avg_speed']
      Each row represents one user's daily health metrics with associated week.
      day_index is 0 for the first day of each week, 1 for the second, etc.
    """

    # Process each health metric using the shared helper function
    daily_steps = _process_passive_data_dataframe(steps_df, 'sum', 'D', start_col, app_user_id=app_user_id, date_range=date_range)
    daily_speed = _process_passive_data_dataframe(speed_df, 'mean', 'D', start_col, app_user_id=app_user_id, date_range=date_range)
    daily_distance = _process_passive_data_dataframe(distance_df, 'sum', 'D', start_col, app_user_id=app_user_id, date_range=date_range)
    daily_calories = _process_passive_data_dataframe(calorie_df, 'sum', 'D', start_col, app_user_id=app_user_id, date_range=date_range)

    # Prepare dataframes for merging - rename time_key to date
    dataframes_with_names = []
    for df, col_name in [(daily_steps, 'daily_steps'),
                          (daily_distance, 'daily_distance'),
                          (daily_calories, 'daily_calories'),
                          (daily_speed, 'daily_avg_speed')]:
        if df is not None:
            df = df.rename(columns={'time_key': 'date'})
            dataframes_with_names.append((df, col_name))

    # Determine merge columns
    has_app_user_id = any(df is not None and 'app_user_id' in df.columns
                          for df, _ in [(daily_steps, ''), (daily_speed, ''),
                                       (daily_distance, ''), (daily_calories, '')])
    merge_cols = ['app_user_id', 'date'] if has_app_user_id else ['date']

    # Merge all dataframes
    result = _merge_health_dataframes(dataframes_with_names, merge_cols)

    if result is None:
        return pd.DataFrame(columns=['app_user_id', 'date', 'week_start', 'day_index',
                                    'daily_steps', 'daily_distance', 'daily_calories', 'daily_avg_speed'])

    # Convert date to datetime and add week metadata
    result['date'] = pd.to_datetime(result['date'])
    result = _add_week_metadata(result, 'date', week_anchor)

    # Reorder columns
    base_cols = ['app_user_id', 'date', 'week_start', 'day_index'] if has_app_user_id else ['date', 'week_start', 'day_index']
    health_cols = [col for col in ['daily_steps', 'daily_distance', 'daily_calories', 'daily_avg_speed']
                   if col in result.columns]
    result = result[base_cols + health_cols]

    # Sort by app_user_id (if exists) and date
    sort_cols = ['app_user_id', 'date'] if has_app_user_id else ['date']
    result = result.sort_values(sort_cols).reset_index(drop=True)

    # Apply fill method if requested
    result = _apply_fill_method(result, health_cols, fill_method, has_app_user_id)

    return result

def hourly_health_data(steps_df=steps_data, speed_df=speed_data, distance_df=distance_data, cal_df=calorie_data, start_col='start_timestamp', week_anchor='MON', app_user_id=-1, fill_method=None, date_range=None):
    """
    Calculate the hourly total steps, distance, calories and average speed for each user
    :param steps_df: DataFrame with steps data
    :param speed_df: DataFrame with speed data
    :param distance_df: DataFrame with distance data
    :param cal_df: DataFrame with calorie data
    :param start_col: name of timestamp column (default 'start_timestamp')
    :param week_anchor: weekday anchor for weekly grouping (e.g. 'MON', 'SUN')
    :param app_user_id: filter rows to this app_user_id; if -1, include all users
    :param date_range: tuple of (start_date, end_date) to filter data. Example: ('2025-01-01', '2025-12-31')
    :param fill_method: method to fill null values. Options:
        - None: leave null values as is
        - 'interpolate': apply linear interpolation
        - 'ffill_bfill': apply forward fill then backward fill
    :return: pandas df with columns ['app_user_id', 'datetime', 'date', 'week_start', 'day_index', 'hour_index',
      'hourly_steps', 'hourly_distance', 'hourly_calories', 'hourly_avg_speed']
      Each row represents one user's hourly health metrics with the associated day and week.
      day_index is 0 for the first day of each week, 1 for the second, etc. hour_index is 0 for the first hour of the
      day where a user has data, 1 for the second, etc.
    """

    # Process each health metric using the shared helper function
    hourly_steps = _process_passive_data_dataframe(steps_df, 'sum', 'H', start_col, app_user_id=app_user_id, date_range=date_range)
    hourly_speed = _process_passive_data_dataframe(speed_df, 'mean', 'H', start_col, app_user_id=app_user_id, date_range=date_range)
    hourly_distance = _process_passive_data_dataframe(distance_df, 'sum', 'H', start_col, app_user_id=app_user_id, date_range=date_range)
    hourly_calories = _process_passive_data_dataframe(cal_df, 'sum', 'H', start_col, app_user_id=app_user_id, date_range=date_range)

    # Prepare dataframes for merging - rename time_key to datetime
    dataframes_with_names = []
    for df, col_name in [(hourly_steps, 'hourly_steps'),
                          (hourly_distance, 'hourly_distance'),
                          (hourly_calories, 'hourly_calories'),
                          (hourly_speed, 'hourly_avg_speed')]:
        if df is not None:
            df = df.rename(columns={'time_key': 'datetime'})
            dataframes_with_names.append((df, col_name))

    # Determine merge columns
    has_app_user_id = any(df is not None and 'app_user_id' in df.columns
                          for df, _ in [(hourly_steps, ''), (hourly_speed, ''),
                                       (hourly_distance, ''), (hourly_calories, '')])
    merge_cols = ['app_user_id', 'datetime'] if has_app_user_id else ['datetime']

    # Merge all dataframes
    result = _merge_health_dataframes(dataframes_with_names, merge_cols)

    if result is None:
        return pd.DataFrame(columns=['app_user_id', 'datetime', 'date', 'week_start', 'day_index', 'hour_index',
                                    'hourly_steps', 'hourly_distance', 'hourly_calories', 'hourly_avg_speed'])

    # Extract date from datetime and add week metadata
    result['date'] = result['datetime'].dt.date
    result['date'] = pd.to_datetime(result['date'])
    result = _add_week_metadata(result, 'date', week_anchor)

    # Add hour index
    result = _add_hour_index(result, 'datetime', 'date')

    # Reorder columns
    base_cols = (['app_user_id', 'datetime', 'date', 'week_start', 'day_index', 'hour_index']
                 if has_app_user_id else ['datetime', 'date', 'week_start', 'day_index', 'hour_index'])
    health_cols = [col for col in ['hourly_steps', 'hourly_distance', 'hourly_calories', 'hourly_avg_speed']
                   if col in result.columns]
    result = result[base_cols + health_cols]

    # Sort by app_user_id (if exists) and datetime
    sort_cols = ['app_user_id', 'datetime'] if has_app_user_id else ['datetime']
    result = result.sort_values(sort_cols).reset_index(drop=True)

    # Apply fill method if requested
    result = _apply_fill_method(result, health_cols, fill_method, has_app_user_id)

    return result

def hourly_screentime_data(screentime_df=screentime_data, start_col='start_time', week_anchor='MON', app_user_id=-1, fill_method=None):
    """
    Calculate the hourly total screentime for each user
    :param screentime_df: DataFrame with screentime data
    :param start_col: name of timestamp column (default 'start_timestamp')
    :param week_anchor: weekday anchor for weekly grouping (e.g. 'MON', 'SUN')
    :param app_user_id: filter rows to this app_user_id; if -1, include all users
    :param fill_method: method to fill null values. Options:
        - None: leave null values as is
        - 'interpolate': apply linear interpolation
        - 'ffill_bfill': apply forward fill then backward fill
    :return: pandas df with columns ['app_user_id', 'datetime', 'date', 'week_start', 'day_index', 'hour_index',
      'hourly_screentime']
      Each row represents one user's hourly screentime with the associated day and week.
      day_index is 0 for the first day of each week, 1 for the second, etc. hour_index is 0 for the first hour of the
      day where a user has data, 1 for the second, etc.
    """

    # Process screentime using the shared helper function
    hourly_screentime = _process_passive_data_dataframe(screentime_df, 'sum', 'H', start_col, app_user_id=app_user_id)

    # Prepare dataframes for merging - rename time_key to datetime
    dataframes_with_names = []
    if hourly_screentime is not None:
        hourly_screentime = hourly_screentime.rename(columns={'time_key': 'datetime'})
        dataframes_with_names.append((hourly_screentime, 'hourly_screentime'))

    # Determine if we have app_user_id
    has_app_user_id = hourly_screentime is not None and 'app_user_id' in hourly_screentime.columns
    merge_cols = ['app_user_id', 'datetime'] if has_app_user_id else ['datetime']

    # Merge (in this case just one dataframe, but using the helper for consistency)
    result = _merge_health_dataframes(dataframes_with_names, merge_cols)

    if result is None:
        return pd.DataFrame(columns=['app_user_id', 'datetime', 'date', 'week_start', 'day_index', 'hour_index',
                                    'hourly_screentime'])

    # Extract date from datetime and add week metadata
    result['date'] = result['datetime'].dt.date
    result['date'] = pd.to_datetime(result['date'])
    result = _add_week_metadata(result, 'date', week_anchor)

    # Add hour index
    result = _add_hour_index(result, 'datetime', 'date')

    # Reorder columns
    base_cols = (['app_user_id', 'datetime', 'date', 'week_start', 'day_index', 'hour_index']
                 if has_app_user_id else ['datetime', 'date', 'week_start', 'day_index', 'hour_index'])
    screentime_cols = [col for col in ['hourly_screentime'] if col in result.columns]
    result = result[base_cols + screentime_cols]

    # Sort by app_user_id (if exists) and datetime
    sort_cols = ['app_user_id', 'datetime'] if has_app_user_id else ['datetime']
    result = result.sort_values(sort_cols).reset_index(drop=True)

    # Apply fill method if requested
    result = _apply_fill_method(result, screentime_cols, fill_method, has_app_user_id)

    return result


def daily_screentime_data(screentime_df=screentime_data, start_col='start_time', week_anchor='MON', app_user_id=-1, fill_method=None, date_range=None):
    """
    Calculate the daily total screentime for each user
    :param screentime_df: DataFrame with screentime data
    :param start_col: name of timestamp column (default 'start_time')
    :param week_anchor: weekday anchor for weekly grouping (e.g. 'MON', 'SUN')
    :param app_user_id: filter rows to this app_user_id; if -1, include all users
    :param fill_method: method to fill null values. Options:
        - None: leave null values as is
        - 'interpolate': apply linear interpolation
        - 'ffill_bfill': apply forward fill then backward fill
    :param date_range: tuple of (start_date, end_date) to filter data. Example: ('2025-01-01', '2025-12-31')
    :return: pandas df with columns ['app_user_id', 'date', 'week_start', 'day_index', 'daily_screentime']
      Each row represents one user's daily screentime with the associated week.
      day_index is 0 for the first day of each week, 1 for the second, etc.
    """

    # Process screentime using the shared helper function
    daily_screentime = _process_passive_data_dataframe(screentime_df, 'sum', 'D', start_col, app_user_id=app_user_id, date_range=date_range)

    # Prepare dataframes for merging - rename time_key to date
    dataframes_with_names = []
    if daily_screentime is not None:
        daily_screentime = daily_screentime.rename(columns={'time_key': 'date'})
        dataframes_with_names.append((daily_screentime, 'daily_screentime'))

    # Determine if we have app_user_id
    has_app_user_id = daily_screentime is not None and 'app_user_id' in daily_screentime.columns
    merge_cols = ['app_user_id', 'date'] if has_app_user_id else ['date']

    # Merge (in this case just one dataframe, but using the helper for consistency)
    result = _merge_health_dataframes(dataframes_with_names, merge_cols)

    if result is None:
        return pd.DataFrame(columns=['app_user_id', 'date', 'week_start', 'day_index', 'daily_screentime'])

    # Convert date to datetime and add week metadata
    result['date'] = pd.to_datetime(result['date'])
    result = _add_week_metadata(result, 'date', week_anchor)

    # Reorder columns
    base_cols = (['app_user_id', 'date', 'week_start', 'day_index']
                 if has_app_user_id else ['date', 'week_start', 'day_index'])
    screentime_cols = [col for col in ['daily_screentime'] if col in result.columns]
    result = result[base_cols + screentime_cols]

    # Sort by app_user_id (if exists) and date
    sort_cols = ['app_user_id', 'date'] if has_app_user_id else ['date']
    result = result.sort_values(sort_cols).reset_index(drop=True)

    # Apply fill method if requested
    result = _apply_fill_method(result, screentime_cols, fill_method, has_app_user_id)

    return result