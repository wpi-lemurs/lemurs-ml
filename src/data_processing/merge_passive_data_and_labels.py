from src.data_processing.passive_data_analysis import *
from src.categorization.PHQ9_categorization_binary import *
from src.categorization.suicide_risk_labels import *
from src.categorization.daily_questions_categorization import get_daily_labels_dataframe
from src.config import DATA_DIR
from functools import reduce

# No need to create data_dir here - it's already created in config.py
data_dir = DATA_DIR

# Load data with synthetic data paths (just for testing)
# steps_data = pd.read_csv(os.path.join(current_dir, 'data', 'synthetic', 'synthetic_step_data.csv'))
# phq9_data = pd.read_csv(os.path.join(current_dir, 'data', 'synthetic', 'synthetic_phq9_data.csv'))

# We eventually will use this instead
weekly_steps = weekly_avg_steps()
phq9_data = get_phq9_dataframe()
weekly_speed = weekly_avg_speed()
weekly_calorie = weekly_avg_calorie()
weekly_distance = weekly_avg_distance()
suicide_risk_data = get_suicide_risk_dataframe()

# Load comprehensive daily labels (includes suicide, self-harm, and sleep risk labels)
daily_labels_data = get_daily_labels_dataframe()

def propagate_positive_labels(data, label_col, positive_class):
    """
    Propagate positive labels to all entries for users with at least one positive label.

    This function treats any user who has been labeled as 'depressed' or 'at_risk'
    at least once as having that label for ALL of their entries. This helps address
    severe class imbalance when there are very few positive cases.

    Parameters:
    - data: DataFrame with merged health data and labels
    - label_col: name of the label column (e.g., 'severity_label' or 'suicide_risk_label')
    - positive_class: the positive class value to propagate (e.g., 'depressed' or 'at_risk')

    Returns:
    - DataFrame with propagated labels
    """
    if data.empty or label_col not in data.columns:
        return data

    data_copy = data.copy()

    # Find users who have at least one positive label
    users_with_positive_label = data_copy[
        data_copy[label_col] == positive_class
    ]['app_user_id'].unique()

    print(f"\nLabel Propagation Summary:")
    print(f"  Total unique users: {data_copy['app_user_id'].nunique()}")
    print(f"  Users with at least one '{positive_class}' label: {len(users_with_positive_label)}")
    print(f"  Original '{positive_class}' entries: {(data_copy[label_col] == positive_class).sum()}")

    # Set all entries for these users to positive class
    data_copy.loc[
        data_copy['app_user_id'].isin(users_with_positive_label),
        label_col
    ] = positive_class

    print(f"  After propagation '{positive_class}' entries: {(data_copy[label_col] == positive_class).sum()}")
    print(f"  Label distribution after propagation:")
    print(f"    {data_copy[label_col].value_counts().to_dict()}")

    return data_copy


def _prepare_phq9_weekly(phq9_df, week_anchor='MON'):
    """
    Helper function to prepare PHQ-9 data for merging with health data.

    Processes PHQ-9 data to calculate week_start for each response and aggregates
    by user and week.

    Parameters:
    - phq9_df: DataFrame with PHQ-9 survey data including 'app_user_id', 'timestamp', 'severity_label'
    - week_anchor: weekday anchor for weekly grouping (default 'MON')

    Returns:
    - DataFrame with columns ['app_user_id', 'week_start', 'severity_label', 'phq9_total_score']
    """
    # Make a copy to avoid modifying original data
    phq9_copy = phq9_df.copy()

    # Parse PHQ-9 response timestamps
    phq9_copy['timestamp'] = pd.to_datetime(phq9_copy['timestamp'], errors='coerce')
    phq9_copy = phq9_copy.dropna(subset=['timestamp'])

    # Calculate the week start for each PHQ-9 response using the same method as health data
    freq = f'W-{week_anchor}'
    phq9_weekly_list = []
    for user_id, user_df in phq9_copy.groupby('app_user_id'):
        for _, row in user_df.iterrows():
            ts = pd.Timestamp(row['timestamp'])
            # Use to_period to match the health data calculation
            week_start = ts.to_period(freq).start_time

            phq9_weekly_list.append({
                'app_user_id': user_id,
                'week_start': week_start,
                'severity_label': row['severity_label'],
                'phq9_total_score': row['phq9_total_score']
            })

    phq9_weekly = pd.DataFrame(phq9_weekly_list)

    # If multiple PHQ-9 responses exist in the same week for a user, take the first one
    phq9_weekly = phq9_weekly.groupby(['app_user_id', 'week_start']).agg({
        'severity_label': 'first',
        'phq9_total_score': 'first'
    }).reset_index()

    return phq9_weekly

def _prepare_risk_labels_per_survey(risk_labels_df, label_column):
    """
    Generic helper function to prepare risk labels for merging with health/screentime data.

    Unlike PHQ-9 which we aggregate weekly, risk labels are kept at the survey level
    and matched to the nearest survey for each data point.

    Parameters:
    - risk_labels_df: DataFrame with risk data including 'app_user_id', 'timestamp', and the specified label column
    - label_column: name of the risk label column (e.g., 'suicide_risk_label', 'self_harm_risk_label', 'sleep_label')

    Returns:
    - DataFrame with columns ['survey_response_id', 'app_user_id', 'timestamp', label_column]
    """
    # Make a copy to avoid modifying original data
    risk_labels_copy = risk_labels_df.copy()

    # Parse timestamps
    risk_labels_copy['timestamp'] = pd.to_datetime(risk_labels_copy['timestamp'], errors='coerce')
    risk_labels_copy = risk_labels_copy.dropna(subset=['timestamp'])

    # Check if label column exists
    if label_column not in risk_labels_copy.columns:
        print(f"Warning: '{label_column}' not found in risk labels dataframe")
        return pd.DataFrame()

    # Keep only necessary columns
    columns_to_keep = ['survey_response_id', 'app_user_id', 'timestamp', label_column]
    # Filter to only existing columns
    columns_to_keep = [col for col in columns_to_keep if col in risk_labels_copy.columns]
    risk_labels_copy = risk_labels_copy[columns_to_keep]

    # Drop rows where the label is null
    risk_labels_copy = risk_labels_copy.dropna(subset=[label_column])

    return risk_labels_copy


def merge_weekly_health_with_phq9(phq9_df=phq9_data, week_anchor='MON', steps_only=False):
    """
    Merge weekly average steps data with PHQ-9 depression labels.

    For each user's weekly steps record, this function finds the PHQ-9 survey response
    from the corresponding week and adds the severity label as the target variable.

    Parameters:
    - weekly_steps_df: DataFrame with columns ['app_user_id', 'week_start', 'avg_daily_steps']
    - phq9_df: DataFrame with PHQ-9 survey data including 'app_user_id', 'timestamp', 'severity_label'
    - week_anchor: weekday anchor for weekly grouping (default 'MON')

    Returns:
    - DataFrame with columns ['app_user_id', 'week_start', 'avg_daily_steps', 'severity_label']
    """
    # Prepare PHQ-9 weekly data
    phq9_weekly = _prepare_phq9_weekly(phq9_df, week_anchor)

    # List all DataFrames to merge
    if steps_only:
        dfs_to_merge = [
            weekly_steps,
            phq9_weekly,
        ]
    else:
        dfs_to_merge = [
        weekly_steps,
        weekly_calorie,
        weekly_speed,
        weekly_distance,
        phq9_weekly
    ]

    # Merge all DataFrames sequentially on app_user_id and week_start
    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on=['app_user_id', 'week_start'], how='inner'),
        dfs_to_merge
    )

    return merged_df

def merge_daily_health_with_phq9(daily_health_df=None, phq9_df=phq9_data, week_anchor='MON', fill_method=None):
    """
    Merge daily health data with PHQ-9 depression labels.

    For each user's daily health record, this function finds the PHQ-9 survey response
    from the corresponding week and adds the severity label.

    Parameters:
    - daily_health_df: DataFrame with daily health data from daily_health_with_week()
                       columns: ['app_user_id', 'date', 'week_start', 'day_index',
                                'daily_steps', 'daily_distance', 'daily_calories', 'daily_avg_speed']
    - phq9_df: DataFrame with PHQ-9 survey data including 'app_user_id', 'timestamp', 'severity_label'
    - week_anchor: weekday anchor for weekly grouping (default 'MON')
    - interpolate: if True, apply linear interpolation to fill null values in health data

    Returns:
    - DataFrame with daily health metrics and associated PHQ-9 labels
    """
    # If no daily health data provided, generate it
    if daily_health_df is None:
        daily_health_df = daily_health_with_week(week_anchor=week_anchor, fill_method=fill_method)

    if daily_health_df.empty:
        return daily_health_df

    # Make a copy to avoid modifying original data
    daily_health_copy = daily_health_df.copy()

    # Ensure week_start is datetime64 type in daily health data
    daily_health_copy['week_start'] = pd.to_datetime(daily_health_copy['week_start'])

    # Prepare PHQ-9 weekly data
    phq9_weekly = _prepare_phq9_weekly(phq9_df, week_anchor)

    # Merge daily health data with PHQ-9 labels based on app_user_id and week_start
    merged_df = pd.merge(
        daily_health_copy,
        phq9_weekly,
        on=['app_user_id', 'week_start'],
        how='inner'
    )

    return merged_df


def merge_hourly_health_with_phq9(hourly_health_df=None, phq9_df=phq9_data, week_anchor='MON', fill_method=None):
    """
    Merge hourly health data with PHQ-9 depression labels.

    For each user's hourly health record, this function finds the PHQ-9 survey response
    from the corresponding week and adds the severity label.

    Parameters:
    - hourly_health_df: DataFrame with hourly health data from hourly_health_data()
                        columns: ['app_user_id', 'datetime', 'date', 'week_start', 'day_index', 'hour_index',
                                 'hourly_steps', 'hourly_distance', 'hourly_calories', 'hourly_avg_speed']
    - phq9_df: DataFrame with PHQ-9 survey data including 'app_user_id', 'timestamp', 'severity_label'
    - week_anchor: weekday anchor for weekly grouping (default 'MON')
    - interpolate: if True, apply linear interpolation to fill null values in health data

    Returns:
    - DataFrame with hourly health metrics and associated PHQ-9 labels
    """
    # If no hourly health data provided, generate it
    if hourly_health_df is None:
        hourly_health_df = hourly_health_data(week_anchor=week_anchor, fill_method=None)

    if hourly_health_df.empty:
        return hourly_health_df

    # Make a copy to avoid modifying original data
    hourly_health_copy = hourly_health_df.copy()

    # Ensure week_start is datetime64 type in hourly health data
    hourly_health_copy['week_start'] = pd.to_datetime(hourly_health_copy['week_start'])

    # Prepare PHQ-9 weekly data
    phq9_weekly = _prepare_phq9_weekly(phq9_df, week_anchor)

    # Merge hourly health data with PHQ-9 labels based on app_user_id and week_start
    merged_df = pd.merge(
        hourly_health_copy,
        phq9_weekly,
        on=['app_user_id', 'week_start'],
        how='inner'
    )

    return merged_df

def _merge_hourly_data_with_risk_labels(hourly_data_df, risk_labels_df, label_column,
                                        datetime_col='datetime', week_anchor='MON', fill_method=None):
    """
    Generic helper function to merge hourly data (health or screentime) with risk labels.

    For each user's hourly data record, this function finds the nearest survey response
    (forward in time) and adds the risk label from that survey.

    Parameters:
    - hourly_data_df: DataFrame with hourly data including 'app_user_id', datetime_col, and metric columns
    - risk_labels_df: DataFrame with risk data including 'app_user_id', 'timestamp', and label_column
    - label_column: name of the risk label column (e.g., 'suicide_risk_label', 'self_harm_risk_label', 'sleep_label')
    - datetime_col: name of the datetime column in hourly_data_df (default 'datetime')
    - week_anchor: weekday anchor for weekly grouping (default 'MON')
    - fill_method: method to fill null values in data ('linear', 'ffill', 'bfill', or None)

    Returns:
    - DataFrame with hourly metrics and associated risk labels
    """
    if hourly_data_df.empty:
        return hourly_data_df

    # Make a copy to avoid modifying original data
    hourly_data_copy = hourly_data_df.copy()

    # Ensure datetime is datetime64 type in hourly data
    hourly_data_copy[datetime_col] = pd.to_datetime(hourly_data_copy[datetime_col])

    # Prepare risk labels
    risk_labels_prepared = _prepare_risk_labels_per_survey(risk_labels_df, label_column)

    if risk_labels_prepared.empty:
        print(f"Warning: No {label_column} data available")
        return pd.DataFrame()

    # For each row in hourly_data_copy, find the nearest survey after that datetime
    result_list = []

    for user_id in hourly_data_copy['app_user_id'].unique():
        # Get user's data
        user_data = hourly_data_copy[hourly_data_copy['app_user_id'] == user_id].copy()

        # Get user's risk surveys
        user_surveys = risk_labels_prepared[risk_labels_prepared['app_user_id'] == user_id].copy()

        if user_surveys.empty:
            continue

        # Sort surveys by timestamp
        user_surveys = user_surveys.sort_values('timestamp')

        # For each data point, find the nearest survey after it
        for _, data_row in user_data.iterrows():
            data_datetime = data_row[datetime_col]

            # Find surveys that occur after this data point
            future_surveys = user_surveys[user_surveys['timestamp'] >= data_datetime]

            if not future_surveys.empty:
                # Take the first (nearest) survey
                nearest_survey = future_surveys.iloc[0]

                # Create a row with data and risk label
                result_row = data_row.to_dict()
                result_row['survey_response_id'] = nearest_survey.get('survey_response_id', None)
                result_row['survey_timestamp'] = nearest_survey['timestamp']
                result_row[label_column] = nearest_survey[label_column]

                result_list.append(result_row)

    if not result_list:
        return pd.DataFrame()

    merged_df = pd.DataFrame(result_list)

    return merged_df


def merge_hourly_health_with_risk_labels(hourly_health_df=None, risk_labels_df=None,
                                         label_column='suicide_risk_label',
                                         week_anchor='MON', fill_method=None):
    """
    Merge hourly health data with risk labels (suicide, self-harm, or sleep).

    For each user's hourly health record, this function finds the nearest survey response
    (forward in time) and adds the risk label from that survey.

    Parameters:
    - hourly_health_df: DataFrame with hourly health data from hourly_health_data()
                        columns: ['app_user_id', 'datetime', 'date', 'week_start', 'day_index', 'hour_index',
                                 'hourly_steps', 'hourly_distance', 'hourly_calories', 'hourly_avg_speed']
    - risk_labels_df: DataFrame with risk data including 'app_user_id', 'timestamp', and label_column
                     If None, uses daily_labels_data (which includes all risk labels)
    - label_column: name of the risk label column to merge (default 'suicide_risk_label')
                   Options: 'suicide_risk_label', 'self_harm_risk_label', 'sleep_label'
    - week_anchor: weekday anchor for weekly grouping (default 'MON')
    - fill_method: method to fill null values in health data ('linear', 'ffill', 'bfill', or None)

    Returns:
    - DataFrame with hourly health metrics and associated risk labels
    """
    # If no hourly health data provided, generate it
    if hourly_health_df is None:
        hourly_health_df = hourly_health_data(week_anchor=week_anchor, fill_method=fill_method)

    # If no risk labels provided, use daily_labels_data
    if risk_labels_df is None:
        risk_labels_df = daily_labels_data

    return _merge_hourly_data_with_risk_labels(
        hourly_health_df, risk_labels_df, label_column,
        datetime_col='datetime', week_anchor=week_anchor, fill_method=fill_method
    )


def merge_hourly_screentime_with_phq9(hourly_screentime_df=None, phq9_df=phq9_data,
                                      week_anchor='MON', fill_method=None,
                                      app_user_id=-1, date_range=None):
    """
    Merge hourly screentime data with PHQ-9 depression labels.

    For each user's hourly screentime record, this function finds the PHQ-9 survey response
    from the corresponding week and adds the severity label.

    Parameters:
    - hourly_screentime_df: DataFrame with hourly screentime data from hourly_screentime_data()
                           columns: ['app_user_id', 'datetime', 'date', 'week_start', 'day_index', 'hour_index',
                                    'hourly_screentime']
    - phq9_df: DataFrame with PHQ-9 survey data including 'app_user_id', 'timestamp', 'severity_label'
    - week_anchor: weekday anchor for weekly grouping (default 'MON')
    - fill_method: method to fill null values in screentime data ('linear', 'ffill', 'bfill', 'zero', or None)
    - app_user_id: filter rows to this app_user_id; if -1, include all users
    - date_range: tuple of (start_date, end_date) to filter data

    Returns:
    - DataFrame with hourly screentime metrics and associated PHQ-9 labels
    """
    # If no hourly screentime data provided, generate it
    if hourly_screentime_df is None:
        from src.data_processing.passive_data_analysis import hourly_screentime_data
        hourly_screentime_df = hourly_screentime_data(week_anchor=week_anchor, fill_method=fill_method,
                                                      app_user_id=app_user_id, date_range=date_range)

    if hourly_screentime_df.empty:
        print("Warning: No hourly screentime data available")
        return pd.DataFrame()

    # Make a copy to avoid modifying original data
    hourly_screentime_copy = hourly_screentime_df.copy()

    # Ensure week_start is datetime64 type in hourly screentime data
    hourly_screentime_copy['week_start'] = pd.to_datetime(hourly_screentime_copy['week_start'])

    # Prepare PHQ-9 weekly data
    phq9_weekly = _prepare_phq9_weekly(phq9_df, week_anchor)

    # Merge hourly screentime data with PHQ-9 labels based on app_user_id and week_start
    merged_df = pd.merge(
        hourly_screentime_copy,
        phq9_weekly,
        on=['app_user_id', 'week_start'],
        how='inner'
    )

    return merged_df


def merge_hourly_screentime_with_risk_labels(hourly_screentime_df=None, risk_labels_df=None,
                                             label_column='suicide_risk_label',
                                             week_anchor='MON', fill_method=None,
                                             app_user_id=-1, date_range=None):
    """
    Merge hourly screentime data with risk labels (suicide, self-harm, or sleep).

    For each user's hourly screentime record, this function finds the nearest survey response
    (forward in time) and adds the risk label from that survey.

    Parameters:
    - hourly_screentime_df: DataFrame with hourly screentime data from hourly_screentime_data()
                           columns: ['app_user_id', 'datetime', 'date', 'week_start', 'day_index', 'hour_index',
                                    'hourly_screentime']
    - risk_labels_df: DataFrame with risk data including 'app_user_id', 'timestamp', and label_column
                     If None, uses daily_labels_data (which includes all risk labels)
    - label_column: name of the risk label column to merge (default 'suicide_risk_label')
                   Options: 'suicide_risk_label', 'self_harm_risk_label', 'sleep_label'
    - week_anchor: weekday anchor for weekly grouping (default 'MON')
    - fill_method: method to fill null values in screentime data ('linear', 'ffill', 'bfill', 'zero', or None)
    - app_user_id: filter rows to this app_user_id; if -1, include all users
    - date_range: tuple of (start_date, end_date) to filter data

    Returns:
    - DataFrame with hourly screentime metrics and associated risk labels
    """
    # If no hourly screentime data provided, generate it
    if hourly_screentime_df is None:
        from src.data_processing.passive_data_analysis import hourly_screentime_data
        hourly_screentime_df = hourly_screentime_data(week_anchor=week_anchor, fill_method=fill_method,
                                                      app_user_id=app_user_id, date_range=date_range)

    # If no risk labels provided, use daily_labels_data
    if risk_labels_df is None:
        risk_labels_df = daily_labels_data

    return _merge_hourly_data_with_risk_labels(
        hourly_screentime_df, risk_labels_df, label_column,
        datetime_col='datetime', week_anchor=week_anchor, fill_method=fill_method
    )


def merge_daily_screentime_features_with_phq9(screentime_df=None, phq9_df=phq9_data,
                                              fill_method='zero', hours_before_survey=24,
                                              week_anchor='MON', app_user_id=-1, date_range=None):
    """
    Merge daily screentime data (with hourly features) with PHQ-9 depression labels.
    For each PHQ-9 survey, this function looks back n hours and creates features from the hourly screentime
    data during that time window before the survey.

    This function is designed for predictive modeling where we want to predict depression based on
    screentime patterns in the n hours before a user submits a PHQ-9 survey.

    Parameters:
    - screentime_df: DataFrame with screentime data; if None, retrieves from database
    - phq9_df: DataFrame with PHQ-9 data including 'app_user_id', 'timestamp', 'severity_label', 'phq9_total_score'
    - fill_method: method to fill null values in hourly screentime features (default 'zero')
    - hours_before_survey: number of hours before a survey to look back for screentime data (default 24)
                          This allows experimenting with different time windows (e.g., 3, 6, 9, 12, 24 hours)
    - week_anchor: weekday anchor for weekly grouping (default 'MON')
    - app_user_id: filter rows to this app_user_id; if -1, include all users
    - date_range: tuple of (start_date, end_date) to filter data

    Returns:
    - DataFrame with columns ['app_user_id', 'phq9_response_id', 'survey_timestamp',
                              'hour_0', 'hour_1', ..., 'hour_N', 'severity_label', 'phq9_total_score']
      where N = hours_before_survey - 1
      Each row represents the screentime in the n hours before a PHQ-9 survey with the survey's depression label.
    """
    from src.data_processing.passive_data_analysis import hourly_screentime_data

    # Get hourly screentime data
    if screentime_df is None:
        hourly_data = hourly_screentime_data(start_col='start_time', week_anchor=week_anchor,
                                             app_user_id=app_user_id, fill_method=fill_method,
                                             date_range=date_range)
    else:
        hourly_data = hourly_screentime_data(screentime_df, 'start_time', week_anchor, app_user_id, fill_method, date_range)

    if hourly_data.empty:
        print("Warning: No screentime data available")
        return pd.DataFrame()

    # Prepare PHQ-9 data
    phq9_copy = phq9_df.copy()
    phq9_copy['timestamp'] = pd.to_datetime(phq9_copy['timestamp'], errors='coerce')
    phq9_copy = phq9_copy.dropna(subset=['timestamp'])

    if phq9_copy.empty:
        print("Warning: No PHQ-9 data available")
        return pd.DataFrame()

    # Ensure datetime is properly typed
    hourly_data['datetime'] = pd.to_datetime(hourly_data['datetime'])

    # For each PHQ-9 survey, look back n hours and create features
    result_list = []

    for user_id in phq9_copy['app_user_id'].unique():
        # Get user's hourly screentime data
        user_screentime = hourly_data[hourly_data['app_user_id'] == user_id].copy()

        # Get user's PHQ-9 surveys
        user_surveys = phq9_copy[phq9_copy['app_user_id'] == user_id].copy()

        if user_screentime.empty:
            continue

        # Sort screentime by datetime
        user_screentime = user_screentime.sort_values('datetime')

        # For each survey, look back n hours
        for _, survey_row in user_surveys.iterrows():
            survey_time = survey_row['timestamp']
            lookback_start = survey_time - pd.Timedelta(hours=hours_before_survey)

            # Get screentime data in the lookback window
            window_data = user_screentime[
                (user_screentime['datetime'] >= lookback_start) &
                (user_screentime['datetime'] < survey_time)
            ].copy()

            if not window_data.empty:
                # Calculate hours before survey for each datapoint
                window_data['hours_before_survey'] = (
                    (survey_time - window_data['datetime']).dt.total_seconds() / 3600
                ).round().astype(int)

                # Create hour features (hour_0 is the hour right before survey, hour_1 is 2 hours before, etc.)
                hour_features = {}
                for i in range(hours_before_survey):
                    # Get screentime for this hour slot
                    hour_data = window_data[window_data['hours_before_survey'] == i]
                    if not hour_data.empty:
                        # Sum screentime if there are multiple entries for this hour
                        hour_features[f'hour_{i}'] = hour_data['hourly_screentime'].sum()
                    else:
                        # No data for this hour
                        hour_features[f'hour_{i}'] = 0.0 if fill_method == 'zero' else np.nan

                # Create result row
                result_row = {
                    'app_user_id': user_id,
                    'survey_timestamp': survey_time,
                    'severity_label': survey_row['severity_label'],
                    'phq9_total_score': survey_row['phq9_total_score']
                }
                result_row.update(hour_features)
                result_list.append(result_row)

    if not result_list:
        print(f"Warning: No screentime data found in the {hours_before_survey} hours before any PHQ-9 surveys")
        return pd.DataFrame()

    merged_df = pd.DataFrame(result_list)

    # Reorder columns
    base_cols = ['app_user_id', 'survey_timestamp']
    hour_cols = [f'hour_{i}' for i in range(hours_before_survey)]
    meta_cols = ['severity_label', 'phq9_total_score']

    # Only include columns that exist
    existing_hour_cols = [col for col in hour_cols if col in merged_df.columns]

    merged_df = merged_df[base_cols + existing_hour_cols + meta_cols]

    return merged_df


def _merge_daily_features_with_risk_labels(hourly_data, risk_labels_df, label_column,
                                           fill_method='zero', hours_before_survey=24):
    """
    Generic helper function to merge daily features (hourly time windows) with risk labels.

    For each risk label survey, this function looks back n hours and creates features from the hourly
    data during that time window before the survey.

    Parameters:
    - hourly_data: DataFrame with hourly data including 'app_user_id', 'datetime', and metric columns
    - risk_labels_df: DataFrame with risk data including 'app_user_id', 'timestamp', and label_column
    - label_column: name of the risk label column (e.g., 'suicide_risk_label', 'self_harm_risk_label', 'sleep_label')
    - fill_method: method to fill null values in hourly features (default 'zero')
    - hours_before_survey: number of hours before a survey to look back for data (default 24)

    Returns:
    - DataFrame with columns ['app_user_id', 'survey_response_id', 'survey_timestamp',
                              'hour_0', 'hour_1', ..., 'hour_N', label_column]
      where N = hours_before_survey - 1
    """
    if hourly_data.empty:
        print("Warning: No hourly data available")
        return pd.DataFrame()

    # Prepare risk labels
    risk_labels_prepared = _prepare_risk_labels_per_survey(risk_labels_df, label_column)

    if risk_labels_prepared.empty:
        print(f"Warning: No {label_column} data available")
        return pd.DataFrame()

    # Ensure datetime is properly typed
    hourly_data['datetime'] = pd.to_datetime(hourly_data['datetime'])

    # For each survey, look back n hours and create features
    result_list = []

    for user_id in risk_labels_prepared['app_user_id'].unique():
        # Get user's hourly data
        user_data = hourly_data[hourly_data['app_user_id'] == user_id].copy()

        # Get user's surveys
        user_surveys = risk_labels_prepared[risk_labels_prepared['app_user_id'] == user_id].copy()

        if user_data.empty:
            continue

        # Sort data by datetime
        user_data = user_data.sort_values('datetime')

        # Determine the metric column (could be 'hourly_screentime' or other metrics)
        metric_col = None
        for col in ['hourly_screentime', 'hourly_steps', 'hourly_distance', 'hourly_calories']:
            if col in user_data.columns:
                metric_col = col
                break

        if metric_col is None:
            print(f"Warning: No metric column found in hourly data for user {user_id}")
            continue

        # For each survey, look back n hours
        for _, survey_row in user_surveys.iterrows():
            survey_time = survey_row['timestamp']
            lookback_start = survey_time - pd.Timedelta(hours=hours_before_survey)

            # Get data in the lookback window
            window_data = user_data[
                (user_data['datetime'] >= lookback_start) &
                (user_data['datetime'] < survey_time)
            ].copy()

            if not window_data.empty:
                # Calculate hours before survey for each datapoint
                window_data['hours_before_survey'] = (
                    (survey_time - window_data['datetime']).dt.total_seconds() / 3600
                ).round().astype(int)

                # Create hour features (hour_0 is the hour right before survey, hour_1 is 2 hours before, etc.)
                hour_features = {}
                for i in range(hours_before_survey):
                    # Get data for this hour slot
                    hour_data = window_data[window_data['hours_before_survey'] == i]
                    if not hour_data.empty:
                        # Sum metric if there are multiple entries for this hour
                        hour_features[f'hour_{i}'] = hour_data[metric_col].sum()
                    else:
                        # No data for this hour
                        hour_features[f'hour_{i}'] = 0.0 if fill_method == 'zero' else np.nan

                # Create result row
                result_row = {
                    'app_user_id': user_id,
                    'survey_response_id': survey_row.get('survey_response_id', None),
                    'survey_timestamp': survey_time,
                    label_column: survey_row[label_column]
                }
                result_row.update(hour_features)
                result_list.append(result_row)

    if not result_list:
        print(f"Warning: No data found in the {hours_before_survey} hours before any surveys")
        return pd.DataFrame()

    merged_df = pd.DataFrame(result_list)

    # Reorder columns
    base_cols = ['app_user_id', 'survey_response_id', 'survey_timestamp']
    hour_cols = [f'hour_{i}' for i in range(hours_before_survey)]
    meta_cols = [label_column]

    # Only include columns that exist
    existing_hour_cols = [col for col in hour_cols if col in merged_df.columns]

    merged_df = merged_df[base_cols + existing_hour_cols + meta_cols]

    return merged_df


def merge_daily_screentime_features_with_risk_labels(screentime_df=None, risk_labels_df=None,
                                                     label_column='suicide_risk_label',
                                                     fill_method='zero', hours_before_survey=24,
                                                     app_user_id=-1, date_range=None):
    """
    Merge daily screentime data (with hourly features) with risk labels from surveys.
    For each survey, this function looks back n hours and creates features from the hourly screentime
    data during that time window before the survey.

    This function is designed for predictive modeling where we want to predict risk based on
    screentime patterns in the n hours before a user submits a survey.

    Parameters:
    - screentime_df: DataFrame with screentime data; if None, retrieves from database
    - risk_labels_df: DataFrame with risk data including 'app_user_id', 'timestamp', and label_column
                     If None, uses daily_labels_data (which includes all risk labels)
    - label_column: name of the risk label column to merge (default 'suicide_risk_label')
                   Options: 'suicide_risk_label', 'self_harm_risk_label', 'sleep_label'
    - fill_method: method to fill null values in hourly screentime features (default 'zero')
    - hours_before_survey: number of hours before a survey to look back for screentime data (default 24)
                          This allows experimenting with different time windows (e.g., 3, 6, 9, 12, 24 hours)
    - app_user_id: filter rows to this app_user_id; if -1, include all users
    - date_range: tuple of (start_date, end_date) to filter data

    Returns:
    - DataFrame with columns ['app_user_id', 'survey_response_id', 'survey_timestamp',
                              'hour_0', 'hour_1', ..., 'hour_N', label_column]
      where N = hours_before_survey - 1
      Each row represents the screentime in the n hours before a survey with the survey's risk label.
    """
    from src.data_processing.passive_data_analysis import hourly_screentime_data

    # Get hourly screentime data
    if screentime_df is None:
        # Use default screentime_data from passive_data_analysis
        hourly_data = hourly_screentime_data(start_col='start_time', week_anchor='MON',
                                             app_user_id=app_user_id, fill_method=fill_method,
                                             date_range=date_range)
    else:
        hourly_data = hourly_screentime_data(screentime_df, 'start_time', 'MON', app_user_id, fill_method, date_range)

    # If no risk labels provided, use daily_labels_data
    if risk_labels_df is None:
        risk_labels_df = daily_labels_data

    return _merge_daily_features_with_risk_labels(
        hourly_data, risk_labels_df, label_column,
        fill_method=fill_method, hours_before_survey=hours_before_survey
    )


# Backward compatibility: Keep the old function name as an alias
def merge_daily_screentime_features_with_suicide_risk(screentime_df=None, suicide_risk_df=suicide_risk_data,
                                                       fill_method='zero', hours_before_survey=24,
                                                       app_user_id=-1, date_range=None):
    """
    Merge daily screentime data (with hourly features) with suicide risk labels from surveys.

    DEPRECATED: Use merge_daily_screentime_features_with_risk_labels() instead with label_column='suicide_risk_label'.
    This function is kept for backward compatibility.
    """
    return merge_daily_screentime_features_with_risk_labels(
        screentime_df=screentime_df,
        risk_labels_df=suicide_risk_df,
        label_column='suicide_risk_label',
        fill_method=fill_method,
        hours_before_survey=hours_before_survey,
        app_user_id=app_user_id,
        date_range=date_range
    )


def export_as_csv(df, output_name='modeling_data_steps_phq9.csv'):
    """
    Export DataFrame to CSV in the project's data directory.

    Parameters:
    - df: DataFrame to export
    - output_name: Name of the output CSV file
    """
    # Export the combined dataset for modeling
    output_path = data_dir / output_name
    df.to_csv(output_path, index=False)
    print(f"Modeling data saved to: {output_path}")



def main():
    # Create the combined dataset for modeling - weekly aggregated data
    print("Creating weekly aggregated health data with PHQ-9 labels...")
    weekly_modeling_data = merge_weekly_health_with_phq9()
    print(weekly_modeling_data.head(10))
    export_as_csv(weekly_modeling_data, 'weekly_health_and_phq9_data.csv')

    # Create daily aggregated health data with PHQ-9 labels
    print("\nCreating daily aggregated health data with PHQ-9 labels...")
    daily_modeling_data = merge_daily_health_with_phq9(fill_method=None)
    print(daily_modeling_data.head(10))
    export_as_csv(daily_modeling_data, 'daily_health_and_phq9_data.csv')

    # Create hourly aggregated health data with PHQ-9 labels
    print("\nCreating hourly aggregated health data with PHQ-9 labels...")
    hourly_modeling_data = merge_hourly_health_with_phq9(fill_method=None)
    print(hourly_modeling_data.head(10))
    export_as_csv(hourly_modeling_data, 'hourly_health_and_phq9_data.csv')

    # Create hourly aggregated health data with suicide risk labels
    print("\nCreating hourly aggregated health data with suicide risk labels...")
    hourly_suicide_risk_data = merge_hourly_health_with_risk_labels(fill_method=None, label_column='suicide_risk_label')
    print(hourly_suicide_risk_data.head(10))
    print(f"\nTotal rows with suicide risk labels: {len(hourly_suicide_risk_data)}")
    if not hourly_suicide_risk_data.empty:
        print(f"Label distribution:\n{hourly_suicide_risk_data['suicide_risk_label'].value_counts()}")
        export_as_csv(hourly_suicide_risk_data, 'hourly_health_and_suicide_risk_data.csv')

    # Create hourly aggregated health data with self-harm risk labels
    print("\nCreating hourly aggregated health data with self-harm risk labels...")
    hourly_selfharm_risk_data = merge_hourly_health_with_risk_labels(fill_method=None, label_column='self_harm_risk_label')
    print(hourly_selfharm_risk_data.head(10))
    print(f"\nTotal rows with self-harm risk labels: {len(hourly_selfharm_risk_data)}")
    if not hourly_selfharm_risk_data.empty:
        print(f"Label distribution:\n{hourly_selfharm_risk_data['self_harm_risk_label'].value_counts()}")
        export_as_csv(hourly_selfharm_risk_data, 'hourly_health_and_selfharm_risk_data.csv')

    # Create hourly aggregated health data with sleep risk labels
    print("\nCreating hourly aggregated health data with sleep risk labels...")
    hourly_sleep_risk_data = merge_hourly_health_with_risk_labels(fill_method=None, label_column='sleep_label')
    # drop afternoon surveys where sleep_label is N/A
    hourly_sleep_risk_data = hourly_sleep_risk_data[hourly_sleep_risk_data['sleep_label'] != 'N/A']
    print(hourly_sleep_risk_data.head(10))
    print(f"\nTotal rows with sleep risk labels: {len(hourly_sleep_risk_data)}")
    if not hourly_sleep_risk_data.empty:
        print(f"Label distribution:\n{hourly_sleep_risk_data['sleep_label'].value_counts()}")
        export_as_csv(hourly_sleep_risk_data, 'hourly_health_and_sleep_risk_data.csv')

    # Create hourly aggregated screentime data with PHQ-9 labels
    print("\nCreating hourly aggregated screentime data with PHQ-9 labels...")
    hourly_screentime_phq9_data = merge_hourly_screentime_with_phq9(fill_method=None)
    print(hourly_screentime_phq9_data.head(10))
    print(f"\nTotal rows with PHQ-9 labels: {len(hourly_screentime_phq9_data)}")
    if not hourly_screentime_phq9_data.empty:
        print(f"Label distribution:\n{hourly_screentime_phq9_data['severity_label'].value_counts()}")
        export_as_csv(hourly_screentime_phq9_data, 'hourly_screentime_and_phq9_data.csv')

    # Create hourly aggregated screentime data with suicide risk labels
    print("\nCreating hourly aggregated screentime data with suicide risk labels...")
    hourly_screentime_suicide_risk_data = merge_hourly_screentime_with_risk_labels(fill_method=None, label_column='suicide_risk_label')
    print(hourly_screentime_suicide_risk_data.head(10))
    print(f"\nTotal rows with suicide risk labels: {len(hourly_screentime_suicide_risk_data)}")
    if not hourly_screentime_suicide_risk_data.empty:
        print(f"Label distribution:\n{hourly_screentime_suicide_risk_data['suicide_risk_label'].value_counts()}")
        export_as_csv(hourly_screentime_suicide_risk_data, 'hourly_screentime_and_suicide_risk_data.csv')

    # Create hourly aggregated screentime data with self-harm risk labels
    print("\nCreating hourly aggregated screentime data with self-harm risk labels...")
    hourly_screentime_selfharm_risk_data = merge_hourly_screentime_with_risk_labels(fill_method=None, label_column='self_harm_risk_label')
    print(hourly_screentime_selfharm_risk_data.head(10))
    print(f"\nTotal rows with self-harm risk labels: {len(hourly_screentime_selfharm_risk_data)}")
    if not hourly_screentime_selfharm_risk_data.empty:
        print(f"Label distribution:\n{hourly_screentime_selfharm_risk_data['self_harm_risk_label'].value_counts()}")
        export_as_csv(hourly_screentime_selfharm_risk_data, 'hourly_screentime_and_selfharm_risk_data.csv')

    # Create hourly aggregated screentime data with sleep risk labels
    print("\nCreating hourly aggregated screentime data with sleep risk labels...")
    hourly_screentime_sleep_risk_data = merge_hourly_screentime_with_risk_labels(fill_method=None, label_column='sleep_label')
    # drop afternoon surveys where sleep_label is N/A
    hourly_screentime_sleep_risk_data = hourly_screentime_sleep_risk_data[hourly_screentime_sleep_risk_data['sleep_label'] != 'N/A']
    print(hourly_screentime_sleep_risk_data.head(10))
    print(f"\nTotal rows with sleep risk labels: {len(hourly_screentime_sleep_risk_data)}")
    if not hourly_screentime_sleep_risk_data.empty:
        print(f"Label distribution:\n{hourly_screentime_sleep_risk_data['sleep_label'].value_counts()}")
        export_as_csv(hourly_screentime_sleep_risk_data, 'hourly_screentime_and_sleep_risk_data.csv')

    # Create daily screentime features (time windows) with PHQ-9 labels
    print("\n" + "="*80)
    print("Creating daily screentime features with PHQ-9 labels (time window approach)...")
    print("="*80)

    time_windows = [3, 6, 9, 12, 24]
    for hours in time_windows:
        print(f"\nCreating screentime features for {hours}-hour window before PHQ-9 surveys...")
        daily_screentime_phq9_data = merge_daily_screentime_features_with_phq9(
            fill_method='zero',
            hours_before_survey=hours
        )
        print(f"Total rows: {len(daily_screentime_phq9_data)}")
        if not daily_screentime_phq9_data.empty:
            print(f"Label distribution:\n{daily_screentime_phq9_data['severity_label'].value_counts()}")
            export_as_csv(daily_screentime_phq9_data, f'daily_screentime_phq9_{hours}h.csv')

    print("\n" + "="*80)
    print("Data merge complete!")
    print("For time window modeling with screentime data, run:")
    print("  python src/modeling/model_screentime_time_windows.py")
    print("="*80)

if __name__ == '__main__':
    main()


