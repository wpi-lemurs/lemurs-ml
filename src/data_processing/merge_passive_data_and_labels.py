from src.data_processing.passive_data_analysis import *
from src.categorization.PHQ9_categorization_binary import *
from src.categorization.suicide_risk_labels import *
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

def _prepare_suicide_risk_labels_per_survey(suicide_risk_df=suicide_risk_data):
    """
    Helper function to prepare suicide risk labels for merging with health data.

    Unlike PHQ-9 which we aggregate weekly, suicide risk labels are kept at the survey level
    and matched to the nearest survey for each health data point.

    Parameters:
    - suicide_risk_df: DataFrame with suicide risk data including 'app_user_id', 'timestamp', 'suicide_risk_label'

    Returns:
    - DataFrame with columns ['survey_response_id', 'app_user_id', 'timestamp', 'suicide_risk_label']
    """
    # Make a copy to avoid modifying original data
    suicide_risk_copy = suicide_risk_df.copy()

    # Parse timestamps
    suicide_risk_copy['timestamp'] = pd.to_datetime(suicide_risk_copy['timestamp'], errors='coerce')
    suicide_risk_copy = suicide_risk_copy.dropna(subset=['timestamp'])

    # Keep only necessary columns
    suicide_risk_copy = suicide_risk_copy[['survey_response_id', 'app_user_id', 'timestamp', 'suicide_risk_label']]

    return suicide_risk_copy


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

def merge_hourly_health_with_suicide_risk_labels(hourly_health_df=None, suicide_risk_df=suicide_risk_data, week_anchor='MON', fill_method=None):
    """
    Merge hourly health data with suicide risk labels.

    For each user's hourly health record, this function finds the nearest survey response
    (forward in time) and adds the suicide risk label from that survey.

    Parameters:
    - hourly_health_df: DataFrame with hourly health data from hourly_health_data()
                        columns: ['app_user_id', 'datetime', 'date', 'week_start', 'day_index', 'hour_index',
                                 'hourly_steps', 'hourly_distance', 'hourly_calories', 'hourly_avg_speed']
    - suicide_risk_df: DataFrame with suicide risk data including 'app_user_id', 'timestamp', 'suicide_risk_label'
    - week_anchor: weekday anchor for weekly grouping (default 'MON')
    - fill_method: method to fill null values in health data ('linear', 'ffill', 'bfill', or None)

    Returns:
    - DataFrame with hourly health metrics and associated suicide risk labels
    """
    # If no hourly health data provided, generate it
    if hourly_health_df is None:
        hourly_health_df = hourly_health_data(week_anchor=week_anchor, fill_method=fill_method)

    if hourly_health_df.empty:
        return hourly_health_df

    # Make a copy to avoid modifying original data
    hourly_health_copy = hourly_health_df.copy()

    # Ensure datetime is datetime64 type in hourly health data
    hourly_health_copy['datetime'] = pd.to_datetime(hourly_health_copy['datetime'])

    # Prepare suicide risk labels
    suicide_risk_prepared = _prepare_suicide_risk_labels_per_survey(suicide_risk_df)

    if suicide_risk_prepared.empty:
        print("Warning: No suicide risk data available")
        return pd.DataFrame()

    # For each row in hourly_health_copy, find the nearest survey after that datetime
    result_list = []

    for user_id in hourly_health_copy['app_user_id'].unique():
        # Get user's health data
        user_health = hourly_health_copy[hourly_health_copy['app_user_id'] == user_id].copy()

        # Get user's suicide risk surveys
        user_surveys = suicide_risk_prepared[suicide_risk_prepared['app_user_id'] == user_id].copy()

        if user_surveys.empty:
            continue

        # Sort surveys by timestamp
        user_surveys = user_surveys.sort_values('timestamp')

        # For each health data point, find the nearest survey after it
        for _, health_row in user_health.iterrows():
            health_datetime = health_row['datetime']

            # Find surveys that occur after this health data point
            future_surveys = user_surveys[user_surveys['timestamp'] >= health_datetime]

            if not future_surveys.empty:
                # Take the first (nearest) survey
                nearest_survey = future_surveys.iloc[0]

                # Create a row with health data and suicide risk label
                result_row = health_row.to_dict()
                result_row['survey_response_id'] = nearest_survey['survey_response_id']
                result_row['survey_timestamp'] = nearest_survey['timestamp']
                result_row['suicide_risk_label'] = nearest_survey['suicide_risk_label']

                result_list.append(result_row)

    if not result_list:
        return pd.DataFrame()

    merged_df = pd.DataFrame(result_list)

    return merged_df


def merge_hourly_screentime_with_suicide_risk(hourly_screentime_df=None, suicide_risk_df=suicide_risk_data,
                                                week_anchor='MON', fill_method=None,
                                                app_user_id=-1, date_range=None):
    """
    Merge hourly screentime data with suicide risk labels.

    For each user's hourly screentime record, this function finds the nearest survey response
    (forward in time) and adds the suicide risk label from that survey.

    Parameters:
    - hourly_screentime_df: DataFrame with hourly screentime data from hourly_screentime_data()
                           columns: ['app_user_id', 'datetime', 'date', 'week_start', 'day_index', 'hour_index',
                                    'hourly_screentime']
    - suicide_risk_df: DataFrame with suicide risk data including 'app_user_id', 'timestamp', 'suicide_risk_label'
    - week_anchor: weekday anchor for weekly grouping (default 'MON')
    - fill_method: method to fill null values in screentime data ('linear', 'ffill', 'bfill', 'zero', or None)
    - app_user_id: filter rows to this app_user_id; if -1, include all users
    - date_range: tuple of (start_date, end_date) to filter data

    Returns:
    - DataFrame with hourly screentime metrics and associated suicide risk labels
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

    # Ensure datetime is datetime64 type in hourly screentime data
    hourly_screentime_copy['datetime'] = pd.to_datetime(hourly_screentime_copy['datetime'])

    # Prepare suicide risk labels
    suicide_risk_prepared = _prepare_suicide_risk_labels_per_survey(suicide_risk_df)

    if suicide_risk_prepared.empty:
        print("Warning: No suicide risk data available")
        return pd.DataFrame()

    # For each row in hourly_screentime_copy, find the nearest survey after that datetime
    result_list = []

    for user_id in hourly_screentime_copy['app_user_id'].unique():
        # Get user's screentime data
        user_screentime = hourly_screentime_copy[hourly_screentime_copy['app_user_id'] == user_id].copy()

        # Get user's suicide risk surveys
        user_surveys = suicide_risk_prepared[suicide_risk_prepared['app_user_id'] == user_id].copy()

        if user_surveys.empty:
            continue

        # Sort surveys by timestamp
        user_surveys = user_surveys.sort_values('timestamp')

        # For each screentime data point, find the nearest survey after it
        for _, screentime_row in user_screentime.iterrows():
            screentime_datetime = screentime_row['datetime']

            # Find surveys that occur after this screentime data point
            future_surveys = user_surveys[user_surveys['timestamp'] >= screentime_datetime]

            if not future_surveys.empty:
                # Take the first (nearest) survey
                nearest_survey = future_surveys.iloc[0]

                # Create a row with screentime data and suicide risk label
                result_row = screentime_row.to_dict()
                result_row['survey_response_id'] = nearest_survey['survey_response_id']
                result_row['survey_timestamp'] = nearest_survey['timestamp']
                result_row['suicide_risk_label'] = nearest_survey['suicide_risk_label']

                result_list.append(result_row)

    if not result_list:
        print("Warning: No screentime data with corresponding suicide risk surveys found")
        return pd.DataFrame()

    merged_df = pd.DataFrame(result_list)

    return merged_df


def merge_daily_screentime_features_with_suicide_risk(screentime_df=None, suicide_risk_df=suicide_risk_data,
                                                       fill_method='zero', hours_before_survey=24,
                                                       app_user_id=-1, date_range=None):
    """
    Merge daily screentime data (with hourly features) with suicide risk labels from surveys.
    For each survey, this function looks back n hours and creates features from the hourly screentime
    data during that time window before the survey.

    This function is designed for predictive modeling where we want to predict suicide risk based on
    screentime patterns in the n hours before a user submits a survey.

    Parameters:
    - screentime_df: DataFrame with screentime data; if None, retrieves from database
    - suicide_risk_df: DataFrame with suicide risk data including 'app_user_id', 'timestamp', 'suicide_risk_label'
    - fill_method: method to fill null values in hourly screentime features (default 'zero')
    - hours_before_survey: number of hours before a survey to look back for screentime data (default 24)
                          This allows experimenting with different time windows (e.g., 3, 6, 9, 12, 24 hours)
    - app_user_id: filter rows to this app_user_id; if -1, include all users
    - date_range: tuple of (start_date, end_date) to filter data

    Returns:
    - DataFrame with columns ['app_user_id', 'survey_response_id', 'survey_timestamp',
                              'hour_0', 'hour_1', ..., 'hour_N', 'suicide_risk_label']
      where N = hours_before_survey - 1
      Each row represents the screentime in the n hours before a survey with the survey's suicide risk label.
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

    if hourly_data.empty:
        print("Warning: No screentime data available")
        return pd.DataFrame()

    # Prepare suicide risk labels
    suicide_risk_prepared = _prepare_suicide_risk_labels_per_survey(suicide_risk_df)

    if suicide_risk_prepared.empty:
        print("Warning: No suicide risk data available")
        return pd.DataFrame()

    # Ensure datetime is properly typed
    hourly_data['datetime'] = pd.to_datetime(hourly_data['datetime'])

    # For each survey, look back n hours and create features
    result_list = []

    for user_id in suicide_risk_prepared['app_user_id'].unique():
        # Get user's hourly screentime data
        user_screentime = hourly_data[hourly_data['app_user_id'] == user_id].copy()

        # Get user's suicide risk surveys
        user_surveys = suicide_risk_prepared[suicide_risk_prepared['app_user_id'] == user_id].copy()

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
                    'survey_response_id': survey_row['survey_response_id'],
                    'survey_timestamp': survey_time,
                    'suicide_risk_label': survey_row['suicide_risk_label']
                }
                result_row.update(hour_features)
                result_list.append(result_row)

    if not result_list:
        print(f"Warning: No screentime data found in the {hours_before_survey} hours before any surveys")
        return pd.DataFrame()

    merged_df = pd.DataFrame(result_list)

    # Reorder columns
    base_cols = ['app_user_id', 'survey_response_id', 'survey_timestamp']
    hour_cols = [f'hour_{i}' for i in range(hours_before_survey)]
    meta_cols = ['suicide_risk_label']

    # Only include columns that exist
    existing_hour_cols = [col for col in hour_cols if col in merged_df.columns]

    merged_df = merged_df[base_cols + existing_hour_cols + meta_cols]

    return merged_df


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
    hourly_suicide_risk_data = merge_hourly_health_with_suicide_risk_labels(fill_method=None)
    print(hourly_suicide_risk_data.head(10))
    print(f"\nTotal rows with suicide risk labels: {len(hourly_suicide_risk_data)}")
    if not hourly_suicide_risk_data.empty:
        print(f"Label distribution:\n{hourly_suicide_risk_data['suicide_risk_label'].value_counts()}")
        export_as_csv(hourly_suicide_risk_data, 'hourly_health_and_suicide_risk_data.csv')

    # Create hourly aggregated screentime data with suicide risk labels
    print("\nCreating hourly aggregated screentime data with suicide risk labels...")
    hourly_screentime_suicide_risk_data = merge_hourly_screentime_with_suicide_risk(fill_method=None)
    print(hourly_screentime_suicide_risk_data.head(10))
    print(f"\nTotal rows with suicide risk labels: {len(hourly_screentime_suicide_risk_data)}")
    if not hourly_screentime_suicide_risk_data.empty:
        print(f"Label distribution:\n{hourly_screentime_suicide_risk_data['suicide_risk_label'].value_counts()}")
        export_as_csv(hourly_screentime_suicide_risk_data, 'hourly_screentime_and_suicide_risk_data.csv')

    print("\n" + "="*80)
    print("Data merge complete!")
    print("For time window modeling with screentime data, run:")
    print("  python src/modeling/model_screentime_time_windows.py")
    print("="*80)

if __name__ == '__main__':
    main()


