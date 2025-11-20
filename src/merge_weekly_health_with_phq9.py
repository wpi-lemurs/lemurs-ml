from src.health_data_analysis import *
from src.categorization.PHQ9_categorization_binary import *
import os
from functools import reduce

current_dir = os.path.dirname(os.path.abspath(__file__))

# Load data with synthetic data paths (just for testing)
# steps_data = pd.read_csv(os.path.join(current_dir, 'data', 'synthetic', 'synthetic_step_data.csv'))
# phq9_data = pd.read_csv(os.path.join(current_dir, 'data', 'synthetic', 'synthetic_phq9_data.csv'))

# We eventually will use this instead
weekly_steps = weekly_avg_steps()
phq9_data = get_phq9_dataframe()
weekly_speed = weekly_avg_speed()
weekly_calorie = weekly_avg_calorie()
weekly_distance = weekly_avg_distance()

# Generate weekly average steps for each user
weekly_steps = weekly_avg_health_data(steps_data, target_col='steps', new_col_name='avg_daily_steps')

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
    # Make a copy to avoid modifying original data
    phq9_copy = phq9_df.copy()

    # Parse PHQ-9 response timestamps
    phq9_copy['timestamp'] = pd.to_datetime(phq9_copy['timestamp'], errors='coerce')
    phq9_copy = phq9_copy.dropna(subset=['timestamp'])

    # Calculate the week start for each PHQ-9 response
    # Use resample to match the weekly_avg_health_data logic exactly
    freq = f'W-{week_anchor}'

    # Group by user and calculate week_start for each response
    phq9_weekly_list = []
    for user_id, user_df in phq9_copy.groupby('app_user_id'):
        user_df = user_df.set_index('timestamp').sort_index()

        # For each response, find which week it belongs to
        for timestamp, row in user_df.iterrows():
            # Calculate week_start by flooring to the beginning of the week
            week_start = pd.Timestamp(timestamp).to_period(freq).start_time
            # Adjust: to_period().start_time gives us the week boundary,
            # but we need to subtract to get the Monday of that week
            # Actually, let's use a simpler approach: normalize and find the Monday
            ts = pd.Timestamp(timestamp)
            days_since_monday = ts.weekday()  # Monday = 0
            week_start = (ts - pd.Timedelta(days=days_since_monday)).normalize()

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

def export_as_csv(df, output_name='modeling_data_steps_phq9.csv'):
    # Export the combined dataset for modeling
    output_path = os.path.join(current_dir, 'data', output_name)
    df.to_csv(output_path, index=False)
    print(f"Modeling data saved to: {output_path}")


def main():
    # Create the combined dataset for modeling
    modeling_data = merge_weekly_health_with_phq9(steps_only=False)
    print(modeling_data.head(10))
    export_as_csv(modeling_data, 'health_and_phq9_data.csv')

if __name__ == '__main__':
    main()



