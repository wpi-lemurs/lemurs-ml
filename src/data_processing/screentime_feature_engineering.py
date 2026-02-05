"""
Screentime Feature Engineering

This script reorganizes app_genre data to create useful features for machine learning:
- Most used app category for different time windows (last 1h, 3h, 6h, 12h, 24h)
- Total time used in each category for those windows
- Category diversity metrics
- Temporal usage patterns

Example output features:
- most_used_category_3h: "SOCIAL_MEDIA"
- total_time_most_used_3h: 120 (minutes)
- category_diversity_3h: 0.75 (entropy-based)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.database_service import DatabaseService
from scipy.stats import entropy
from typing import Dict, List, Tuple

# Time windows to analyze (in hours)
TIME_WINDOWS = [1, 3, 6, 12, 24, 48, 72, 168]  # 1h, 3h, 6h, 12h, 1d, 2d, 3d, 1week


def load_and_clean_screentime_data() -> pd.DataFrame:
    """
    Load screentime data and clean it to remove duplicate/cumulative records.

    Returns:
        Cleaned DataFrame with screentime app usage data
    """
    print("="*80)
    print("LOADING AND CLEANING SCREENTIME DATA")
    print("="*80)

    # Load categorized screentime data
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'screentime_app_categorized.csv')
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} records from screentime_app_categorized.csv")

    # Connect to database to get user mapping and timestamps
    service = DatabaseService()

    # Get app_user_id and timestamp from screentime table
    screentime_data = service.extract_from_database("screentime")
    screentime_mapping = screentime_data[['id', 'app_user_id', 'start_time']].rename(
        columns={'id': 'screentime_id'}
    )
    df = df.merge(screentime_mapping, on='screentime_id', how='left')

    service.disconnect()

    print(f"\nRecords before deduplication: {len(df):,}")

    # DEDUPLICATION: Keep only the most recent record for each app per screentime_id
    # Since total_time_ms is cumulative, we want the latest record which has the final count
    df['last_time_used'] = pd.to_datetime(df['last_time_used'])
    df['start_time'] = pd.to_datetime(df['start_time'])
    df = df.sort_values('last_time_used', ascending=True)
    df = df.drop_duplicates(subset=['screentime_id', 'app_name'], keep='last')

    print(f"Records after deduplication: {len(df):,}")
    print(f"Records removed: {len(pd.read_csv(data_path)) - len(df):,}\n")

    # Convert time to minutes for easier interpretation
    df['total_time_minutes'] = df['total_time_ms'] / (1000 * 60)

    return df


def calculate_category_usage_in_window(df: pd.DataFrame,
                                      reference_time: pd.Timestamp,
                                      window_hours: int,
                                      user_id: int) -> Dict:
    """
    Calculate category usage statistics for a specific time window.

    Args:
        df: Screentime DataFrame
        reference_time: The reference timestamp to look back from
        window_hours: Number of hours to look back
        user_id: User ID (app_user_id)

    Returns:
        Dictionary with features for this time window
    """
    # Filter data for this user and time window
    window_start = reference_time - timedelta(hours=window_hours)
    mask = (
        (df['app_user_id'] == user_id) &
        (df['last_time_used'] <= reference_time) &
        (df['last_time_used'] >= window_start)
    )
    window_data = df[mask].copy()

    features = {}
    suffix = f"_{window_hours}h"

    if len(window_data) == 0:
        # No data in this window
        features[f'most_used_category{suffix}'] = None
        features[f'most_used_category_time{suffix}'] = 0
        features[f'total_screentime{suffix}'] = 0
        features[f'num_apps_used{suffix}'] = 0
        features[f'num_categories_used{suffix}'] = 0
        features[f'category_diversity{suffix}'] = 0
        features[f'data_available{suffix}'] = 0
        return features

    # Aggregate by category
    category_usage = window_data.groupby('app_category')['total_time_minutes'].sum().sort_values(ascending=False)

    # Most used category and its time
    most_used_category = category_usage.index[0] if len(category_usage) > 0 else None
    most_used_time = category_usage.iloc[0] if len(category_usage) > 0 else 0

    # Total screentime
    total_time = window_data['total_time_minutes'].sum()

    # Number of unique apps and categories
    num_apps = window_data['app_name'].nunique()
    num_categories = window_data['app_category'].nunique()

    # Category diversity (using normalized entropy)
    if len(category_usage) > 1:
        proportions = category_usage / category_usage.sum()
        category_entropy = entropy(proportions)
        # Normalize by max possible entropy for this number of categories
        max_entropy = np.log(len(category_usage))
        normalized_entropy = category_entropy / max_entropy if max_entropy > 0 else 0
    else:
        normalized_entropy = 0

    # Store features
    features[f'most_used_category{suffix}'] = most_used_category
    features[f'most_used_category_time{suffix}'] = most_used_time
    features[f'total_screentime{suffix}'] = total_time
    features[f'num_apps_used{suffix}'] = num_apps
    features[f'num_categories_used{suffix}'] = num_categories
    features[f'category_diversity{suffix}'] = normalized_entropy
    features[f'data_available{suffix}'] = 1

    # Top 3 categories and their times
    for i in range(min(3, len(category_usage))):
        features[f'category_rank_{i+1}{suffix}'] = category_usage.index[i]
        features[f'category_rank_{i+1}_time{suffix}'] = category_usage.iloc[i]

    # Proportion of time in most used category
    features[f'most_used_category_proportion{suffix}'] = most_used_time / total_time if total_time > 0 else 0

    return features


def generate_features_for_all_timepoints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate features for all users at all submission timepoints.

    This creates a feature matrix where each row represents a screentime submission,
    and columns are features about app category usage in various time windows
    leading up to that submission.

    Returns:
        DataFrame with engineered features
    """
    print("="*80)
    print("GENERATING FEATURES FOR ALL TIMEPOINTS")
    print("="*80)

    all_features = []

    # Get all unique submission events (one per screentime_id)
    submission_events = df[['screentime_id', 'app_user_id', 'start_time']].drop_duplicates()
    submission_events = submission_events.sort_values(['app_user_id', 'start_time'])

    print(f"Processing {len(submission_events):,} submission events for {submission_events['app_user_id'].nunique()} users...")

    for idx, row in submission_events.iterrows():
        if idx % 100 == 0:
            print(f"  Processed {idx}/{len(submission_events)} submissions...")

        screentime_id = row['screentime_id']
        user_id = row['app_user_id']
        reference_time = row['start_time']

        # Base features
        features = {
            'screentime_id': screentime_id,
            'app_user_id': user_id,
            'reference_time': reference_time
        }

        # Generate features for each time window
        for window_hours in TIME_WINDOWS:
            window_features = calculate_category_usage_in_window(
                df, reference_time, window_hours, user_id
            )
            features.update(window_features)

        all_features.append(features)

    print(f"  Completed processing all {len(submission_events)} submissions!")

    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)

    print(f"\nGenerated {len(features_df):,} feature rows with {len(features_df.columns)} columns")
    print(f"Feature columns: {list(features_df.columns[:10])}... (showing first 10)")

    return features_df


def add_temporal_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features like hour of day, day of week, etc.

    Args:
        features_df: DataFrame with basic features

    Returns:
        DataFrame with added temporal features
    """
    features_df['hour_of_day'] = features_df['reference_time'].dt.hour
    features_df['day_of_week'] = features_df['reference_time'].dt.dayofweek
    features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
    features_df['is_weekday'] = (~features_df['day_of_week'].isin([5, 6])).astype(int)

    # Time of day categories
    def categorize_time_of_day(hour):
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night'

    features_df['time_of_day'] = features_df['hour_of_day'].apply(categorize_time_of_day)

    return features_df


def create_feature_summary(features_df: pd.DataFrame) -> None:
    """
    Print summary statistics about the generated features.
    """
    print("\n" + "="*80)
    print("FEATURE SUMMARY")
    print("="*80)

    print(f"\nTotal feature rows: {len(features_df):,}")
    print(f"Total users: {features_df['app_user_id'].nunique()}")
    print(f"Date range: {features_df['reference_time'].min()} to {features_df['reference_time'].max()}")

    print("\n--- Most Used Categories (3-hour window) ---")
    if features_df['most_used_category_3h'].notna().any():
        category_counts = features_df['most_used_category_3h'].value_counts().head(10)
        print(category_counts)

    print("\n--- Average Screentime by Time Window ---")
    for window in TIME_WINDOWS:
        col = f'total_screentime_{window}h'
        if col in features_df.columns:
            avg_time = features_df[col].mean()
            print(f"  {window}h window: {avg_time:.2f} minutes")

    print("\n--- Data Availability by Time Window ---")
    for window in TIME_WINDOWS:
        col = f'data_available_{window}h'
        if col in features_df.columns:
            availability = features_df[col].mean() * 100
            print(f"  {window}h window: {availability:.1f}% of timepoints have data")


def main():
    """
    Main function to orchestrate feature engineering.
    """
    print("\n" + "="*80)
    print("SCREENTIME FEATURE ENGINEERING")
    print("="*80)
    print("\nThis script generates features for machine learning from screentime data.")
    print("Features include most-used categories and total time for various time windows.\n")

    # Load and clean data
    df = load_and_clean_screentime_data()

    # Generate features
    features_df = generate_features_for_all_timepoints(df)

    # Add temporal features
    features_df = add_temporal_features(features_df)

    # Create summary
    create_feature_summary(features_df)

    # Save features
    output_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'screentime_features_engineered.csv'
    )
    features_df.to_csv(output_path, index=False)
    print(f"\n{'='*80}")
    print(f"Features saved to: {output_path}")
    print(f"{'='*80}\n")

    # Show example of features for one user
    print("\n--- EXAMPLE: Features for first submission ---")
    example = features_df.iloc[0]
    print(f"\nUser: {example['app_user_id']}")
    print(f"Reference time: {example['reference_time']}")
    print(f"\nSample features:")
    print(f"  Most used category (3h): {example['most_used_category_3h']}")
    print(f"  Time in that category (3h): {example['most_used_category_time_3h']:.2f} minutes")
    print(f"  Total screentime (3h): {example['total_screentime_3h']:.2f} minutes")
    print(f"  Number of apps used (3h): {example['num_apps_used_3h']}")
    print(f"  Category diversity (3h): {example['category_diversity_3h']:.3f}")
    print(f"  Time of day: {example['time_of_day']}")

    return features_df


if __name__ == "__main__":
    features_df = main()
