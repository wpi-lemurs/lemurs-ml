"""
Custom scikit-learn transformers for mental health prediction pipeline.

This module provides custom transformers that wrap the existing data processing,
categorization, and feature engineering logic into scikit-learn compatible components.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.database_service import DatabaseService
from src.data_processing.passive_data_analysis import (
    _process_passive_data_dataframe,
    hourly_screentime_data as get_hourly_screentime
)
from src.categorization.PHQ9_categorization_binary import get_phq9_dataframe
from src.categorization.daily_questions_categorization import get_daily_labels_dataframe
from src.data_processing.merge_passive_data_and_labels import (
    merge_daily_screentime_features_with_phq9,
    merge_daily_screentime_features_with_risk_labels,
    propagate_positive_labels
)


class ScreentimeAppCategorizer(BaseEstimator, TransformerMixin):
    """
    Extracts and categorizes screentime app data from the database.

    This transformer performs the expensive categorization step ONCE during fit()
    and stores the result, which can then be reused by downstream transformers.

    This is the recommended first step in any pipeline that uses app categories,
    as it separates the slow categorization from the fast feature engineering.

    Parameters:
    -----------
    None

    Attributes:
    -----------
    categorized_data_ : pd.DataFrame
        The categorized screentime app data, set during fit()
    """

    def __init__(self):
        self.categorized_data_ = None

    def fit(self, X=None, y=None):
        """
        Extract and categorize screentime app data.

        This is where the expensive categorization happens - only during fit(),
        not during transform().
        """
        from src.data_processing.screentime_feature_engineering import load_and_clean_screentime_data

        print("\n" + "="*80)
        print("CATEGORIZING SCREENTIME APPS")
        print("="*80)

        # This does the categorization
        self.categorized_data_ = load_and_clean_screentime_data()

        print(f"Categorized {len(self.categorized_data_):,} records")
        print(f"{self.categorized_data_['app_name'].nunique():,} unique apps")
        print("="*80 + "\n")

        return self

    def transform(self, X=None):
        """
        Return the categorized data that was created during fit().

        Returns:
        --------
        pd.DataFrame : The categorized screentime app data
        """
        if self.categorized_data_ is None:
            raise RuntimeError("Must call fit() before transform()")

        return self.categorized_data_


class DataExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts raw data from the database or CSV files.

    Parameters:
    -----------
    source : str, default='database'
        Data source: 'database' to extract from PostgreSQL, 'csv' to load from files
    data_types : list of str, default=['screentime', 'steps', 'speed', 'distance', 'calorie']
        Types of passive data to extract
    """

    def __init__(self, source='database', data_types=None):
        self.source = source
        self.data_types = data_types or ['screentime', 'steps', 'speed', 'distance', 'calorie']
        self.db_service = None
        self.raw_data_ = {}

    def fit(self, X=None, y=None):
        """Initialize database connection if needed."""
        if self.source == 'database':
            self.db_service = DatabaseService()
        return self

    def transform(self, X=None):
        """
        Extract data from specified source.

        Returns:
        --------
        dict : Dictionary with data type as keys and DataFrames as values
        """
        for data_type in self.data_types:
            if self.source == 'database':
                self.raw_data_[data_type] = self.db_service.extract_from_database(data_type)
            else:
                # Could implement CSV loading here if needed
                raise NotImplementedError("CSV loading not yet implemented")

        return self.raw_data_


class ScreentimeProcessor(BaseEstimator, TransformerMixin):
    """
    Processes screentime data into hourly features.

    Parameters:
    -----------
    fill_method : str or None, default=None
        Method for handling missing values: 'zero', 'ffill', 'bfill', 'interpolate', or None
    app_user_id : int, default=-1
        Filter to specific user ID, or -1 for all users
    date_range : tuple or None, default=None
        Optional (start_date, end_date) tuple for filtering
    use_accurate_method : bool, default=True
        DEPRECATED and ignored. App-table screentime is always used.
    """

    def __init__(self, fill_method=None, app_user_id=-1, date_range=None, use_accurate_method=True):
        self.fill_method = fill_method
        self.app_user_id = app_user_id
        self.date_range = date_range
        self.use_accurate_method = use_accurate_method

    def fit(self, X, y=None):
        """Fit does nothing, returns self."""
        return self

    def transform(self, X):
        """
        Transform raw screentime data into hourly aggregated features.

        Parameters:
        -----------
        X : dict or DataFrame
            If dict, expects key 'screentime' with DataFrame value
            If DataFrame, uses directly as screentime data

        Returns:
        --------
        DataFrame : Hourly aggregated screentime data
        """
        if isinstance(X, dict):
            screentime_df = X.get('screentime')
        else:
            screentime_df = X

        if screentime_df is None or screentime_df.empty:
            return pd.DataFrame()

        # Calculate duration if needed
        if 'value' not in screentime_df.columns:
            screentime_df = screentime_df.copy()
            screentime_df['value'] = (
                pd.to_datetime(screentime_df['end_time']) -
                pd.to_datetime(screentime_df['start_time'])
            ).dt.total_seconds()

        # Use existing hourly processing function
        hourly_data = get_hourly_screentime(
            fill_method=self.fill_method,
            app_user_id=self.app_user_id,
            date_range=self.date_range,
            use_accurate_method=True
        )

        return hourly_data


class HealthDataProcessor(BaseEstimator, TransformerMixin):
    """
    Processes health data (steps, speed, distance, calories) into aggregated features.

    Parameters:
    -----------
    aggregation : str, default='hourly'
        Time aggregation: 'hourly', 'daily', or 'weekly'
    metrics : list of str, default=['steps', 'speed', 'distance', 'calorie']
        Health metrics to process
    fill_method : str or None, default=None
        Method for handling missing values
    app_user_id : int, default=-1
        Filter to specific user ID
    date_range : tuple or None, default=None
        Optional date range for filtering
    """

    def __init__(self, aggregation='hourly', metrics=None, fill_method=None,
                 app_user_id=-1, date_range=None):
        self.aggregation = aggregation
        self.metrics = metrics or ['steps', 'speed', 'distance', 'calorie']
        self.fill_method = fill_method
        self.app_user_id = app_user_id
        self.date_range = date_range

    def fit(self, X, y=None):
        """Fit does nothing, returns self."""
        return self

    def transform(self, X):
        """
        Transform raw health data into aggregated features.

        Parameters:
        -----------
        X : dict
            Dictionary with health metric names as keys and DataFrames as values

        Returns:
        --------
        DataFrame : Aggregated health data with all metrics
        """
        processed_dfs = []

        for metric in self.metrics:
            if metric not in X or X[metric] is None or X[metric].empty:
                continue

            df = X[metric]

            # Determine aggregation function
            if metric == 'speed':
                agg_func = 'mean'
            else:
                agg_func = 'sum'

            # Determine time unit
            time_unit = 'H' if self.aggregation == 'hourly' else 'D'

            # Process the dataframe
            processed = _process_passive_data_dataframe(
                df=df,
                agg_func=agg_func,
                time_unit=time_unit,
                app_user_id=self.app_user_id,
                date_range=self.date_range
            )

            if processed is not None and not processed.empty:
                # Rename value column to metric name
                value_cols = [col for col in processed.columns if col not in ['time_key', 'app_user_id']]
                if value_cols:
                    processed = processed.rename(columns={value_cols[0]: metric})
                processed_dfs.append(processed)

        # Merge all processed dataframes
        if not processed_dfs:
            return pd.DataFrame()

        if len(processed_dfs) == 1:
            return processed_dfs[0]

        # Merge on time_key and app_user_id
        result = processed_dfs[0]
        for df in processed_dfs[1:]:
            result = pd.merge(
                result, df,
                on=['time_key', 'app_user_id'],
                how='outer'
            )

        return result


class LabelExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts and prepares target labels from survey data.

    Supports all daily labels:
    - phq9 (depression)
    - suicide_risk, self_harm, positive_emotion, negative_emotion
    - social_stress, social_connection, minority_stress
    - emotion_regulation, sleep

    Parameters:
    -----------
    target_type : str, default='phq9'
        Type of target: 'phq9' or any daily label
    """

    def __init__(self, target_type='phq9'):
        self.target_type = target_type
        self.labels_ = None

    def fit(self, X=None, y=None):
        """Extract label data based on target type."""
        if self.target_type == 'phq9':
            self.labels_ = get_phq9_dataframe()
        elif self.target_type in [
            'suicide_risk', 'self_harm', 'sleep', 'positive_emotion',
            'negative_emotion', 'social_stress', 'social_connection',
            'minority_stress', 'emotion_regulation'
        ]:
            self.labels_ = get_daily_labels_dataframe()
        else:
            raise ValueError(f"Unknown target_type: {self.target_type}")

        return self

    def transform(self, X=None):
        """Return the extracted labels."""
        return self.labels_


class FeatureLabelMerger(BaseEstimator, TransformerMixin):
    """
    Merges processed features with target labels.

    Supports all daily labels:
    - phq9 (depression)
    - suicide_risk, self_harm, positive_emotion, negative_emotion
    - social_stress, social_connection, minority_stress
    - emotion_regulation, sleep

    Parameters:
    -----------
    target_type : str, default='phq9'
        Type of target: 'phq9' or any daily label
    time_windows : list of int or None, default=None
        List of time windows (hours before survey) for screentime features
        If None, uses weekly aggregation for PHQ9
    propagate_labels : bool, default=False
        Whether to propagate positive labels to all user entries
    use_accurate_method : bool, default=True
        DEPRECATED and ignored. App-table screentime is always used.
    """

    def __init__(self, target_type='phq9', time_windows=None, propagate_labels=False, use_accurate_method=True):
        self.target_type = target_type
        self.time_windows = time_windows
        self.propagate_labels = propagate_labels
        self.use_accurate_method = use_accurate_method

    def fit(self, X, y=None):
        """Fit does nothing, returns self."""
        return self

    def transform(self, X=None):
        """
        Merge features with labels by calling the existing merge functions.

        Parameters:
        -----------
        X : None
            Ignored - the merge functions extract data internally

        Returns:
        --------
        dict : Dictionary of DataFrames by time window, ready for modeling
        """
        # Use existing merge functions which handle data extraction internally
        if self.target_type == 'phq9':
            if self.time_windows:
                # Screentime with PHQ9 - returns dict of DataFrames by window
                merged_dict = {}
                for window in self.time_windows:
                    merged = merge_daily_screentime_features_with_phq9(
                        hours_before_survey=window,
                        use_accurate_method=True
                    )
                    if self.propagate_labels and not merged.empty:
                        merged = propagate_positive_labels(
                            merged, 'severity_label', 'depressed'
                        )
                    merged_dict[window] = merged
                return merged_dict
            else:
                # Weekly health data with PHQ9
                raise NotImplementedError("Weekly health-PHQ9 merge not yet wrapped")

        else:  # Risk labels
            if self.time_windows:
                # Screentime with risk labels
                label_col_map = {
                    'suicide_risk': 'suicide_risk_label',
                    'self_harm': 'self_harm_risk_label',
                    'sleep': 'sleep_label',
                    'positive_emotion': 'positive_emotion_label',
                    'negative_emotion': 'negative_emotion_label',
                    'social_stress': 'social_stress_label',
                    'social_connection': 'social_connection_label',
                    'minority_stress': 'minority_stress_label',
                    'emotion_regulation': 'emotion_regulation_label'
                }
                label_col = label_col_map[self.target_type]

                merged_dict = {}
                for window in self.time_windows:
                    merged = merge_daily_screentime_features_with_risk_labels(
                        label_column=label_col,
                        hours_before_survey=window,
                        use_accurate_method=True
                    )

                    # Filter out N/A labels for sleep (afternoon surveys without sleep data)
                    if self.target_type == 'sleep' and not merged.empty:
                        merged = merged[merged['sleep_label'] != 'N/A']

                    if self.propagate_labels and not merged.empty:
                        merged = propagate_positive_labels(
                            merged, label_col, 'at_risk'
                        )
                    merged_dict[window] = merged
                return merged_dict
            else:
                raise NotImplementedError("Weekly health-risk merge not yet wrapped")


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects relevant features for modeling.

    Parameters:
    -----------
    feature_type : str, default='hourly'
        Type of features: 'hourly', 'daily', 'weekly'
    exclude_cols : list of str or None
        Columns to exclude from features (e.g., IDs, timestamps)
    """

    def __init__(self, feature_type='hourly', exclude_cols=None):
        self.feature_type = feature_type
        self.exclude_cols = exclude_cols or [
            'app_user_id', 'survey_response_id', 'timestamp',
            'week_start', 'time_key', 'date'
        ]
        self.feature_names_ = None

    def fit(self, X, y=None):
        """Determine which columns are features."""
        if isinstance(X, dict):
            # Get features from first window
            sample_df = next(iter(X.values()))
        else:
            sample_df = X

        # Identify label columns
        label_cols = [
            'severity_label', 'phq9_total_score',
            'suicide_risk_label', 'self_harm_risk_label', 'sleep_label'
        ]

        all_exclude = self.exclude_cols + label_cols

        # Get feature columns
        self.feature_names_ = [
            col for col in sample_df.columns
            if col not in all_exclude
        ]

        return self

    def transform(self, X):
        """
        Extract feature columns.

        Returns:
        --------
        DataFrame or dict : Data with only feature columns
        """
        if isinstance(X, dict):
            # Handle dict of DataFrames (multiple time windows)
            result = {}
            for key, df in X.items():
                result[key] = df[self.feature_names_]
            return result
        else:
            return X[self.feature_names_]


class LabelEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes target labels for modeling.

    Supports all daily labels:
    - phq9 (depression)
    - suicide_risk, self_harm, positive_emotion, negative_emotion
    - social_stress, social_connection, minority_stress
    - emotion_regulation, sleep

    Parameters:
    -----------
    target_type : str, default='phq9'
        Type of target to encode
    """

    def __init__(self, target_type='phq9'):
        self.target_type = target_type
        self.label_col_ = None
        self.classes_ = None

    def fit(self, X, y=None):
        """Determine label column and classes."""
        if self.target_type == 'phq9':
            self.label_col_ = 'severity_label'
            self.classes_ = ['not_depressed', 'depressed']
        elif self.target_type in [
            'suicide_risk', 'self_harm', 'sleep', 'positive_emotion',
            'negative_emotion', 'social_stress', 'social_connection',
            'minority_stress', 'emotion_regulation'
        ]:
            label_col_map = {
                'suicide_risk': 'suicide_risk_label',
                'self_harm': 'self_harm_risk_label',
                'sleep': 'sleep_label',
                'positive_emotion': 'positive_emotion_label',
                'negative_emotion': 'negative_emotion_label',
                'social_stress': 'social_stress_label',
                'social_connection': 'social_connection_label',
                'minority_stress': 'minority_stress_label',
                'emotion_regulation': 'emotion_regulation_label'
            }
            self.label_col_ = label_col_map[self.target_type]
            self.classes_ = ['not_at_risk', 'at_risk']
        else:
            raise ValueError(f"Unknown target_type: {self.target_type}")

        return self

    def transform(self, X):
        """
        Extract target labels.

        Returns:
        --------
        Series or dict : Target labels
        """
        if isinstance(X, dict):
            result = {}
            for key, df in X.items():
                if self.label_col_ in df.columns:
                    result[key] = df[self.label_col_]
            return result
        else:
            if self.label_col_ in X.columns:
                return X[self.label_col_]
            return None


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Handles missing values in the feature data.

    Parameters:
    -----------
    strategy : str, default='zero'
        Strategy for handling missing values: 'zero', 'mean', 'median',
        'ffill', 'bfill', 'interpolate', or 'drop'
    """

    def __init__(self, strategy='zero'):
        self.strategy = strategy
        self.fill_values_ = {}

    def fit(self, X, y=None):
        """Learn fill values if using mean/median strategy."""
        if self.strategy == 'mean':
            if isinstance(X, dict):
                sample_df = next(iter(X.values()))
            else:
                sample_df = X
            self.fill_values_ = sample_df.mean().to_dict()
        elif self.strategy == 'median':
            if isinstance(X, dict):
                sample_df = next(iter(X.values()))
            else:
                sample_df = X
            self.fill_values_ = sample_df.median().to_dict()

        return self

    def transform(self, X):
        """Apply missing value handling strategy."""
        def _handle_missing(df):
            if self.strategy == 'zero':
                return df.fillna(0)
            elif self.strategy == 'mean':
                return df.fillna(self.fill_values_)
            elif self.strategy == 'median':
                return df.fillna(self.fill_values_)
            elif self.strategy == 'ffill':
                return df.fillna(method='ffill')
            elif self.strategy == 'bfill':
                return df.fillna(method='bfill')
            elif self.strategy == 'interpolate':
                return df.interpolate(method='linear')
            elif self.strategy == 'drop':
                return df.dropna()
            else:
                return df

        if isinstance(X, dict):
            return {key: _handle_missing(df) for key, df in X.items()}
        else:
            return _handle_missing(X)


class SubWindowFeatureLabelMerger(BaseEstimator, TransformerMixin):
    """
    Merges traditional hourly screentime features AND sub-window category features with labels.

    This transformer creates a COMBINED feature set with:
    1. Traditional hourly screentime features (hour_0, hour_1, ..., hour_N)
    2. Sub-window category features calculated from app usage patterns:
       - Most used app category per sub-window
       - Time spent in that category
       - Number of unique apps used

    For example, with lookback_hours=12 and subwindow_hours=3:
    - 12 hourly features (hour_0 through hour_11)
    - 12 sub-window features (4 sub-windows × 3 features each)
    - Total: 24 features + metadata/labels

    Parameters:
    -----------
    target_type : str, default='suicide_risk'
        Type of target: 'phq9', 'suicide_risk', 'self_harm', or 'sleep'
    lookback_hours : int, default=12
        Total hours to look back from survey
    subwindow_hours : int, default=3
        Size of each sub-window in hours
    propagate_labels : bool, default=False
        Whether to propagate positive labels to all user entries
    use_accurate_method : bool, default=True
        DEPRECATED and ignored. App-table screentime is always used.
    """

    def __init__(self, target_type='suicide_risk', lookback_hours=12,
                 subwindow_hours=3, propagate_labels=False, use_accurate_method=True, standardized=True):
        self.target_type = target_type
        self.lookback_hours = lookback_hours
        self.subwindow_hours = subwindow_hours
        self.propagate_labels = propagate_labels
        self.use_accurate_method = use_accurate_method
        self.standardized = standardized

    def fit(self, X, y=None):
        """Store reference to categorized data if provided."""
        self.categorized_data_ = X if isinstance(X, pd.DataFrame) else None
        return self

    def transform(self, X=None):
        """
        Merge sub-window features with traditional hourly features and labels.

        This combines BOTH:
        - Traditional hourly screentime features (hour_0, hour_1, etc.)
        - Sub-window category features (most_used_category, time, num_apps)

        Parameters:
        -----------
        X : pd.DataFrame or None
            Pre-categorized screentime app data from upstream transformer (e.g., ScreentimeAppCategorizer).
            If None, will load data internally (slower).

        Returns:
        --------
        dict : Dictionary with single key (lookback_hours) mapping to DataFrame
               DataFrame contains both hourly features AND sub-window features
        """
        from src.data_processing.merge_passive_data_and_labels import (
            merge_subwindow_screentime_features_with_risk_labels,
            merge_subwindow_screentime_features_with_phq9,
            merge_daily_screentime_features_with_risk_labels,
            merge_daily_screentime_features_with_phq9,
            propagate_positive_labels
        )
        import pandas as pd

        def _standardize_to_9am(series: pd.Series) -> pd.Series:
            series = pd.to_datetime(series, errors='coerce')
            return series.dt.normalize() + pd.Timedelta(hours=9)

        # Use pre-categorized data if available
        categorized_data = X if isinstance(X, pd.DataFrame) else getattr(self, 'categorized_data_', None)

        # Step 1: Get traditional hourly screentime features
        if self.target_type == 'phq9':
            hourly_features = merge_daily_screentime_features_with_phq9(
                hours_before_survey=self.lookback_hours,
                use_accurate_method=True
            )
            if self.standardized and not hourly_features.empty:
                if 'survey_timestamp' in hourly_features.columns:
                    hourly_features['survey_timestamp'] = _standardize_to_9am(hourly_features['survey_timestamp'])
                elif 'timestamp' in hourly_features.columns:
                    hourly_features['timestamp'] = _standardize_to_9am(hourly_features['timestamp'])
            label_col = 'severity_label'
            positive_class = 'depressed'
        else:
            # Risk labels - support all 9 daily labels
            label_col_map = {
                'suicide_risk': 'suicide_risk_label',
                'self_harm': 'self_harm_risk_label',
                'sleep': 'sleep_label',
                'positive_emotion': 'positive_emotion_label',
                'negative_emotion': 'negative_emotion_label',
                'social_stress': 'social_stress_label',
                'social_connection': 'social_connection_label',
                'minority_stress': 'minority_stress_label',
                'emotion_regulation': 'emotion_regulation_label'
            }
            label_col = label_col_map.get(self.target_type)
            if label_col is None:
                raise ValueError(f"Unknown target_type: {self.target_type}")

            hourly_features = merge_daily_screentime_features_with_risk_labels(
                label_column=label_col,
                hours_before_survey=self.lookback_hours,
                use_accurate_method=True
            )
            if self.standardized and not hourly_features.empty:
                if 'survey_timestamp' in hourly_features.columns:
                    hourly_features['survey_timestamp'] = _standardize_to_9am(hourly_features['survey_timestamp'])
                elif 'timestamp' in hourly_features.columns:
                    hourly_features['timestamp'] = _standardize_to_9am(hourly_features['timestamp'])
            positive_class = 'at_risk'

        # Step 2: Get sub-window category features using pre-categorized data
        if self.target_type == 'phq9':
            subwindow_features = merge_subwindow_screentime_features_with_phq9(
                screentime_app_df=categorized_data,  # Pass pre-categorized data
                lookback_hours=self.lookback_hours,
                subwindow_hours=self.subwindow_hours,
                standardized=self.standardized
            )
        else:
            subwindow_features = merge_subwindow_screentime_features_with_risk_labels(
                screentime_app_df=categorized_data,  # Pass pre-categorized data
                label_column=label_col,
                lookback_hours=self.lookback_hours,
                subwindow_hours=self.subwindow_hours,
                standardized=self.standardized
            )

        # If standardized, align sub-window timestamps to 9AM to match hourly features
        if self.standardized and not subwindow_features.empty:
            if 'timestamp' in subwindow_features.columns:
                subwindow_features['timestamp'] = _standardize_to_9am(subwindow_features['timestamp'])
            if 'survey_timestamp' in subwindow_features.columns:
                subwindow_features['survey_timestamp'] = _standardize_to_9am(subwindow_features['survey_timestamp'])

        # Step 3: Merge both feature sets
        # Merge on common columns: app_user_id and timestamp/survey_timestamp
        if hourly_features.empty or subwindow_features.empty:
            print("Warning: One of the feature sets is empty, returning hourly features only")
            merged = hourly_features
        else:
            # Identify merge keys
            merge_keys = ['app_user_id']

            hourly_has_survey = 'survey_timestamp' in hourly_features.columns
            hourly_has_ts = 'timestamp' in hourly_features.columns
            sub_has_survey = 'survey_timestamp' in subwindow_features.columns
            sub_has_ts = 'timestamp' in subwindow_features.columns

            if hourly_has_survey and sub_has_ts:
                subwindow_features = subwindow_features.rename(columns={'timestamp': 'survey_timestamp'})
                sub_has_survey = True
                sub_has_ts = False

            if hourly_has_survey and sub_has_survey:
                merge_keys.append('survey_timestamp')
            elif hourly_has_ts and sub_has_ts:
                merge_keys.append('timestamp')

            # Merge the dataframes
            subwindow_cols_to_keep = [col for col in subwindow_features.columns
                                     if col not in [label_col, 'survey_response_id'] or col in merge_keys]

            merged = pd.merge(
                hourly_features,
                subwindow_features[subwindow_cols_to_keep],
                on=merge_keys,
                how='inner'
            )

            if merged.empty and len(merge_keys) > 1:
                print("Warning: No rows after merging on timestamp; retrying merge on app_user_id only")
                merged = pd.merge(
                    hourly_features,
                    subwindow_features[subwindow_cols_to_keep],
                    on=['app_user_id'],
                    how='inner'
                )

        # Filter out N/A labels for sleep
        if self.target_type == 'sleep' and not merged.empty:
            merged = merged[merged['sleep_label'] != 'N/A']


        # Apply label propagation if requested
        if self.propagate_labels and not merged.empty:
            merged = propagate_positive_labels(merged, label_col, positive_class)

        # Return as dict with lookback_hours as key (for consistency with other pipelines)
        return {self.lookback_hours: merged}
