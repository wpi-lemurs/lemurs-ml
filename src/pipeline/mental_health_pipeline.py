"""
Scikit-learn Pipeline implementation for mental health prediction.

This module provides pre-configured pipelines for different prediction tasks
using the custom transformers defined in transformers.py.

Example usage:
    # Create a pipeline for suicide risk prediction using screentime data
    pipeline = create_screentime_risk_pipeline(
        target_type='suicide_risk',
        time_windows=[3, 6, 9, 12],
        fill_method='zero',
        propagate_labels=False
    )

    # Fit and transform
    pipeline.fit(None)  # Extracts and processes data
    merged_data = pipeline.transform(None)

    # Use with models
    from src.pipeline.model_pipeline import ScreentimeModelPipeline
    model_pipeline = ScreentimeModelPipeline(
        target_type='suicide_risk',
        time_windows=[3, 6, 9],
        use_loocv=True,
        balanced_class_weight=True
    )
    results = model_pipeline.fit_predict()
"""

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from src.pipeline.transformers import (
    DataExtractor,
    ScreentimeProcessor,
    HealthDataProcessor,
    LabelExtractor,
    FeatureLabelMerger,
    SubWindowFeatureLabelMerger,
    ScreentimeAppCategorizer
)


def create_screentime_risk_pipeline(
    target_type='suicide_risk',
    time_windows=None,
    fill_method='zero',
    app_user_id=-1,
    date_range=None,
    propagate_labels=False,
    use_accurate_method=False
):
    """
    Create a pipeline for risk prediction using screentime data.

    Parameters:
    -----------
    target_type : str, default='suicide_risk'
        Target to predict: 'suicide_risk', 'self_harm', or 'sleep'
    time_windows : list of int, default=[3, 6, 9, 12]
        Hours before survey to use for features
    fill_method : str or None, default='zero'
        Method for handling missing values
    app_user_id : int, default=-1
        Filter to specific user or -1 for all
    date_range : tuple or None
        Optional date range filter
    propagate_labels : bool, default=False
        Whether to propagate positive labels
    use_accurate_method : bool, default=False
        If True, uses calculate_accurate_screentime_from_app_table for more precise calculations

    Returns:
    --------
    Pipeline : Configured scikit-learn pipeline
    """
    if time_windows is None:
        time_windows = [3, 6, 9, 12]

    # Define pipeline steps
    # Note: We use a single combined transformer because sklearn pipelines
    # pass output of one step to the next, but we need parallel data extraction
    steps = [
        ('merge_features_labels', FeatureLabelMerger(
            target_type=target_type,
            time_windows=time_windows,
            propagate_labels=propagate_labels,
            use_accurate_method=use_accurate_method
        ))
    ]

    return Pipeline(steps)


def create_screentime_phq9_pipeline(
    time_windows=None,
    fill_method='zero',
    app_user_id=-1,
    date_range=None,
    propagate_labels=False,
    use_accurate_method=False
):
    """
    Create a pipeline for PHQ-9 depression prediction using screentime data.

    Parameters:
    -----------
    time_windows : list of int, default=[3, 6, 9, 12]
        Hours before survey to use for features
    fill_method : str or None, default='zero'
        Method for handling missing values
    app_user_id : int, default=-1
        Filter to specific user or -1 for all
    date_range : tuple or None
        Optional date range filter
    propagate_labels : bool, default=False
        Whether to propagate positive labels
    use_accurate_method : bool, default=False
        If True, uses calculate_accurate_screentime_from_app_table for more precise calculations

    Returns:
    --------
    Pipeline : Configured scikit-learn pipeline
    """
    if time_windows is None:
        time_windows = [3, 6, 9, 12]

    steps = [
        ('merge_features_labels', FeatureLabelMerger(
            target_type='phq9',
            time_windows=time_windows,
            propagate_labels=propagate_labels,
            use_accurate_method=use_accurate_method
        ))
    ]

    return Pipeline(steps)


def create_health_weekly_pipeline(
    metrics=None,
    fill_method=None,
    app_user_id=-1,
    date_range=None
):
    """
    Create a pipeline for weekly health data aggregation.

    Parameters:
    -----------
    metrics : list of str, default=['steps', 'speed', 'distance', 'calorie']
        Health metrics to include
    fill_method : str or None, default=None
        Method for handling missing values
    app_user_id : int, default=-1
        Filter to specific user or -1 for all
    date_range : tuple or None
        Optional date range filter

    Returns:
    --------
    Pipeline : Configured scikit-learn pipeline
    """
    if metrics is None:
        metrics = ['steps', 'speed', 'distance', 'calorie']

    steps = [
        ('extract', DataExtractor(source='database', data_types=metrics)),
        ('process_health', HealthDataProcessor(
            aggregation='weekly',
            metrics=metrics,
            fill_method=fill_method,
            app_user_id=app_user_id,
            date_range=date_range
        ))
    ]

    return Pipeline(steps)


def create_multimodal_pipeline(
    target_type='phq9',
    screentime_windows=None,
    health_metrics=None,
    fill_method='zero',
    app_user_id=-1,
    date_range=None,
    propagate_labels=False
):
    """
    Create a pipeline that combines multiple data modalities (screentime + health).

    Parameters:
    -----------
    target_type : str, default='phq9'
        Target to predict
    screentime_windows : list of int or None
        Time windows for screentime features
    health_metrics : list of str or None
        Health metrics to include
    fill_method : str or None, default='zero'
        Method for handling missing values
    app_user_id : int, default=-1
        Filter to specific user or -1 for all
    date_range : tuple or None
        Optional date range filter
    propagate_labels : bool, default=False
        Whether to propagate positive labels

    Returns:
    --------
    Pipeline : Configured scikit-learn pipeline
    """
    if screentime_windows is None:
        screentime_windows = [3, 6, 9]
    if health_metrics is None:
        health_metrics = ['steps', 'speed', 'distance', 'calorie']

    # This would require more complex merging logic
    # For now, we'll create parallel pipelines that can be combined
    raise NotImplementedError(
        "Multimodal pipeline not yet implemented. "
        "Use separate screentime and health pipelines for now."
    )


class CombineTransformer(BaseEstimator, TransformerMixin):
    """
    Utility transformer to combine outputs from previous pipeline steps.

    This is used to combine features and labels into a dict for the merger step.
    """

    def __init__(self):
        self.features_ = None
        self.labels_ = None

    def fit(self, X, y=None):
        """Store the most recent features."""
        if isinstance(X, dict):
            self.features_ = X
        else:
            self.features_ = X
        return self

    def transform(self, X):
        """
        Combine features and labels.

        Note: This is a simplified version. In practice, you'd need to
        handle the pipeline state more carefully to combine outputs from
        different branches.
        """
        # For now, just pass through
        # The actual combining happens in FeatureLabelMerger
        return X


class DataValidationTransformer(BaseEstimator, TransformerMixin):
    """
    Validates data quality and logs warnings/errors.

    Parameters:
    -----------
    min_samples : int, default=10
        Minimum number of samples required
    min_features : int, default=1
        Minimum number of features required
    require_both_classes : bool, default=True
        Whether to require both classes in target
    """

    def __init__(self, min_samples=10, min_features=1, require_both_classes=True):
        self.min_samples = min_samples
        self.min_features = min_features
        self.require_both_classes = require_both_classes
        self.validation_results_ = {}

    def fit(self, X, y=None):
        """Validate data quality."""
        self.validation_results_ = {}

        if isinstance(X, dict):
            # Validate each window
            for window, df in X.items():
                self.validation_results_[window] = self._validate_df(df, y)
        else:
            self.validation_results_['main'] = self._validate_df(X, y)

        return self

    def _validate_df(self, df, y):
        """Validate a single DataFrame."""
        results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }

        # Check sample count
        if len(df) < self.min_samples:
            results['errors'].append(
                f"Insufficient samples: {len(df)} < {self.min_samples}"
            )
            results['valid'] = False

        # Check feature count
        if df.shape[1] < self.min_features:
            results['errors'].append(
                f"Insufficient features: {df.shape[1]} < {self.min_features}"
            )
            results['valid'] = False

        # Check for missing values
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        if missing_pct > 0.5:
            results['warnings'].append(
                f"High percentage of missing values: {missing_pct:.1%}"
            )

        # Check class balance if y provided
        if y is not None:
            if hasattr(y, 'value_counts'):
                class_counts = y.value_counts()
                if len(class_counts) < 2 and self.require_both_classes:
                    results['errors'].append(
                        f"Only one class present: {class_counts.index[0]}"
                    )
                    results['valid'] = False
                elif len(class_counts) >= 2:
                    imbalance_ratio = class_counts.max() / class_counts.min()
                    if imbalance_ratio > 10:
                        results['warnings'].append(
                            f"High class imbalance: {imbalance_ratio:.1f}:1"
                        )

        return results

    def transform(self, X):
        """Pass through data unchanged, but log validation results."""
        # Log validation results
        for key, results in self.validation_results_.items():
            if not results['valid']:
                print(f"\nValidation FAILED for {key}:")
                for error in results['errors']:
                    print(f"  ERROR: {error}")

            if results['warnings']:
                print(f"\nValidation warnings for {key}:")
                for warning in results['warnings']:
                    print(f"  WARNING: {warning}")

        return X


def get_pipeline_for_task(
    task='screentime_suicide_risk',
    **kwargs
):
    """
    Factory function to get a pre-configured pipeline for common tasks.

    Parameters:
    -----------
    task : str
        Task name: 'screentime_suicide_risk', 'screentime_self_harm',
        'screentime_sleep', 'screentime_phq9', 'health_weekly'
    **kwargs : dict
        Additional arguments to pass to the pipeline constructor

    Returns:
    --------
    Pipeline : Configured pipeline for the task
    """
    # Filter kwargs based on task type
    if task == 'health_weekly':
        # Remove time_windows for health_weekly task
        health_kwargs = {k: v for k, v in kwargs.items() if k != 'time_windows'}
        task_map = {
            'health_weekly': lambda: create_health_weekly_pipeline(**health_kwargs),
        }
    else:
        task_map = {
            'screentime_suicide_risk': lambda: create_screentime_risk_pipeline(
                target_type='suicide_risk', **kwargs
            ),
            'screentime_self_harm': lambda: create_screentime_risk_pipeline(
                target_type='self_harm', **kwargs
            ),
            'screentime_sleep': lambda: create_screentime_risk_pipeline(
                target_type='sleep', **kwargs
            ),
            'screentime_phq9': lambda: create_screentime_phq9_pipeline(**kwargs),
        }

    if task not in task_map:
        raise ValueError(
            f"Unknown task: {task}. "
            f"Available tasks: screentime_suicide_risk, screentime_self_harm, "
            f"screentime_sleep, screentime_phq9, health_weekly"
        )

    return task_map[task]()


def create_subwindow_pipeline(
    target_type='suicide_risk',
    lookback_hours=12,
    subwindow_hours=3,
    propagate_labels=False,
    use_accurate_method=False
):
    """
    Create a pipeline for mental health prediction using sub-window screentime features.

    This pipeline separates the slow app categorization step from feature engineering:
    1. ScreentimeAppCategorizer - extracts and categorizes apps (SLOW, done once in fit())
    2. SubWindowFeatureLabelMerger - creates features using categorized data (FAST)

    The categorization happens ONCE during fit(), then transform() reuses the result.
    This makes running multiple models with different parameters much faster.

    Parameters:
    -----------
    target_type : str, default='suicide_risk'
        Target to predict: 'phq9', 'suicide_risk', 'self_harm', or 'sleep'
    lookback_hours : int, default=12
        Total hours to look back from survey
    subwindow_hours : int, default=3
        Size of each sub-window in hours
    propagate_labels : bool, default=False
        Whether to propagate positive labels
    use_accurate_method : bool, default=False
        If True, uses calculate_accurate_screentime_from_app_table for more precise calculations

    Returns:
    --------
    Pipeline : Configured scikit-learn pipeline

    Example:
    --------
    >>> # Create pipeline
    >>> pipeline = create_subwindow_pipeline(
    ...     target_type='sleep',
    ...     lookback_hours=12,
    ...     subwindow_hours=3
    ... )
    >>>
    >>> # Fit once - this does the slow categorization
    >>> pipeline.fit(None)  # Takes a few minutes
    >>>
    >>> # Transform to get features - this is fast!
    >>> data = pipeline.transform(None)  # Reuses categorized data from fit()
    >>>
    >>> # Can transform multiple times without re-categorizing
    >>> data2 = pipeline.transform(None)  # Still fast!
    """
    steps = [
        # Step 1: Categorize apps (SLOW - happens once in fit())
        ('categorize_apps', ScreentimeAppCategorizer()),

        # Step 2: Create features from categorized data (FAST - uses output from step 1)
        ('merge_features_labels', SubWindowFeatureLabelMerger(
            target_type=target_type,
            lookback_hours=lookback_hours,
            subwindow_hours=subwindow_hours,
            propagate_labels=propagate_labels,
            use_accurate_method=use_accurate_method
        ))
    ]

    return Pipeline(steps)


