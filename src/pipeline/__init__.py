"""
Scikit-learn Pipeline Package for Mental Health Prediction

This package provides a refactored, modular pipeline architecture for
mental health prediction using passive data (screentime, health metrics)
and survey labels (PHQ-9, suicide risk, self-harm, sleep).

Main Components:
----------------
- transformers.py: Custom sklearn transformers for data processing
- mental_health_pipeline.py: Pre-configured data processing pipelines
- model_pipeline.py: Complete end-to-end modeling pipelines

Quick Start:
------------
from src.pipeline.model_pipeline import ScreentimeModelPipeline

# Create pipeline for suicide risk prediction
pipeline = ScreentimeModelPipeline(
    target_type='suicide_risk',
    time_windows=[3, 6, 9, 12],
    use_loocv=True,
    balanced_class_weight=True
)

# Run complete pipeline
results = pipeline.fit_predict()
"""

from src.pipeline.transformers import (
    DataExtractor,
    ScreentimeProcessor,
    HealthDataProcessor,
    LabelExtractor,
    FeatureLabelMerger,
    FeatureSelector,
    LabelEncoder,
    MissingValueHandler
)

from src.pipeline.mental_health_pipeline import (
    create_screentime_risk_pipeline,
    create_screentime_phq9_pipeline,
    create_health_weekly_pipeline,
    get_pipeline_for_task
)

from src.pipeline.model_pipeline import (
    ScreentimeModelPipeline,
    run_experiment
)

__all__ = [
    # Transformers
    'DataExtractor',
    'ScreentimeProcessor',
    'HealthDataProcessor',
    'LabelExtractor',
    'FeatureLabelMerger',
    'FeatureSelector',
    'LabelEncoder',
    'MissingValueHandler',

    # Pipeline builders
    'create_screentime_risk_pipeline',
    'create_screentime_phq9_pipeline',
    'create_health_weekly_pipeline',
    'get_pipeline_for_task',

    # Complete pipelines
    'ScreentimeModelPipeline',
    'run_experiment'
]

