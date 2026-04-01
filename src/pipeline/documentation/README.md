# Scikit-Learn Pipeline Architecture for Mental Health Prediction

This directory contains a refactored pipeline architecture using scikit-learn's Pipeline API for mental health prediction tasks.

## Overview

The pipeline architecture provides:
- **Modular components** as custom sklearn transformers
- **Reproducible workflows** with clear dependencies
- **Easy experimentation** with different configurations
- **Type safety** and validation at each step
- **Clean separation** between data processing and modeling

## Architecture

### 1. Custom Transformers (`transformers.py`)

All data processing steps are implemented as sklearn `BaseEstimator` and `TransformerMixin` subclasses:

- **DataExtractor**: Extracts raw data from database or CSV files
- **ScreentimeProcessor**: Processes screentime data into hourly features
- **HealthDataProcessor**: Processes health metrics (steps, speed, distance, calories)
- **LabelExtractor**: Extracts target labels from survey data
- **FeatureLabelMerger**: Merges features with labels based on timestamps
- **FeatureSelector**: Selects relevant features for modeling
- **LabelEncoder**: Encodes target labels
- **MissingValueHandler**: Handles missing values with various strategies

### 2. Pipeline Builders (`mental_health_pipeline.py`)

Pre-configured pipelines for common tasks:

- **create_screentime_risk_pipeline()**: For suicide/self-harm/sleep risk prediction
- **create_screentime_phq9_pipeline()**: For depression prediction
- **create_health_weekly_pipeline()**: For weekly health data aggregation
- **get_pipeline_for_task()**: Factory function for common tasks

### 3. Complete Model Pipelines (`model_pipeline.py`)

End-to-end pipelines that include modeling:

- **ScreentimeModelPipeline**: Complete pipeline with training and evaluation
- **run_experiment()**: Convenience function for running experiments

## Usage Examples

### Example 1: Basic Suicide Risk Prediction

```python
from src.pipeline.model_pipeline import ScreentimeModelPipeline

# Create pipeline
pipeline = ScreentimeModelPipeline(
    target_type='suicide_risk',
    time_windows=[3, 6, 9, 12],  # Hours before survey
    fill_method='zero',           # Handle missing values
    use_loocv=False,              # Use train/test split
    balanced_class_weight=False   # Don't balance classes
)

# Run complete pipeline (extract, process, train, evaluate)
results = pipeline.fit_predict()

# Access results
for window, window_results in results.items():
    print(f"\nWindow: {window} hours")
    for model_name, metrics in window_results['models'].items():
        print(f"{model_name}: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}")
```

### Example 2: With Label Propagation and Class Balancing

```python
from src.pipeline.model_pipeline import ScreentimeModelPipeline

pipeline = ScreentimeModelPipeline(
    target_type='self_harm',
    time_windows=[6, 12, 24],
    propagate_labels=True,        # Treat users with any positive label as always positive
    balanced_class_weight=True,   # Use balanced class weights
    use_loocv=True               # Use leave-one-user-out CV
)

results = pipeline.fit_predict()
```

### Example 3: Depression Prediction (PHQ-9)

```python
from src.pipeline.model_pipeline import ScreentimeModelPipeline

pipeline = ScreentimeModelPipeline(
    target_type='phq9',
    time_windows=[3, 6, 9],
    fill_method='interpolate',    # Use linear interpolation for missing values
    balanced_class_weight=True
)

results = pipeline.fit_predict()
```

### Example 4: Using the Convenience Function

```python
from src.pipeline.model_pipeline import run_experiment

# Run a complete experiment with one function call
results = run_experiment(
    target_type='suicide_risk',
    time_windows=[3, 6, 9, 12],
    propagate_labels=True,
    balanced_class_weight=True,
    use_loocv=True
)
```

### Example 5: Custom Pipeline with Individual Transformers

```python
from sklearn.pipeline import Pipeline
from src.pipeline.transformers import (
    DataExtractor,
    ScreentimeProcessor,
    MissingValueHandler
)

# Build custom pipeline
custom_pipeline = Pipeline([
    ('extract', DataExtractor(source='database', data_types=['screentime'])),
    ('process', ScreentimeProcessor(fill_method='zero', app_user_id=5)),
    ('handle_missing', MissingValueHandler(strategy='interpolate'))
])

# Fit and transform
custom_pipeline.fit(None)
processed_data = custom_pipeline.transform(None)
```

### Example 6: Data Processing Only (No Modeling)

```python
from src.pipeline.mental_health_pipeline import get_pipeline_for_task

# Get pre-configured pipeline
pipeline = get_pipeline_for_task(
    task='screentime_suicide_risk',
    time_windows=[6, 12],
    fill_method='zero'
)

# Process data
pipeline.fit(None)
merged_data = pipeline.transform(None)

# Now use merged_data for custom modeling
```

## Configuration Options

### Target Types (Daily Labels)

The pipeline supports the following daily mental health labels for prediction:

#### Risk & Safety Labels
- `'suicide_risk'`: Suicide risk prediction (binary: at_risk/not_at_risk)
  - Classifies likelihood of suicidal ideation or intent
  - Based on daily survey responses
  
- `'self_harm'`: Self-harm risk prediction (binary: at_risk/not_at_risk)
  - Classifies likelihood of self-injurious behavior
  - Based on daily survey responses

#### Sleep
- `'sleep'`: Sleep quality/adequacy prediction (binary: at_risk/not_at_risk)
  - Classifies sleep issues or insufficient sleep
  - Based on daily self-reported sleep data

#### Emotional Wellbeing
- `'positive_emotion'`: Positive emotion prediction (binary: present/absent)
  - Predicts presence of positive affect or life satisfaction
  - Based on daily survey responses
  
- `'negative_emotion'`: Negative emotion prediction (binary: present/absent)
  - Classifies presence of negative affect or distress
  - Based on daily survey responses
  
- `'emotion_regulation'`: Emotion regulation capacity (binary: at_risk/not_at_risk)
  - Classifies ability to manage and regulate emotions
  - Based on daily survey responses

#### Social & Relational
- `'social_connection'`: Social connection prediction (binary: poor/adequate)
  - Classifies quality and strength of social relationships
  - Based on daily survey responses
  
- `'social_stress'`: Social stress prediction (binary: stressed/not_stressed)
  - Classifies presence of interpersonal conflict or stress
  - Based on daily survey responses

#### Minority Mental Health
- `'minority_stress'`: Minority stress prediction (binary: stressed/not_stressed)
  - Classifies stress related to minority identity/stigma
  - Based on daily survey responses

#### Clinical Assessment
- `'phq9'`: PHQ-9 depression screening (binary: depressed/not_depressed)
  - Classifies clinical depression based on PHQ-9 criteria
  - Based on weekly PHQ-9 survey responses

**Example usage:**
```python
from src.pipeline.model_pipeline import ScreentimeModelPipeline

# Predict any of the 10 labels
for target in ['suicide_risk', 'self_harm', 'sleep', 'positive_emotion', 
               'negative_emotion', 'emotion_regulation', 'social_connection',
               'social_stress', 'minority_stress', 'phq9']:
    pipeline = ScreentimeModelPipeline(
        target_type=target,
        time_windows=[6, 12]
    )
    results = pipeline.fit_predict()
    print(f"{target}: {results}")
```

### Time Windows
List of integers representing hours before survey to use for features.
Example: `[3, 6, 9, 12]` will create separate models for 3h, 6h, 9h, and 12h windows.

### Fill Methods (Missing Value Handling)
- `None`: No filling, keep NaN values
- `'zero'`: Replace NaN with 0
- `'ffill'`: Forward fill
- `'bfill'`: Backward fill
- `'interpolate'`: Linear interpolation
- `'mean'`: Replace with mean
- `'median'`: Replace with median
- `'drop'`: Drop rows with NaN

### Cross-Validation Options
- `use_loocv=False`: Standard train/test split (default 70/30)
- `use_loocv=True`: Leave-one-user-out cross-validation (LOOCV)

### Class Balancing Options
- `balanced_class_weight=False`: No class balancing
- `balanced_class_weight=True`: Use sklearn's 'balanced' class weights

### Label Propagation
- `propagate_labels=False`: Use original labels
- `propagate_labels=True`: If a user has any positive label, treat all their entries as positive

## Benefits of This Architecture

### 1. Modularity
Each processing step is a separate, testable component that can be used independently or combined.

### 2. Reproducibility
Pipelines ensure the same transformations are applied consistently across train/test splits and different runs.

### 3. Easy Experimentation
Change parameters without modifying code:
```python
# Experiment with different configurations
for window in [3, 6, 9, 12, 24]:
    for fill_method in ['zero', 'ffill', 'interpolate']:
        pipeline = ScreentimeModelPipeline(
            target_type='suicide_risk',
            time_windows=[window],
            fill_method=fill_method
        )
        results = pipeline.fit_predict()
        # Compare results...
```

### 4. Grid Search Compatible
Can use sklearn's GridSearchCV with the pipelines:
```python
from sklearn.model_selection import GridSearchCV

# Create pipeline
pipeline = create_screentime_risk_pipeline(target_type='suicide_risk')

# Define parameter grid
param_grid = {
    'process_screentime__fill_method': ['zero', 'ffill', 'interpolate'],
    'merge_features_labels__propagate_labels': [True, False]
}

# Grid search (would need additional wrapper for compatibility)
# grid_search = GridSearchCV(pipeline, param_grid, cv=3)
```

### 5. Easy to Extend
Add new transformers by implementing `fit()` and `transform()`:
```python
class MyCustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Learn parameters from data
        return self
    
    def transform(self, X):
        # Apply transformation
        return X_transformed
```

## Integration with Existing Code

The pipeline wraps your existing functions from:
- `src/data_processing/passive_data_analysis.py`
- `src/data_processing/merge_passive_data_and_labels.py`
- `src/categorization/*.py`
- `src/modeling/model_screentime_time_windows.py`

You can continue using the old code or gradually migrate to the pipeline architecture.

## Comparison: Old vs New

### Old Approach (model_screentime_time_windows.py)
```python
# Scattered logic across multiple functions
data = merge_daily_screentime_features_with_risk_labels(
    label_column='suicide_risk_label',
    screentime_window_hours=6
)

if propagate_labels:
    data = propagate_positive_labels(data, 'suicide_risk_label', 'at_risk')

X = data[feature_cols]
y = data['suicide_risk_label']

# Train model...
```

### New Approach (Pipeline)
```python
# Clear, modular pipeline
pipeline = ScreentimeModelPipeline(
    target_type='suicide_risk',
    time_windows=[6],
    propagate_labels=True
)

results = pipeline.fit_predict()
```

## Future Enhancements

Potential additions:
1. **Multimodal pipelines**: Combine screentime + health data
2. **Feature engineering transformers**: Automatic feature creation
3. **Hyperparameter optimization**: Built-in grid search
4. **Model ensembles**: Combine multiple models
5. **Real-time prediction**: Streaming data support
6. **Model persistence**: Save/load trained pipelines
7. **Explainability**: SHAP/LIME integration

## Requirements

The pipeline uses:
- `scikit-learn` (pipelines, models, metrics)
- `pandas` (data manipulation)
- `numpy` (numerical operations)
- `matplotlib` / `seaborn` (visualization)

All requirements are already in your `requirements.txt`.

## Testing

Run tests for the pipeline:
```bash
# Test individual transformers
python -m pytest src/pipeline/tests/test_transformers.py

# Test complete pipelines
python -m pytest src/pipeline/tests/test_pipelines.py

# Test modeling
python -m pytest src/pipeline/tests/test_model_pipeline.py
```

## Questions?

For more details on scikit-learn pipelines:
- [Sklearn Pipeline Documentation](https://scikit-learn.org/stable/modules/compose.html)
- [Custom Transformers Guide](https://scikit-learn.org/stable/developers/develop.html)
