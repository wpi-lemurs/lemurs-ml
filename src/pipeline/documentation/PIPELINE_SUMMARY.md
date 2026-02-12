# Pipeline Refactoring Summary

## Overview

This document summarizes the refactoring of your mental health prediction codebase to use scikit-learn's Pipeline architecture.

## What Was Created

### 1. Core Pipeline Components (`src/pipeline/`)

#### `transformers.py` (624 lines)
Custom scikit-learn transformers that wrap your existing data processing logic:

- **DataExtractor**: Extracts data from database or CSV files
- **ScreentimeProcessor**: Processes screentime into hourly features
- **HealthDataProcessor**: Processes health metrics (steps, speed, distance, calories)
- **LabelExtractor**: Extracts target labels from survey data
- **FeatureLabelMerger**: Merges features with labels based on timestamps
- **FeatureSelector**: Selects relevant features for modeling
- **LabelEncoder**: Encodes target labels
- **MissingValueHandler**: Handles missing values with various strategies

#### `mental_health_pipeline.py` (295 lines)
Pre-configured pipelines for common tasks:

- `create_screentime_risk_pipeline()`: For suicide/self-harm/sleep risk prediction
- `create_screentime_phq9_pipeline()`: For depression prediction
- `create_health_weekly_pipeline()`: For weekly health data aggregation
- `get_pipeline_for_task()`: Factory function for common tasks
- Helper classes for combining transformer outputs

#### `model_pipeline.py` (458 lines)
Complete end-to-end modeling pipelines:

- **ScreentimeModelPipeline**: Complete pipeline with data extraction, processing, training, and evaluation
- Supports multiple time windows, models, cross-validation strategies
- Automatic result saving and visualization
- `run_experiment()`: Convenience function for quick experiments

#### `__init__.py` (58 lines)
Package initialization with clean API exports

### 2. Documentation

#### `README.md`
Comprehensive guide with:
- Architecture overview
- 6 usage examples
- Configuration options
- Benefits explanation
- Future enhancements

#### `MIGRATION_GUIDE.md`
Step-by-step migration guide with:
- 6 before/after code examples
- Function mapping table
- Gradual migration strategy
- Troubleshooting tips

#### `example_pipeline_usage.py` (205 lines)
Runnable examples demonstrating:
- Basic suicide risk prediction
- Label propagation
- Class balancing
- LOOCV
- Configuration comparison

## Key Benefits

### 1. Code Reduction
Your modeling code goes from ~100 lines to ~10 lines:

**Before:**
```python
# ~100 lines of data extraction, processing, merging, training, evaluation
```

**After:**
```python
pipeline = ScreentimeModelPipeline(
    target_type='suicide_risk',
    time_windows=[3, 6, 9, 12],
    use_loocv=True,
    balanced_class_weight=True
)
results = pipeline.fit_predict()
```

### 2. Modularity
Each component is independent and testable:
```python
# Use just the data processing part
data_pipeline = create_screentime_risk_pipeline(...)
processed_data = data_pipeline.fit_transform(None)

# Or just extract data
extractor = DataExtractor()
raw_data = extractor.fit_transform(None)
```

### 3. Reproducibility
Pipelines guarantee consistent transformations:
- Same preprocessing across train/test splits
- Same feature engineering across runs
- No risk of train/test leakage

### 4. Experimentation
Easy to try different configurations:
```python
for window in [3, 6, 9, 12]:
    for fill_method in ['zero', 'ffill', 'interpolate']:
        for balanced in [True, False]:
            pipeline = ScreentimeModelPipeline(
                target_type='suicide_risk',
                time_windows=[window],
                fill_method=fill_method,
                balanced_class_weight=balanced
            )
            results = pipeline.fit_predict()
            # Compare results...
```

### 5. Sklearn Integration
Compatible with sklearn tools:
- Grid search for hyperparameter tuning
- Pipeline composition
- Model selection utilities

## How It Maps to Your Existing Code

### Old Structure
```
model_screentime_time_windows.py
├── Data extraction (merge functions)
├── Label propagation (if needed)
├── Feature preparation
├── Train/test split or LOOCV
├── Model training (LR and RF)
├── Evaluation
└── Plotting
```

### New Structure
```
ScreentimeModelPipeline
├── Data Pipeline
│   ├── DataExtractor
│   ├── ScreentimeProcessor
│   ├── LabelExtractor
│   └── FeatureLabelMerger
└── Model Training
    ├── Train/test split or LOOCV
    ├── Multiple models
    ├── Evaluation
    └── Visualization
```

## Integration with Existing Code

The pipeline **wraps** your existing functions without replacing them:

- `DataExtractor` → calls `DatabaseService.extract_from_database()`
- `ScreentimeProcessor` → calls `hourly_screentime_data()`
- `LabelExtractor` → calls `get_phq9_dataframe()`, `get_daily_labels_dataframe()`
- `FeatureLabelMerger` → calls `merge_daily_screentime_features_with_phq9()`, etc.

**Your existing code still works!** The pipeline is an alternative interface.

## What Stays the Same

You still have access to all your original code:
- `src/data_processing/` - All functions still work
- `src/categorization/` - All labeling logic unchanged
- `src/modeling/` - Original `model_screentime_time_windows.py` still works
- `database_service.py` - Used internally by pipeline

## Usage Comparison

### Running a Complete Experiment

#### Old Way
```bash
python src/modeling/model_screentime_time_windows.py suicide_risk --balanced --loocv
```

#### New Way (Option 1)
```python
from src.pipeline import run_experiment

results = run_experiment(
    target_type='suicide_risk',
    balanced_class_weight=True,
    use_loocv=True
)
```

#### New Way (Option 2)
```python
from src.pipeline import ScreentimeModelPipeline

pipeline = ScreentimeModelPipeline(
    target_type='suicide_risk',
    time_windows=[3, 6, 9, 12],
    balanced_class_weight=True,
    use_loocv=True
)

results = pipeline.fit_predict()
```

## File Structure

```
src/pipeline/
├── __init__.py                 # Package exports
├── transformers.py             # Custom sklearn transformers
├── mental_health_pipeline.py   # Pre-configured pipelines
├── model_pipeline.py           # Complete modeling pipelines
├── README.md                   # Usage guide
├── MIGRATION_GUIDE.md          # Migration instructions
├── example_pipeline_usage.py   # Runnable examples
└── PIPELINE_SUMMARY.md         # This file
```

## Quick Start

### 1. Run Examples
```bash
cd src/pipeline
python example_pipeline_usage.py
```

### 2. Try a Simple Experiment
```python
from src.pipeline import run_experiment

results = run_experiment(
    target_type='suicide_risk',
    time_windows=[6, 12]
)
```

### 3. Compare with Old Code
```python
# Run old code
from src.modeling.model_screentime_time_windows import main as old_main
old_results = old_main(target_type='suicide_risk')

# Run new pipeline
from src.pipeline import run_experiment
new_results = run_experiment(target_type='suicide_risk')

# Compare...
```

## Advanced Features

### Custom Transformers
Create your own:
```python
from sklearn.base import BaseEstimator, TransformerMixin

class MyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Your logic
        return X_transformed
```

### Data Validation
Built-in validation:
```python
from src.pipeline.mental_health_pipeline import DataValidationTransformer

pipeline = Pipeline([
    ('extract', DataExtractor()),
    ('validate', DataValidationTransformer(min_samples=10)),
    ('process', ScreentimeProcessor())
])
```

### Multiple Time Windows
Automatically experiments with multiple windows:
```python
pipeline = ScreentimeModelPipeline(
    target_type='suicide_risk',
    time_windows=[3, 6, 9, 12, 24, 48]  # 6 experiments in one call
)
results = pipeline.fit_predict()
```

## Testing

Each component is independently testable:

```python
# Test data extraction
extractor = DataExtractor()
extractor.fit(None)
data = extractor.transform(None)
assert 'screentime' in data

# Test processing
processor = ScreentimeProcessor(fill_method='zero')
processor.fit(None)
processed = processor.transform(data)
assert processed is not None
```

## Performance Considerations

### Caching
Pipeline components can cache intermediate results:
```python
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

# Save processed data
joblib.dump(processed_data, 'processed_data.pkl')

# Load later
processed_data = joblib.load('processed_data.pkl')
```

### Parallel Processing
Can process multiple windows in parallel (future enhancement):
```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(
    delayed(run_experiment)(target_type='suicide_risk', time_windows=[w])
    for w in [3, 6, 9, 12]
)
```

## Future Enhancements

Planned additions:
1. **Multimodal pipelines**: Combine screentime + health data
2. **Feature engineering**: Automatic feature creation
3. **Hyperparameter tuning**: Built-in grid search
4. **Model ensembles**: Combine multiple models
5. **Model persistence**: Save/load trained pipelines
6. **Real-time prediction**: Streaming data support
7. **Explainability**: SHAP/LIME integration
8. **More transformers**: Text data, audio features, etc.

## Migration Path

### Phase 1: Experiment (Now)
- Try the examples
- Run pipeline alongside old code
- Compare results

### Phase 2: New Work (1-2 weeks)
- Use pipeline for new experiments
- Keep old code for established workflows
- Build confidence

### Phase 3: Full Migration (Optional)
- Replace old code calls with pipeline
- Update notebooks
- Archive old implementation

## Support

Resources:
- `README.md` - Detailed usage guide
- `MIGRATION_GUIDE.md` - Step-by-step migration
- `example_pipeline_usage.py` - Working examples
- Docstrings in source files
- Sklearn Pipeline documentation

## Conclusion

The pipeline architecture provides:
✅ Cleaner, more maintainable code
✅ Better reproducibility
✅ Easier experimentation
✅ Sklearn ecosystem integration
✅ Professional ML engineering practices

**Start using it today:**
```python
from src.pipeline import ScreentimeModelPipeline

pipeline = ScreentimeModelPipeline(target_type='suicide_risk')
results = pipeline.fit_predict()
```

That's it! Your complete mental health prediction pipeline in 3 lines.

