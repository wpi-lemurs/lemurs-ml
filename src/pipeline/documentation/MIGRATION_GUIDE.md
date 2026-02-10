# Migration Guide: Old Code → Pipeline Architecture

This guide helps you transition from the existing modeling code to the new scikit-learn pipeline architecture.

## Why Migrate?

The pipeline architecture offers:
- ✅ **Better organization**: Clear separation of concerns
- ✅ **Reproducibility**: Same transformations applied consistently
- ✅ **Easier experimentation**: Change parameters without code changes
- ✅ **Type safety**: Validation at each step
- ✅ **Maintainability**: Modular components are easier to update
- ✅ **Grid search support**: Compatible with sklearn's hyperparameter tuning

## Quick Migration Examples

### Example 1: Basic Model Training

#### Old Code (model_screentime_time_windows.py)
```python
from src.data_processing.merge_passive_data_and_labels import (
    merge_daily_screentime_features_with_risk_labels
)

# Get data
screentime_data = merge_daily_screentime_features_with_risk_labels(
    label_column='suicide_risk_label',
    screentime_window_hours=6
)

# Prepare features
feature_cols = [col for col in screentime_data.columns 
                if col not in ['app_user_id', 'suicide_risk_label', ...]]
X = screentime_data[feature_cols]
y = screentime_data['suicide_risk_label']

# Train model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
from sklearn.metrics import accuracy_score, f1_score
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label='at_risk')
```

#### New Code (Pipeline)
```python
from src.pipeline.model_pipeline import ScreentimeModelPipeline

# Create and run pipeline
pipeline = ScreentimeModelPipeline(
    target_type='suicide_risk',
    time_windows=[6],
    use_loocv=False
)

results = pipeline.fit_predict()

# Results automatically include accuracy, F1, confusion matrix, etc.
acc = results[6]['models']['logistic_regression']['accuracy']
f1 = results[6]['models']['logistic_regression']['f1_score']
```

### Example 2: With Label Propagation

#### Old Code
```python
from src.data_processing.merge_passive_data_and_labels import (
    merge_daily_screentime_features_with_risk_labels,
    propagate_positive_labels
)

screentime_data = merge_daily_screentime_features_with_risk_labels(
    label_column='suicide_risk_label',
    screentime_window_hours=6
)

# Apply label propagation
screentime_data = propagate_positive_labels(
    screentime_data,
    'suicide_risk_label',
    'at_risk'
)

# ... rest of training code
```

#### New Code
```python
from src.pipeline.model_pipeline import ScreentimeModelPipeline

pipeline = ScreentimeModelPipeline(
    target_type='suicide_risk',
    time_windows=[6],
    propagate_labels=True  # Just add this flag
)

results = pipeline.fit_predict()
```

### Example 3: Multiple Time Windows

#### Old Code
```python
time_windows = [3, 6, 9, 12]
all_results = {}

for window in time_windows:
    # Get data for this window
    data = merge_daily_screentime_features_with_risk_labels(
        label_column='suicide_risk_label',
        screentime_window_hours=window
    )
    
    # Train models
    # ... training code ...
    
    all_results[window] = {
        'accuracy': acc,
        'f1_score': f1
    }
```

#### New Code
```python
from src.pipeline.model_pipeline import ScreentimeModelPipeline

pipeline = ScreentimeModelPipeline(
    target_type='suicide_risk',
    time_windows=[3, 6, 9, 12]  # Just list all windows
)

results = pipeline.fit_predict()  # Automatically trains all windows
```

### Example 4: Leave-One-User-Out Cross-Validation

#### Old Code
```python
from sklearn.model_selection import LeaveOneGroupOut

logo = LeaveOneGroupOut()
groups = data['app_user_id'].values

all_y_test = []
all_y_pred = []

for train_idx, test_idx in logo.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Check if both classes present
    if y_train.nunique() < 2:
        continue
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    all_y_test.extend(y_test)
    all_y_pred.extend(y_pred)

# Calculate metrics
acc = accuracy_score(all_y_test, all_y_pred)
f1 = f1_score(all_y_test, all_y_pred, pos_label='at_risk')
```

#### New Code
```python
from src.pipeline.model_pipeline import ScreentimeModelPipeline

pipeline = ScreentimeModelPipeline(
    target_type='suicide_risk',
    time_windows=[6],
    use_loocv=True  # Just add this flag
)

results = pipeline.fit_predict()  # LOOCV handled automatically
```

### Example 5: Balanced Class Weights

#### Old Code
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

lr_model = LogisticRegression(class_weight='balanced')
rf_model = RandomForestClassifier(class_weight='balanced')

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
```

#### New Code
```python
from src.pipeline.model_pipeline import ScreentimeModelPipeline

pipeline = ScreentimeModelPipeline(
    target_type='suicide_risk',
    time_windows=[6],
    balanced_class_weight=True  # Just add this flag
)

results = pipeline.fit_predict()
```

### Example 6: All Features Combined

#### Old Code
```python
time_windows = [3, 6, 9, 12]
propagate = True
balanced = True
use_cv = True

for window in time_windows:
    data = merge_daily_screentime_features_with_risk_labels(
        label_column='suicide_risk_label',
        screentime_window_hours=window
    )
    
    if propagate:
        data = propagate_positive_labels(data, 'suicide_risk_label', 'at_risk')
    
    # ... LOOCV setup ...
    # ... balanced class weights ...
    # ... training and evaluation ...
```

#### New Code
```python
from src.pipeline.model_pipeline import ScreentimeModelPipeline

pipeline = ScreentimeModelPipeline(
    target_type='suicide_risk',
    time_windows=[3, 6, 9, 12],
    propagate_labels=True,
    balanced_class_weight=True,
    use_loocv=True
)

results = pipeline.fit_predict()  # Everything handled automatically
```

## Mapping: Old Functions → Pipeline Components

### Data Extraction
| Old Code | Pipeline Component |
|----------|-------------------|
| `service.extract_from_database("screentime")` | `DataExtractor(source='database', data_types=['screentime'])` |
| `pd.read_csv('data.csv')` | `DataExtractor(source='csv')` |

### Data Processing
| Old Code | Pipeline Component |
|----------|-------------------|
| `hourly_screentime_data(fill_method='zero')` | `ScreentimeProcessor(fill_method='zero')` |
| `weekly_avg_steps()` | `HealthDataProcessor(aggregation='weekly', metrics=['steps'])` |
| `_process_passive_data_dataframe(...)` | Called internally by processors |

### Label Extraction
| Old Code | Pipeline Component |
|----------|-------------------|
| `get_phq9_dataframe()` | `LabelExtractor(target_type='phq9')` |
| `get_suicide_risk_dataframe()` | `LabelExtractor(target_type='suicide_risk')` |
| `get_daily_labels_dataframe()` | `LabelExtractor(target_type='self_harm'/'sleep')` |

### Feature-Label Merging
| Old Code | Pipeline Component |
|----------|-------------------|
| `merge_daily_screentime_features_with_phq9(...)` | `FeatureLabelMerger(target_type='phq9', ...)` |
| `merge_daily_screentime_features_with_risk_labels(...)` | `FeatureLabelMerger(target_type='suicide_risk', ...)` |
| `propagate_positive_labels(...)` | `FeatureLabelMerger(..., propagate_labels=True)` |

### Modeling
| Old Code | Pipeline Component |
|----------|-------------------|
| Manual train/test split | `ScreentimeModelPipeline(use_loocv=False)` |
| Manual LOOCV | `ScreentimeModelPipeline(use_loocv=True)` |
| Manual class balancing | `ScreentimeModelPipeline(balanced_class_weight=True)` |

## Gradual Migration Strategy

You don't have to migrate everything at once. Here's a gradual approach:

### Phase 1: Run Pipeline Alongside Old Code
```python
# Keep running old code
old_results = train_and_evaluate_models(data, 6, 'suicide_risk', ...)

# Also run pipeline to compare
from src.pipeline.model_pipeline import ScreentimeModelPipeline
pipeline = ScreentimeModelPipeline(target_type='suicide_risk', time_windows=[6])
new_results = pipeline.fit_predict()

# Compare results to ensure consistency
```

### Phase 2: Use Pipeline for New Experiments
- Continue using old code for established workflows
- Use pipeline for any new experiments or analyses
- Gradually build confidence in the pipeline

### Phase 3: Full Migration
- Once comfortable, replace old code with pipeline calls
- Update scripts and notebooks
- Archive old implementation files

## Using Custom Transformers

You can create custom transformers for project-specific logic:

```python
from sklearn.base import BaseEstimator, TransformerMixin

class MyCustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, param1=None):
        self.param1 = param1
    
    def fit(self, X, y=None):
        # Learn from data if needed
        return self
    
    def transform(self, X):
        # Apply transformation
        return transformed_X

# Use in pipeline
from sklearn.pipeline import Pipeline

my_pipeline = Pipeline([
    ('extract', DataExtractor()),
    ('custom', MyCustomTransformer(param1='value')),
    ('process', ScreentimeProcessor())
])
```

## Data Processing Only (No Modeling)

If you only need data processing:

```python
from src.pipeline.mental_health_pipeline import create_screentime_risk_pipeline

# Create data processing pipeline
pipeline = create_screentime_risk_pipeline(
    target_type='suicide_risk',
    time_windows=[6, 12],
    fill_method='zero'
)

# Process data
pipeline.fit(None)
processed_data = pipeline.transform(None)

# Now use processed_data for custom analysis
for window, data in processed_data.items():
    print(f"Window {window}h: {len(data)} samples")
    # Your custom code here...
```

## Command-Line Usage

You can still use command-line arguments by creating a wrapper script:

```python
# run_pipeline.py
import argparse
from src.pipeline.model_pipeline import ScreentimeModelPipeline

parser = argparse.ArgumentParser()
parser.add_argument('target_type', choices=['phq9', 'suicide_risk', 'self_harm', 'sleep'])
parser.add_argument('--windows', nargs='+', type=int, default=[3, 6, 9, 12])
parser.add_argument('--propagate', action='store_true')
parser.add_argument('--balanced', action='store_true')
parser.add_argument('--loocv', action='store_true')

args = parser.parse_args()

pipeline = ScreentimeModelPipeline(
    target_type=args.target_type,
    time_windows=args.windows,
    propagate_labels=args.propagate,
    balanced_class_weight=args.balanced,
    use_loocv=args.loocv
)

results = pipeline.fit_predict()
```

Usage:
```bash
python run_pipeline.py suicide_risk --windows 3 6 9 --balanced --loocv
```

## Troubleshooting

### Issue: "Module not found"
**Solution**: Ensure `src/pipeline` is in your Python path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Issue: Different results than old code
**Solution**: Check:
- Same random seed (`random_state=42`)
- Same data filtering parameters
- Same preprocessing steps
- Same train/test split strategy

### Issue: Pipeline is slow
**Solution**: 
- Use fewer time windows for testing
- Set `save_results=False` during development
- Consider caching intermediate results

## Getting Help

If you encounter issues:
1. Check the README in `src/pipeline/`
2. Look at examples in `example_pipeline_usage.py`
3. Review the docstrings in the source files
4. Compare with old implementation in `model_screentime_time_windows.py`

## Benefits Recap

After migration, your code will be:
- **30-50% shorter**: Less boilerplate
- **More maintainable**: Clear component boundaries
- **Easier to test**: Each component testable independently
- **More flexible**: Easy to swap components or try different configurations
- **More reproducible**: Same transformations guaranteed across runs
- **Compatible with sklearn tools**: Grid search, pipelines, model selection

