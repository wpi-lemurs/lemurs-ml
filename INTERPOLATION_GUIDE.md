# Linear Interpolation for Health Data

## Overview

Linear interpolation has been added to the health data analysis functions to fill in missing (null) values in health metrics. This is particularly useful for machine learning models that require complete data.

## How It Works

Linear interpolation estimates missing values by drawing a straight line between the known values before and after the gap. For example:

- Day 1: 100 steps (known)
- Day 2: ??? steps (missing)
- Day 3: 200 steps (known)

With interpolation, Day 2 would be estimated as **150 steps** (the midpoint).

### Per-User Interpolation

**Important:** Interpolation is performed separately for each user to prevent data leakage across different users. This ensures that User A's health patterns don't influence the interpolated values for User B.

## Functions with Interpolation Support

### 1. `daily_health_with_week(..., interpolate=False)`

**Parameters:**
- `interpolate` (bool): Set to `True` to apply linear interpolation to daily health metrics

**Example:**
```python
from src.health_data_analysis import daily_health_with_week

# Without interpolation (default)
daily_data = daily_health_with_week(interpolate=False)

# With interpolation
daily_data_filled = daily_health_with_week(interpolate=True)
```

**What gets interpolated:**
- `daily_steps`
- `daily_distance`
- `daily_calories`
- `daily_avg_speed`

### 2. `hourly_health_data(..., interpolate=False)`

**Parameters:**
- `interpolate` (bool): Set to `True` to apply linear interpolation to hourly health metrics

**Example:**
```python
from src.health_data_analysis import hourly_health_data

# Without interpolation (default)
hourly_data = hourly_health_data(interpolate=False)

# With interpolation
hourly_data_filled = hourly_health_data(interpolate=True)
```

**What gets interpolated:**
- `hourly_steps`
- `hourly_distance`
- `hourly_calories`
- `hourly_avg_speed`

### 3. `merge_daily_health_with_phq9(..., interpolate=False)`

**Parameters:**
- `interpolate` (bool): Set to `True` to apply linear interpolation before merging with PHQ-9 data

**Example:**
```python
from src.merge_weekly_health_with_phq9 import merge_daily_health_with_phq9

# With interpolation for modeling
modeling_data = merge_daily_health_with_phq9(interpolate=True)
```

### 4. `merge_hourly_health_with_phq9(..., interpolate=False)`

**Parameters:**
- `interpolate` (bool): Set to `True` to apply linear interpolation before merging with PHQ-9 data

**Example:**
```python
from src.merge_weekly_health_with_phq9 import merge_hourly_health_with_phq9

# With interpolation for modeling
modeling_data = merge_hourly_health_with_phq9(interpolate=True)
```

## Test Results

Based on the test output, interpolation significantly reduces null values:

### Daily Data
- **Without interpolation:** 6 null values across health columns
- **With interpolation:** Reduced to 1-3 null values (depending on the metric)

### Hourly Data
- **Without interpolation:** 18 null values across health columns
- **With interpolation:** Reduced to 1 null value

### Daily + PHQ-9 Merged Data
- **Without interpolation:** 4 total null values in health columns
- **With interpolation:** Reduced to 1 null value

### Hourly + PHQ-9 Merged Data
- **Without interpolation:** 10 total null values in health columns
- **With interpolation:** Reduced to 3 null values

## When to Use Interpolation

### ✅ Use Interpolation When:
- Training machine learning models that require complete data
- The missing data points are surrounded by valid measurements
- You want to maintain temporal continuity in the data
- Missing values represent short gaps (1-2 time periods)

### ❌ Avoid Interpolation When:
- You need to preserve the exact nature of data gaps for analysis
- There are long sequences of missing data (e.g., user didn't wear device for a week)
- You're performing exploratory data analysis to understand missingness patterns
- Your model can handle missing values natively (e.g., some tree-based models)

## Implementation Details

The interpolation is implemented using pandas' `interpolate()` method with the following settings:

```python
df[column].interpolate(method='linear', limit_direction='both')
```

- **method='linear'**: Uses linear interpolation (straight line between points)
- **limit_direction='both'**: Interpolates in both forward and backward directions

## Example Usage in ML Pipeline

```python
from src.merge_weekly_health_with_phq9 import merge_daily_health_with_phq9

# Generate complete training data with interpolation
training_data = merge_daily_health_with_phq9(
    interpolate=True,
    week_anchor='MON'
)

# Extract features and target
X = training_data[['daily_steps', 'daily_distance', 'daily_calories', 'daily_avg_speed']]
y = training_data['severity_label']

# Train your model
# model.fit(X, y)
```

## Notes

1. **Interpolation happens per user**: Each user's data is interpolated independently
2. **Preserves original data**: The functions create copies, so original data is not modified
3. **Optional parameter**: Defaults to `False` to maintain backward compatibility
4. **Can be combined with other parameters**: Works with `app_user_id` filtering and `week_anchor` settings

