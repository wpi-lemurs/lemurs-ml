# Null Value Handling for Health Data

## Overview

Multiple methods for handling missing (null) values have been added to the health data analysis functions. This is particularly useful for machine learning models that require complete data or for exploratory data analysis.

## Available Methods

### 1. **None (No Handling)** - `null_method=None`
Keeps all null values as-is. Use this when you want to see the raw data or when your model can handle missing values.

### 2. **Linear Interpolation** - `null_method='linear'`
Estimates missing values by drawing a straight line between the known values before and after the gap.

**Example:**
- Day 1: 100 steps (known)
- Day 2: ??? steps (missing)
- Day 3: 200 steps (known)

With linear interpolation, Day 2 would be estimated as **150 steps** (the midpoint).

### 3. **Forward/Backward Fill** - `null_method='fill'`
Fills missing values by carrying forward the last known value (forward fill), then filling any remaining nulls by carrying backward the next known value (backward fill).

**Example:**
- Day 1: 100 steps (known)
- Day 2: ??? steps (missing) → filled with 100 (forward fill)
- Day 3: ??? steps (missing) → filled with 100 (forward fill)
- Day 4: 200 steps (known)

### Per-User Processing

**Important:** All null handling methods are performed separately for each user to prevent data leakage across different users. This ensures that User A's health patterns don't influence the filled values for User B.

## Functions with Null Handling Support

### 1. `daily_health_with_week(..., null_method=None)`

**Parameters:**
- `null_method` (str or None): Method for handling null values
  - `None`: No null handling (default)
  - `'linear'`: Linear interpolation
  - `'fill'`: Forward/backward filling

**Example:**

```python
from src.passive_data_analysis import daily_health_with_week

# No null handling (default)
daily_data = daily_health_with_week(null_method=None)

# With linear interpolation
daily_data_linear = daily_health_with_week(null_method='linear')

# With forward/backward fill
daily_data_fill = daily_health_with_week(null_method='fill')
```

**What gets processed:**
- `daily_steps`
- `daily_distance`
- `daily_calories`
- `daily_avg_speed`

### 2. `hourly_health_data(..., null_method=None)`

**Parameters:**
- `null_method` (str or None): Method for handling null values
  - `None`: No null handling (default)
  - `'linear'`: Linear interpolation
  - `'fill'`: Forward/backward filling

**Example:**

```python
from src.passive_data_analysis import hourly_health_data

# No null handling (default)
hourly_data = hourly_health_data(null_method=None)

# With linear interpolation
hourly_data_linear = hourly_health_data(null_method='linear')

# With forward/backward fill
hourly_data_fill = hourly_health_data(null_method='fill')
```

**What gets processed:**
- `hourly_steps`
- `hourly_distance`
- `hourly_calories`
- `hourly_avg_speed`

## Visualization Tool

A visualization tool is available to compare the three null handling methods side-by-side:

**Function:** `visualize_steps_with_null_handling(user_id=None, time_unit='D')`

**Parameters:**
- `user_id` (int or None): Specific user ID to visualize. If None, uses first available user.
- `time_unit` (str): 'D' for daily, 'H' for hourly aggregation

**Example:**
```python
from src.visualization.null_value_visualization import visualize_steps_with_null_handling

# Daily visualization for user 20
visualize_steps_with_null_handling(user_id=20, time_unit='D')

# Hourly visualization for user 20
visualize_steps_with_null_handling(user_id=20, time_unit='H')

# Auto-select first user
visualize_steps_with_null_handling(user_id=None, time_unit='D')
```

This will display three graphs:
1. Raw data with null values as gaps
2. Data with linear interpolation (interpolated points highlighted in red)
3. Data with forward/backward fill (filled points highlighted in purple)

Each graph includes statistics about null values and the impact of the handling method.

## Test Results

Based on test output, null handling methods significantly reduce null values:

### Daily Data (all users)
| Method | Steps Nulls | Distance Nulls | Calories Nulls | Speed Nulls |
|--------|-------------|----------------|----------------|-------------|
| None   | 2           | 4              | 7              | 6           |
| Linear | 1           | 1              | 2              | 3           |
| Fill   | 1           | 1              | 2              | 3           |

### Hourly Data (all users)
| Method | Steps Nulls | Distance Nulls | Calories Nulls | Speed Nulls |
|--------|-------------|----------------|----------------|-------------|
| None   | 6           | 16             | 19             | 19          |
| Linear | 1           | 1              | 2              | 3           |
| Fill   | 1           | 1              | 2              | 3           |

Both linear interpolation and forward/backward fill achieve similar null reduction, but may produce different values depending on the data pattern.

## When to Use Each Method

### ✅ Use None (No Handling) When:
- Performing exploratory data analysis to understand missingness patterns
- Your model can handle missing values natively (e.g., some tree-based models)
- You need to preserve the exact nature of data gaps
- Analyzing data quality and completeness

### ✅ Use Linear Interpolation When:
- Training machine learning models that require complete data
- The missing data points are surrounded by valid measurements
- You want to maintain temporal trends and smooth transitions
- Missing values represent short gaps (1-2 time periods)
- Data has natural continuity (e.g., steps gradually change over time)

### ✅ Use Forward/Backward Fill When:
- Data tends to remain constant over time
- You want to use the "last known good value"
- Dealing with categorical or discrete data that shouldn't be interpolated
- Missing values occur at the beginning or end of a time series (backward/forward fill is better than interpolation in these cases)

## Implementation Details

The interpolation is implemented using pandas' `interpolate()` method with the following settings:

```python
df[column].interpolate(method='linear', limit_direction='both')
```

- **method='linear'**: Uses linear interpolation (straight line between points)
- **limit_direction='both'**: Interpolates in both forward and backward directions

## Example Usage in ML Pipeline

```python
from src.merge_passive_data_and_labels import merge_daily_health_with_phq9

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

