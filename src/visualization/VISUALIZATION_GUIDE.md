# Health Metrics Visualization Guide

## Overview
The `visualize_steps_with_null_handling()` function allows you to visualize health metrics (steps, distance, calories, speed) with three different null handling approaches. You can now **select which metrics to display** on the graphs.

## Function Signature
```python
visualize_steps_with_null_handling(user_id=None, time_unit='D', metrics_to_plot=None)
```

## Parameters

### `user_id` (int or None)
- Specific user ID to visualize
- If `None`, automatically uses the first available user
- Example: `user_id=20`

### `time_unit` (str)
- Time granularity for aggregation
- Options:
  - `'D'` - Daily aggregation (default)
  - `'H'` - Hourly aggregation
- Example: `time_unit='D'`

### `metrics_to_plot` (list or None)
- **NEW!** Select which health metrics to display
- Options: `['steps', 'distance', 'calories', 'speed']`
- If `None`, displays all available metrics (default)
- Examples:
  - `metrics_to_plot=None` - Show all metrics
  - `metrics_to_plot=['steps']` - Show only steps
  - `metrics_to_plot=['steps', 'distance']` - Show steps and distance
  - `metrics_to_plot=['calories', 'speed']` - Show calories and speed

## Usage Examples

### Example 1: Display All Metrics (Default)
```python
from src.visualization.null_value_visualization import visualize_steps_with_null_handling

# Show all metrics for user 20
visualize_steps_with_null_handling(user_id=20, time_unit='D')
```

### Example 2: Display Only Steps
```python
# Show only steps for user 20
visualize_steps_with_null_handling(
    user_id=20, 
    time_unit='D', 
    metrics_to_plot=['steps']
)
```

### Example 3: Display Steps and Distance
```python
# Show steps and distance for user 20
visualize_steps_with_null_handling(
    user_id=20, 
    time_unit='D', 
    metrics_to_plot=['steps', 'distance']
)
```

### Example 4: Display Calories and Speed
```python
# Show calories and speed for user 20
visualize_steps_with_null_handling(
    user_id=20, 
    time_unit='D', 
    metrics_to_plot=['calories', 'speed']
)
```

### Example 5: Display Only Speed
```python
# Show only speed for user 20
visualize_steps_with_null_handling(
    user_id=20, 
    time_unit='D', 
    metrics_to_plot=['speed']
)
```

### Example 6: Hourly Data with Selected Metrics
```python
# Show steps and calories at hourly granularity
visualize_steps_with_null_handling(
    user_id=20, 
    time_unit='H', 
    metrics_to_plot=['steps', 'calories']
)
```

## Visualization Details

### Graph Structure
Each visualization creates **3 vertically stacked graphs**:
1. **Raw Data** - Shows null values as gaps in the lines
2. **Linear Interpolation** - Fills nulls with interpolated values (marked with red X)
3. **Forward/Backward Fill** - Fills nulls with forward/backward filled values (marked with purple squares)

### Color Coding
Each metric has a distinct color:
- **Steps**: Blue
- **Distance**: Green
- **Calories**: Red
- **Speed**: Orange

### Axes
- **Primary Y-axis (left)**: Steps, Distance, Calories
- **Secondary Y-axis (right)**: Speed (only appears when speed is included with other metrics)

### Legend
- Each graph includes a legend showing which metrics are displayed
- Legends are positioned in the upper left corner

### Statistics
After displaying the graphs, the function prints detailed statistics for each metric:
- Time range
- Null value counts
- Mean, Std, Min, Max for each null handling method

## Tips

### When to Use Specific Metrics

**Steps Only**
- Best for focusing on activity patterns
- Easiest to interpret without distractions

**Steps + Distance**
- Compare movement quantity (steps) vs. distance covered
- Useful for understanding stride length patterns

**Steps + Calories**
- Compare activity level with energy expenditure
- Good for fitness/weight management analysis

**Speed Only**
- Focus on movement intensity
- Useful for identifying running vs. walking periods

**All Metrics**
- Comprehensive overview of all health data
- Best for exploratory data analysis
- Can be visually complex with many overlapping lines

### Error Handling
The function validates your metric selection:
- If you request invalid metrics, it will show an error message
- If requested metrics aren't available in the data, it will notify you
- Valid options are always: `['steps', 'distance', 'calories', 'speed']`

## Summary Statistics Output

After the visualization, you'll see statistics like:
```
======================================================================
SUMMARY STATISTICS
======================================================================

Time Range: 2025-09-23 to 2025-11-21
Total Time Points: 60
Total Null Values (all metrics): 13/240

STEPS Statistics:
----------------------------------------------------------------------
  Null values: 2/60

  Raw Data (with nulls):
    Mean: 1250.45
    Std: 523.12
    Min: 0.00
    Max: 3716.00
  ...
```

This helps you understand:
- Data completeness
- Impact of null handling methods
- Distribution of values for each metric

