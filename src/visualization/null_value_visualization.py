import matplotlib.pyplot as plt
from src.health_data_analysis import daily_health_with_week, hourly_health_data


def visualize_steps_with_null_handling(user_id=None, time_unit='D', metrics_to_plot=None, date_range=None):
    """
    Visualize health data (steps, speed, distance, calories) for a given user with three different null handling approaches:
    1. Raw data (with null values as gaps)
    2. Linear interpolation
    3. Forward/backward filling

    Parameters:
    - user_id: specific user ID to visualize. If None, will use the first user found.
    - time_unit: 'D' for daily, 'H' for hourly aggregation
    - metrics_to_plot: list of metrics to plot. Options: ['steps', 'distance', 'calories', 'speed']
                       If None, plots all available metrics.
    - date_range: tuple of (start_date, end_date) to filter data. Example: ('2025-11-01', '2025-11-30')
                  If None, shows all available data.
    """

    # Get data using the appropriate function based on time_unit
    if time_unit == 'D':
        # Get daily data with three different null handling methods
        raw_df = daily_health_with_week(app_user_id=user_id if user_id else -1, null_method=None, date_range=date_range)
        interpolated_df = daily_health_with_week(app_user_id=user_id if user_id else -1, null_method='linear', date_range=date_range)
        filled_df = daily_health_with_week(app_user_id=user_id if user_id else -1, null_method='fill', date_range=date_range)

        time_col = 'date'
        metrics = {
            'steps': 'daily_steps',
            'distance': 'daily_distance',
            'calories': 'daily_calories',
            'speed': 'daily_avg_speed'
        }
        time_label = 'Date'
        title_suffix = 'Daily'
    elif time_unit == 'H':
        # Get hourly data with three different null handling methods
        raw_df = hourly_health_data(app_user_id=user_id if user_id else -1, null_method=None, date_range=date_range)
        interpolated_df = hourly_health_data(app_user_id=user_id if user_id else -1, null_method='linear', date_range=date_range)
        filled_df = hourly_health_data(app_user_id=user_id if user_id else -1, null_method='fill', date_range=date_range)

        time_col = 'datetime'
        metrics = {
            'steps': 'hourly_steps',
            'distance': 'hourly_distance',
            'calories': 'hourly_calories',
            'speed': 'hourly_avg_speed'
        }
        time_label = 'DateTime'
        title_suffix = 'Hourly'
    else:
        print(f"Unsupported time_unit: {time_unit}. Use 'D' for daily or 'H' for hourly.")
        return

    # Check if we have data
    if raw_df is None or raw_df.empty:
        print(f"No data available for user_id: {user_id}")
        return

    # Select a user if not specified and multiple users are in the data
    if user_id is None and 'app_user_id' in raw_df.columns:
        user_id = raw_df['app_user_id'].iloc[0]
        print(f"No user_id specified. Using user_id: {user_id}")
        # Filter to single user
        raw_df = raw_df[raw_df['app_user_id'] == user_id]
        interpolated_df = interpolated_df[interpolated_df['app_user_id'] == user_id]
        filled_df = filled_df[filled_df['app_user_id'] == user_id]

    if raw_df.empty:
        print(f"No data found for user {user_id}")
        return

    # Filter metrics based on user selection
    if metrics_to_plot is None:
        # Use all available metrics
        metrics_to_plot = list(metrics.keys())
    else:
        # Validate and filter metrics
        valid_metrics = [m for m in metrics_to_plot if m in metrics.keys()]
        if not valid_metrics:
            print(f"Error: None of the requested metrics {metrics_to_plot} are valid.")
            print(f"Valid options are: {list(metrics.keys())}")
            return
        metrics_to_plot = valid_metrics

    # Filter metrics dictionary to only include selected metrics
    metrics = {k: v for k, v in metrics.items() if k in metrics_to_plot}

    # Filter to only metrics that exist in the dataframe
    metrics = {k: v for k, v in metrics.items() if v in raw_df.columns}

    if not metrics:
        print(f"Error: None of the requested metrics are available in the data.")
        return

    print(f"Visualizing health metrics: {', '.join(metrics.keys())}")

    # Define colors for each metric
    colors = {
        'steps': 'blue',
        'distance': 'green',
        'calories': 'red',
        'speed': 'orange'
    }

    # Create the visualization
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle(f'Health Data Visualization for User {user_id} - {title_suffix}', fontsize=16, fontweight='bold')

    # Count total null values across all metrics in raw data
    total_nulls = sum([raw_df[metrics[m]].isna().sum() for m in metrics.keys() if metrics[m] in raw_df.columns])
    total_values = len(raw_df) * len([m for m in metrics.keys() if metrics[m] in raw_df.columns])

    # Plot 1: Raw data with null values
    ax1 = axes[0]

    # Determine if we need a secondary axis for speed
    has_speed = 'speed' in metrics
    has_other_metrics = any(m in metrics for m in ['steps', 'distance', 'calories'])

    if has_speed and has_other_metrics:
        ax1_speed = ax1.twinx()  # Create secondary y-axis for speed
    else:
        ax1_speed = None

    for metric_name, col_name in metrics.items():
        if col_name in raw_df.columns:
            data = raw_df.set_index(time_col)[col_name]
            if metric_name == 'speed' and ax1_speed is not None:
                ax1_speed.plot(data.index, data.values, marker='o', linestyle='-',
                             linewidth=2, markersize=4, color=colors[metric_name],
                             label=f'{metric_name.capitalize()}', alpha=0.7)
            else:
                ax1.plot(data.index, data.values, marker='o', linestyle='-',
                        linewidth=2, markersize=4, color=colors[metric_name],
                        label=f'{metric_name.capitalize()}', alpha=0.7)

    ax1.set_title('1. Raw Data (Null Values as Gaps)', fontsize=12, fontweight='bold')
    ax1.set_xlabel(time_label)

    # Dynamic y-axis labels
    metric_names = [m.capitalize() for m in metrics.keys() if m != 'speed']
    if metric_names:
        ax1.set_ylabel(' / '.join(metric_names), fontsize=10)

    if ax1_speed is not None:
        ax1_speed.set_ylabel('Speed (m/s)', fontsize=10, color=colors['speed'])
        ax1_speed.tick_params(axis='y', labelcolor=colors['speed'])

    ax1.grid(True, alpha=0.3)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    if ax1_speed is not None:
        lines2, labels2 = ax1_speed.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax1.legend(loc='upper left')

    ax1.text(0.02, 0.98, f'Total null values: {total_nulls}/{total_values}',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Linear interpolation
    ax2 = axes[1]

    if has_speed and has_other_metrics:
        ax2_speed = ax2.twinx()  # Create secondary y-axis for speed
    else:
        ax2_speed = None

    for metric_name, col_name in metrics.items():
        if col_name in interpolated_df.columns:
            data = interpolated_df.set_index(time_col)[col_name]
            raw_data_metric = raw_df.set_index(time_col)[col_name]

            if metric_name == 'speed' and ax2_speed is not None:
                ax2_speed.plot(data.index, data.values, marker='o', linestyle='-',
                             linewidth=2, markersize=4, color=colors[metric_name],
                             label=f'{metric_name.capitalize()}', alpha=0.7)
                # Highlight interpolated points
                interpolated_mask = raw_data_metric.isna()
                if interpolated_mask.any():
                    ax2_speed.scatter(data.index[interpolated_mask], data.values[interpolated_mask],
                                    color='red', s=60, zorder=5, marker='x', alpha=0.8, linewidths=2)
            else:
                ax2.plot(data.index, data.values, marker='o', linestyle='-',
                        linewidth=2, markersize=4, color=colors[metric_name],
                        label=f'{metric_name.capitalize()}', alpha=0.7)
                # Highlight interpolated points
                interpolated_mask = raw_data_metric.isna()
                if interpolated_mask.any():
                    ax2.scatter(data.index[interpolated_mask], data.values[interpolated_mask],
                              color='red', s=60, zorder=5, marker='x', alpha=0.8, linewidths=2)

    ax2.set_title('2. Linear Interpolation for Null Values (interpolated points marked with X)', fontsize=12, fontweight='bold')
    ax2.set_xlabel(time_label)

    if metric_names:
        ax2.set_ylabel(' / '.join(metric_names), fontsize=10)

    if ax2_speed is not None:
        ax2_speed.set_ylabel('Speed (m/s)', fontsize=10, color=colors['speed'])
        ax2_speed.tick_params(axis='y', labelcolor=colors['speed'])

    ax2.grid(True, alpha=0.3)

    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    if ax2_speed is not None:
        lines2, labels2 = ax2_speed.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax2.legend(loc='upper left')

    interpolated_nulls = sum([interpolated_df[metrics[m]].isna().sum() for m in metrics.keys() if metrics[m] in interpolated_df.columns])
    ax2.text(0.02, 0.98, f'Remaining null values: {interpolated_nulls}/{total_values}',
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Plot 3: Forward/backward filling
    ax3 = axes[2]

    if has_speed and has_other_metrics:
        ax3_speed = ax3.twinx()  # Create secondary y-axis for speed
    else:
        ax3_speed = None

    for metric_name, col_name in metrics.items():
        if col_name in filled_df.columns:
            data = filled_df.set_index(time_col)[col_name]
            raw_data_metric = raw_df.set_index(time_col)[col_name]

            if metric_name == 'speed' and ax3_speed is not None:
                ax3_speed.plot(data.index, data.values, marker='o', linestyle='-',
                             linewidth=2, markersize=4, color=colors[metric_name],
                             label=f'{metric_name.capitalize()}', alpha=0.7)
                # Highlight filled points
                filled_mask = raw_data_metric.isna()
                if filled_mask.any():
                    ax3_speed.scatter(data.index[filled_mask], data.values[filled_mask],
                                    color='purple', s=60, zorder=5, marker='s', alpha=0.8)
            else:
                ax3.plot(data.index, data.values, marker='o', linestyle='-',
                        linewidth=2, markersize=4, color=colors[metric_name],
                        label=f'{metric_name.capitalize()}', alpha=0.7)
                # Highlight filled points
                filled_mask = raw_data_metric.isna()
                if filled_mask.any():
                    ax3.scatter(data.index[filled_mask], data.values[filled_mask],
                              color='purple', s=60, zorder=5, marker='s', alpha=0.8)

    ax3.set_title('3. Forward/Backward Filling for Null Values (filled points marked with squares)', fontsize=12, fontweight='bold')
    ax3.set_xlabel(time_label)

    if metric_names:
        ax3.set_ylabel(' / '.join(metric_names), fontsize=10)

    if ax3_speed is not None:
        ax3_speed.set_ylabel('Speed (m/s)', fontsize=10, color=colors['speed'])
        ax3_speed.tick_params(axis='y', labelcolor=colors['speed'])

    ax3.grid(True, alpha=0.3)

    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    if ax3_speed is not None:
        lines2, labels2 = ax3_speed.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax3.legend(loc='upper left')

    filled_nulls = sum([filled_df[metrics[m]].isna().sum() for m in metrics.keys() if metrics[m] in filled_df.columns])
    ax3.text(0.02, 0.98, f'Remaining null values: {filled_nulls}/{total_values}',
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"\nTime Range: {raw_df[time_col].min()} to {raw_df[time_col].max()}")
    print(f"Total Time Points: {len(raw_df)}")
    print(f"Total Null Values (all metrics): {total_nulls}/{total_values}")

    # Statistics for each metric
    for metric_name, col_name in metrics.items():
        if col_name not in raw_df.columns:
            continue

        print(f"\n{metric_name.upper()} Statistics:")
        print("-" * 70)

        raw_data = raw_df[col_name]
        interpolated_data = interpolated_df[col_name]
        filled_data = filled_df[col_name]

        null_count = raw_data.isna().sum()
        print(f"  Null values: {null_count}/{len(raw_data)}")

        print(f"\n  Raw Data (with nulls):")
        print(f"    Mean: {raw_data.mean():.2f}")
        print(f"    Std: {raw_data.std():.2f}")
        print(f"    Min: {raw_data.min():.2f}")
        print(f"    Max: {raw_data.max():.2f}")

        print(f"\n  Linear Interpolation:")
        print(f"    Mean: {interpolated_data.mean():.2f}")
        print(f"    Std: {interpolated_data.std():.2f}")
        print(f"    Min: {interpolated_data.min():.2f}")
        print(f"    Max: {interpolated_data.max():.2f}")
        print(f"    Remaining nulls: {interpolated_data.isna().sum()}")

        print(f"\n  Forward/Backward Fill:")
        print(f"    Mean: {filled_data.mean():.2f}")
        print(f"    Std: {filled_data.std():.2f}")
        print(f"    Min: {filled_data.min():.2f}")
        print(f"    Max: {filled_data.max():.2f}")
        print(f"    Remaining nulls: {filled_data.isna().sum()}")

    print("\n" + "="*70)


if __name__ == "__main__":
    # Example usage: visualize daily health metrics for the first available user
    # You can specify a user_id and time_unit ('D' for daily, 'H' for hourly)
    visualize_steps_with_null_handling(user_id=20, time_unit='H', metrics_to_plot=['steps', 'distance'], date_range=['2025-09-22', '2025-09-26'])

