import matplotlib.pyplot as plt
from src.health_data_analysis import daily_health_with_week, hourly_health_data


def visualize_steps_with_null_handling(user_id=None, time_unit='D'):
    """
    Visualize steps data for a given user with three different null handling approaches:
    1. Raw data (with null values as gaps)
    2. Linear interpolation
    3. Forward/backward filling

    Parameters:
    - user_id: specific user ID to visualize. If None, will use the first user found.
    - time_unit: 'D' for daily, 'H' for hourly aggregation
    """

    # Get data using the appropriate function based on time_unit
    if time_unit == 'D':
        # Get daily data with three different null handling methods
        raw_df = daily_health_with_week(app_user_id=user_id if user_id else -1, null_method=None)
        interpolated_df = daily_health_with_week(app_user_id=user_id if user_id else -1, null_method='linear')
        filled_df = daily_health_with_week(app_user_id=user_id if user_id else -1, null_method='fill')

        time_col = 'date'
        value_col = 'daily_steps'
        time_label = 'Date'
        title_suffix = 'Daily'
    elif time_unit == 'H':
        # Get hourly data with three different null handling methods
        raw_df = hourly_health_data(app_user_id=user_id if user_id else -1, null_method=None)
        interpolated_df = hourly_health_data(app_user_id=user_id if user_id else -1, null_method='linear')
        filled_df = hourly_health_data(app_user_id=user_id if user_id else -1, null_method='fill')

        time_col = 'datetime'
        value_col = 'hourly_steps'
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

    print(f"Using column '{value_col}' for steps data")

    # Extract time series data
    raw_data = raw_df.set_index(time_col)[value_col]
    interpolated_data = interpolated_df.set_index(time_col)[value_col]
    filled_data = filled_df.set_index(time_col)[value_col]

    # Count null values in raw data
    null_count = raw_data.isna().sum()
    total_count = len(raw_data)
    null_percent = (null_count / total_count) * 100 if total_count > 0 else 0

    # Create the visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f'Steps Data Visualization for User {user_id} - {title_suffix}', fontsize=16, fontweight='bold')

    # Plot 1: Raw data with null values
    axes[0].plot(raw_data.index, raw_data.values, marker='o', linestyle='-', linewidth=1.5, markersize=3, label='Steps (with nulls)')
    axes[0].set_title('1. Raw Data (Null Values as Gaps)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel(time_label)
    axes[0].set_ylabel('Steps')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].text(0.02, 0.98, f'Null values: {null_count}/{total_count} ({null_percent:.1f}%)',
                 transform=axes[0].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Linear interpolation
    axes[1].plot(interpolated_data.index, interpolated_data.values, marker='o', linestyle='-',
                linewidth=1.5, markersize=3, color='green', label='Steps (linear interpolation)')
    # Highlight interpolated points
    interpolated_mask = raw_data.isna()
    if interpolated_mask.any():
        axes[1].scatter(interpolated_data.index[interpolated_mask],
                       interpolated_data.values[interpolated_mask],
                       color='red', s=50, zorder=5, label='Interpolated values', alpha=0.7)
    axes[1].set_title('2. Linear Interpolation for Null Values', fontsize=12, fontweight='bold')
    axes[1].set_xlabel(time_label)
    axes[1].set_ylabel('Steps')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].text(0.02, 0.98, f'Interpolated values: {interpolated_mask.sum()}',
                 transform=axes[1].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Plot 3: Forward/backward filling
    axes[2].plot(filled_data.index, filled_data.values, marker='o', linestyle='-',
                linewidth=1.5, markersize=3, color='orange', label='Steps (forward/backward fill)')
    # Highlight filled points
    filled_mask = raw_data.isna()
    if filled_mask.any():
        axes[2].scatter(filled_data.index[filled_mask],
                       filled_data.values[filled_mask],
                       color='purple', s=50, zorder=5, label='Filled values', alpha=0.7)
    axes[2].set_title('3. Forward/Backward Filling for Null Values', fontsize=12, fontweight='bold')
    axes[2].set_xlabel(time_label)
    axes[2].set_ylabel('Steps')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].text(0.02, 0.98, f'Filled values: {filled_mask.sum()}',
                 transform=axes[2].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"\nTime Range: {raw_data.index.min()} to {raw_data.index.max()}")
    print(f"Total Time Points: {total_count}")
    print(f"Null Values: {null_count} ({null_percent:.1f}%)")
    print(f"\nRaw Data (with nulls):")
    print(f"  Mean: {raw_data.mean():.2f}")
    print(f"  Std: {raw_data.std():.2f}")
    print(f"  Min: {raw_data.min():.2f}")
    print(f"  Max: {raw_data.max():.2f}")
    print(f"\nLinear Interpolation:")
    print(f"  Mean: {interpolated_data.mean():.2f}")
    print(f"  Std: {interpolated_data.std():.2f}")
    print(f"  Min: {interpolated_data.min():.2f}")
    print(f"  Max: {interpolated_data.max():.2f}")
    print(f"\nForward/Backward Fill:")
    print(f"  Mean: {filled_data.mean():.2f}")
    print(f"  Std: {filled_data.std():.2f}")
    print(f"  Min: {filled_data.min():.2f}")
    print(f"  Max: {filled_data.max():.2f}")
    print("="*60)


if __name__ == "__main__":
    # Example usage: visualize daily steps for the first available user
    # You can specify a user_id and time_unit ('D' for daily, 'H' for hourly)
    visualize_steps_with_null_handling(time_unit='H')

