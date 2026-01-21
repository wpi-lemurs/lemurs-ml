"""
Passive data completeness analysis
- Loads steps and screentime from DB (via DatabaseService) or CSV fallbacks
- Computes per-user per-day record counts over a 28-day study window
- Identifies the most-complete, median, and least-complete users
- Plots per-day counts for those three users and per-hour averages across users
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.database_service import DatabaseService
from datetime import timedelta

STUDY_DAYS = 28


def load_table_or_csv(service: DatabaseService, table_name: str, csv_fallback: Path = None) -> pd.DataFrame:
    """Try loading a table via DatabaseService; fall back to CSV if DB not available or table missing."""
    df = None
    try:
        df = service.extract_from_database(table_name)
    except Exception:
        df = None

    if (df is None or df.empty) and csv_fallback is not None and csv_fallback.exists():
        df = pd.read_csv(csv_fallback)
    return df


def normalize_timestamp_column(df: pd.DataFrame, possible_cols=None):
    """Find and parse a timestamp column to pd.Timestamp series and return the name used."""
    if df is None or df.empty:
        return None
    possible_cols = possible_cols or ['start_timestamp', 'start_time', 'timestamp', 'created_at', 'start']
    for c in possible_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
            if df[c].notna().any():
                return c
    # try to infer any datetime-like column
    for c in df.columns:
        if df[c].dtype == 'object':
            try:
                parsed = pd.to_datetime(df[c], errors='coerce')
                if parsed.notna().any():
                    df[c] = parsed
                    return c
            except Exception:
                continue
    return None


def per_user_daily_counts(df: pd.DataFrame, timestamp_col: str, app_user_col: str = 'app_user_id', study_start=None):
    """Return DataFrame with rows (app_user_id, day_index) and counts for each day index 1..STUDY_DAYS."""
    if df is None or df.empty:
        return pd.DataFrame(columns=[app_user_col, 'day', 'count'])

    if app_user_col not in df.columns:
        # treat entire dataset as single user
        df['app_user_id'] = 'all'
        app_user_col = 'app_user_id'

    # drop rows without timestamps
    df = df.dropna(subset=[timestamp_col]).copy()
    if df.empty:
        return pd.DataFrame(columns=[app_user_col, 'day', 'count'])

    # determine study_start: earliest timestamp per user or global
    if study_start is None:
        # align to the earliest date across dataset
        min_ts = df[timestamp_col].min().normalize()
        study_start = min_ts
    else:
        study_start = pd.to_datetime(study_start).normalize()

    # compute day index 1..STUDY_DAYS
    df['day'] = (df[timestamp_col].dt.normalize() - study_start).dt.days + 1
    # keep only days within 1..STUDY_DAYS
    df = df[(df['day'] >= 1) & (df['day'] <= STUDY_DAYS)]

    grouped = df.groupby([app_user_col, 'day']).size().reset_index(name='count')

    # ensure all day rows exist for each user (fill missing days with 0)
    users = grouped[app_user_col].unique().tolist()
    all_rows = []
    for u in users:
        user_days = grouped[grouped[app_user_col] == u].set_index('day')['count']
        for d in range(1, STUDY_DAYS + 1):
            cnt = int(user_days.get(d, 0))
            all_rows.append({app_user_col: u, 'day': d, 'count': cnt})
    result = pd.DataFrame(all_rows)
    return result


def select_three_users(daily_counts: pd.DataFrame, app_user_col='app_user_id'):
    """Select user with most total records, median (by total), and least total records."""
    totals = daily_counts.groupby(app_user_col)['count'].sum().reset_index(name='total')
    totals = totals.sort_values('total')
    if totals.empty:
        return []
    least = totals.iloc[0][app_user_col]
    most = totals.iloc[-1][app_user_col]
    median_idx = len(totals) // 2
    median = totals.iloc[median_idx][app_user_col]
    return [most, median, least]


def plot_three_timelines(daily_counts: pd.DataFrame, users, app_user_col='app_user_id', out_path: Path = Path('passive_three_timelines.png'), ylabel='Number of Records'):
    """Plot timelines for three users with record counts displayed on the line."""
    plt.figure(figsize=(16, 8))

    colors = ['#228B22', '#C71585', '#F18F01']  # Forest Green, Dark Pink, Orange
    labels = ['Most Complete', 'Median', 'Least Complete']

    for idx, u in enumerate(users):
        user_df = daily_counts[daily_counts[app_user_col] == u].sort_values('day')

        # Plot line with markers
        plt.plot(user_df['day'], user_df['count'],
                marker='o', markersize=8, linewidth=2,
                color=colors[idx], label=f'{labels[idx]}: {u}')

        # Add count labels above each point - increased vertical offset
        for _, row in user_df.iterrows():
            y_offset = row['count'] * 0.10  # 10% above the point
            plt.text(row['day'], row['count'] + y_offset, str(int(row['count'])),
                    ha='center', va='bottom', fontsize=9,
                    color=colors[idx], fontweight='bold')

    plt.xlabel('Days in Study', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Records', fontsize=12, fontweight='bold')
    plt.title('Daily Record Step Counts for Most/Median/Least Complete Users',
             fontsize=14, fontweight='bold', pad=20)
    plt.legend(title='User Completeness', fontsize=10, title_fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(range(1, 29))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def hourly_average_across_users(df: pd.DataFrame, timestamp_col: str, app_user_col='app_user_id'):
    """Return DataFrame with hour (0..23) and avg_records_per_user and avg_value_per_user (if numeric value present).
    avg_records_per_user: average number of records recorded in that hour across users (per day-hour aggregated).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['hour', 'avg_records_per_user'])

    if app_user_col not in df.columns:
        df['app_user_id'] = 'all'

    df = df.dropna(subset=[timestamp_col]).copy()
    df['hour'] = df[timestamp_col].dt.hour
    df['date'] = df[timestamp_col].dt.date

    # per user-day-hour counts
    per_user_hour = df.groupby(['app_user_id', 'date', 'hour']).size().reset_index(name='count')

    # average across days per user -> mean count per hour per user
    mean_per_user_hour = per_user_hour.groupby(['app_user_id', 'hour'])['count'].mean().reset_index()

    # now average across users
    avg_across_users = mean_per_user_hour.groupby('hour')['count'].mean().reset_index(name='avg_records_per_user')

    # optionally compute avg steps/value if a numeric column exists
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ('hour',)]
    if numeric_cols:
        val_col = numeric_cols[0]
        per_user_hour_val = df.groupby(['app_user_id', 'date', 'hour'])[val_col].mean().reset_index(name='val')
        mean_per_user_hour_val = per_user_hour_val.groupby(['app_user_id', 'hour'])['val'].mean().reset_index()
        avg_val_across_users = mean_per_user_hour_val.groupby('hour')['val'].mean().reset_index(name='avg_value')
        avg_across_users = avg_across_users.merge(avg_val_across_users, on='hour', how='left')

    # ensure hours 0..23 present
    hours = pd.DataFrame({'hour': list(range(24))})
    avg_across_users = hours.merge(avg_across_users, on='hour', how='left').fillna(0)
    return avg_across_users


def hourly_average_for_users(df: pd.DataFrame, users: list, timestamp_col: str, app_user_col='app_user_id'):
    """Return DataFrame with hour (0..23), user, avg_records and avg_steps for specified users.
    Returns one row per (user, hour) combination.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['user', 'hour', 'avg_records', 'avg_steps'])

    # Filter to specified users
    df = df[df[app_user_col].isin(users)].copy()

    if df.empty:
        return pd.DataFrame(columns=['user', 'hour', 'avg_records', 'avg_steps'])

    df = df.dropna(subset=[timestamp_col]).copy()
    df['hour'] = df[timestamp_col].dt.hour
    df['date'] = df[timestamp_col].dt.date

    # Find the numeric column for steps (could be 'step_count', 'steps', 'value', etc.)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ('hour', 'app_user_id')]
    step_col = numeric_cols[0] if numeric_cols else None

    results = []

    for user in users:
        user_df = df[df[app_user_col] == user]

        # Per day-hour: count records and average steps
        per_day_hour = user_df.groupby(['date', 'hour']).agg({
            timestamp_col: 'count',  # count records
            step_col: 'mean' if step_col else lambda x: 0  # average steps
        }).reset_index()
        per_day_hour.columns = ['date', 'hour', 'record_count', 'avg_steps']

        # Average across all days for this user
        per_hour = per_day_hour.groupby('hour').agg({
            'record_count': 'mean',
            'avg_steps': 'mean'
        }).reset_index()

        per_hour['user'] = user
        results.append(per_hour)

    if not results:
        return pd.DataFrame(columns=['user', 'hour', 'avg_records', 'avg_steps'])

    combined = pd.concat(results, ignore_index=True)
    combined.columns = ['hour', 'avg_records', 'avg_steps', 'user']

    # Ensure all hours 0..23 present for each user
    all_rows = []
    for user in users:
        for hour in range(24):
            user_hour = combined[(combined['user'] == user) & (combined['hour'] == hour)]
            if user_hour.empty:
                all_rows.append({'user': user, 'hour': hour, 'avg_records': 0, 'avg_steps': 0})
            else:
                all_rows.append(user_hour.iloc[0].to_dict())

    return pd.DataFrame(all_rows)


def plot_hourly_average(avg_df: pd.DataFrame, out_path: Path = Path('passive_hourly_average.png')):
    """Plot hourly averages with values displayed on bars."""
    plt.figure(figsize=(16, 8))

    # Create bar chart
    bars = plt.bar(avg_df['hour'], avg_df['avg_records_per_user'],
                   width=0.8, color='#2E86AB', alpha=0.7, edgecolor='black')

    # Add value labels on top of each bar
    for idx, (hour, val) in enumerate(zip(avg_df['hour'], avg_df['avg_records_per_user'])):
        if val > 0:  # Only show non-zero values
            plt.text(hour, val, f'{val:.1f}',
                    ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='#2E86AB')

    plt.xlabel('Hour of Day (24-hour format)', fontsize=12, fontweight='bold')
    plt.ylabel('Average Records per User', fontsize=12, fontweight='bold')
    plt.title('Average Number of Recordings per Hour Across All Users',
             fontsize=14, fontweight='bold', pad=20)
    plt.xticks(range(0, 24), [f'{h:02d}:00' for h in range(24)], rotation=45)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def plot_hourly_records_by_user(hourly_df: pd.DataFrame, users: list, out_path: Path = Path('passive_hourly_records_by_user.png')):
    """Plot hourly record counts for three users."""
    fig, ax = plt.subplots(figsize=(18, 8))

    colors = ['#228B22', '#C71585', '#F18F01']  # Forest Green, Dark Pink, Orange
    labels = ['Most Complete', 'Median', 'Least Complete']

    # Plot lines for each user
    for idx, user in enumerate(users):
        user_df = hourly_df[hourly_df['user'] == user].sort_values('hour')

        # Plot record counts
        ax.plot(user_df['hour'], user_df['avg_records'],
                marker='o', markersize=8, linewidth=2.5,
                color=colors[idx], label=f'{labels[idx]}: {user}',
                alpha=0.8, linestyle='-')

        # Add text labels for each hour - increased vertical offset
        for _, row in user_df.iterrows():
            if row['avg_records'] > 0:
                base_offset = 0.12  # Base 12% offset
                stagger = 0.08 * (idx - 1)  # Additional stagger for each user
                y_offset = base_offset + stagger
                ax.text(row['hour'], row['avg_records'] * (1 + y_offset),
                       f"{row['avg_records']:.1f}",
                       ha='center', va='bottom', fontsize=8,
                       color=colors[idx], fontweight='bold', alpha=0.7)

    # Configure axes
    ax.set_xlabel('Hour of Day', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Records per Hour', fontsize=12, fontweight='bold')

    plt.title('Hourly Record Counts by User Completeness Level',
             fontsize=14, fontweight='bold', pad=20)

    # Set x-axis ticks
    ax.set_xticks(range(0, 24))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right')

    ax.grid(axis='both', alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path

def plot_hourly_steps_by_user(hourly_df: pd.DataFrame, users: list, out_path: Path = Path('passive_hourly_steps_by_user.png')):
    """Plot hourly average steps for three users."""
    fig, ax = plt.subplots(figsize=(18, 8))

    colors = ['#228B22', '#C71585', '#F18F01']  # Forest Green, Dark Pink, Orange
    labels = ['Most Complete', 'Median', 'Least Complete']

    # Plot lines for each user
    for idx, user in enumerate(users):
        user_df = hourly_df[hourly_df['user'] == user].sort_values('hour')

        # Plot average steps
        ax.plot(user_df['hour'], user_df['avg_steps'],
                marker='s', markersize=8, linewidth=2.5,
                color=colors[idx], label=f'{labels[idx]}: {user}',
                alpha=0.8, linestyle='-')

        # Add text labels for each hour - increased vertical offset
        for _, row in user_df.iterrows():
            if row['avg_steps'] > 1:
                base_offset = 0.12  # Base 12% offset
                stagger = 0.08 * (idx - 1)  # Additional stagger for each user
                y_offset = base_offset + stagger
                ax.text(row['hour'], row['avg_steps'] * (1 + y_offset),
                       f"{row['avg_steps']:.0f}",
                       ha='center', va='bottom', fontsize=8,
                       color=colors[idx], fontweight='bold', alpha=0.7)

    # Configure axes
    ax.set_xlabel('Hour of Day', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Steps per Hour', fontsize=12, fontweight='bold')

    plt.title('Hourly Average Steps by User Completeness Level',
             fontsize=14, fontweight='bold', pad=20)

    # Set x-axis ticks
    ax.set_xticks(range(0, 24))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right')

    ax.grid(axis='both', alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def run_analysis(output_dir: Path = Path('analysis_outputs')):
    output_dir.mkdir(parents=True, exist_ok=True)
    service = DatabaseService()

    # try to load steps data
    steps_csv_candidates = [
        Path(__file__).parents[2] / 'data' / 'synthetic' / 'synthetic_step_data.csv',
        Path(__file__).parents[1] / 'data' / 'synthetic' / 'synthetic_step_data.csv',
    ]
    steps = None
    for p in steps_csv_candidates:
        if p.exists():
            steps = load_table_or_csv(service, 'step', csv_fallback=p)
            break

    results = {}

    # Process steps data separately
    if steps is not None and not steps.empty:
        print("\n" + "="*70)
        print("ANALYZING STEPS DATA")
        print("="*70)

        ts_col = normalize_timestamp_column(steps)
        if ts_col is None:
            print('No timestamp column found in steps data.')
        else:
            daily_counts = per_user_daily_counts(steps, ts_col)
            users = select_three_users(daily_counts)
            timeline_path = plot_three_timelines(
                daily_counts,
                users,
                out_path=output_dir / 'steps_three_timelines.png'
            )

            # Updated hourly analysis for the three selected users
            hourly = hourly_average_for_users(steps, users, ts_col)

            # Generate separate plots for records and steps
            hourly_records_path = plot_hourly_records_by_user(
                hourly,
                users,
                out_path=output_dir / 'steps_hourly_records_by_user.png'
            )

            hourly_steps_path = plot_hourly_steps_by_user(
                hourly,
                users,
                out_path=output_dir / 'steps_hourly_steps_by_user.png'
            )

            # save summary dataframe for steps
            summary = daily_counts.groupby('app_user_id')['count'].sum().reset_index(name='total_records')
            summary = summary.sort_values('total_records', ascending=False)
            summary.to_csv(output_dir / 'steps_summary_by_user.csv', index=False)

            print(f'Saved steps timelines to {timeline_path}')
            print(f'Saved steps hourly records plot to {hourly_records_path}')
            print(f'Saved steps hourly steps plot to {hourly_steps_path}')
            print(f'Saved steps summary CSV to {output_dir / "steps_summary_by_user.csv"}')

            results['steps'] = {
                'daily_counts': daily_counts,
                'summary': summary,
                'timeline_path': timeline_path,
                'hourly_records_path': hourly_records_path,
                'hourly_steps_path': hourly_steps_path,
                'hourly_df': hourly
            }
    else:
        print('No steps data found.')

    if not results:
        print('No passive data could be analyzed.')
        return None

    return results

if __name__ == '__main__':
    run_analysis()