"""
Passive data completeness analysis
- Loads steps from DB (via DatabaseService) only
- Computes per-user per-day record counts over the full study window
- Identifies the most-complete, median, and least-complete users
- Plots per-day counts for those three users and per-hour averages across users
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.database_service import DatabaseService
from datetime import timedelta

# STUDY_DAYS = 28  # Removed the fixed 28-day constant


def load_table_from_db(service: DatabaseService, table_name: str) -> pd.DataFrame:
    try:
        return service.extract_from_database(table_name)
    except Exception as e:
        print(f"Error loading table '{table_name}': {e}")
        return pd.DataFrame()


def map_to_umass_ids(df: pd.DataFrame, id_mapping: pd.DataFrame, app_user_col='app_user_id') -> pd.DataFrame:
    """Map app_user_id to umass_id using the id_mapping table.
    Returns a new DataFrame with umass_id replacing app_user_id.
    """
    if df is None or df.empty:
        return df

    if id_mapping is None or id_mapping.empty:
        print("Warning: No ID mapping table provided, keeping app_user_id")
        return df

    # Ensure the mapping table has the required columns
    if 'app_user_id' not in id_mapping.columns or 'umass_id' not in id_mapping.columns:
        print("Warning: ID mapping table missing required columns, keeping app_user_id")
        return df

    print(f"Before mapping - unique {app_user_col}: {sorted(df[app_user_col].unique().tolist())}")

    # Create a simple mapping dictionary
    id_map_dict = id_mapping.set_index('app_user_id')['umass_id'].to_dict()

    # Replace app_user_id with umass_id using the mapping
    df[app_user_col] = df[app_user_col].map(id_map_dict).fillna(df[app_user_col])

    print(f"After mapping - unique {app_user_col}: {sorted(df[app_user_col].unique().tolist())}")

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
    """Return DataFrame with rows (app_user_id, day_index) and total steps for each day index.
    Also add a 'date' column that maps each day index to the calendar date (study_start + day-1).
    Only includes data from November 15th, 2024 onwards.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[app_user_col, 'day', 'count', 'date'])

    # Ensure the user column exists
    if app_user_col not in df.columns:
        print(f"Warning: '{app_user_col}' not found in columns: {df.columns.tolist()}")
        # Try to find any user-related column
        possible_user_cols = [c for c in df.columns if 'user' in c.lower() or 'id' in c.lower()]
        if possible_user_cols:
            print(f"Using '{possible_user_cols[0]}' as user column")
            app_user_col = possible_user_cols[0]
        else:
            df[app_user_col] = 'all'

    df = df.dropna(subset=[timestamp_col]).copy()
    if df.empty:
        return pd.DataFrame(columns=[app_user_col, 'day', 'count', 'date'])

    # Filter to only include dates from November 15th, 2024 onwards and before December 31, 2025 (first study)
    start_cutoff_date = pd.Timestamp('2025-11-15')
    end_cutoff_date = pd.Timestamp('2025-12-31')
    df = df[(df[timestamp_col] >= start_cutoff_date) & (df[timestamp_col] < end_cutoff_date)]

    if df.empty:
        return pd.DataFrame(columns=[app_user_col, 'day', 'count', 'date'])

    # Set study_start to November 15th, 2024
    if study_start is None:
        study_start = pd.Timestamp('2025-11-15').normalize()
    else:
        study_start = pd.to_datetime(study_start).normalize()

    df['day'] = (df[timestamp_col].dt.normalize() - study_start).dt.days + 1

    # Calculate actual study duration from data instead of using fixed STUDY_DAYS
    max_day = df['day'].max()
    study_days = int(max_day) if max_day > 0 else 28

    df = df[df['day'] >= 1]  # Remove the upper limit check

    # Find step count column
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    step_col = [c for c in numeric_cols if c not in ('day', 'app_user_id')]
    step_col = step_col[0] if step_col else None

    # Sum steps per day instead of counting records
    if step_col:
        grouped = df.groupby([app_user_col, 'day'])[step_col].sum().reset_index(name='count')
    else:
        grouped = df.groupby([app_user_col, 'day']).size().reset_index(name='count')

    # ensure all day rows exist for each user (fill missing days with 0) and attach date
    users = grouped[app_user_col].unique().tolist()
    all_rows = []
    for u in users:
        user_days = grouped[grouped[app_user_col] == u].set_index('day')['count']
        for d in range(1, study_days + 1):
            cnt = int(user_days.get(d, 0))
            date_for_d = (pd.to_datetime(study_start) + pd.Timedelta(days=d-1)).date()
            all_rows.append({app_user_col: u, 'day': d, 'count': cnt, 'date': date_for_d})
    result = pd.DataFrame(all_rows)
    return result


def per_user_daily_record_counts(df: pd.DataFrame, timestamp_col: str, app_user_col: str = 'app_user_id', study_start=None):
    """Return DataFrame with rows (app_user_id, day_index) and count of records for each day index.
    Also add a 'date' column that maps each day index to the calendar date (study_start + day-1).
    Only includes data from November 15th, 2025 onwards and before December 31, 2025.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[app_user_col, 'day', 'count', 'date'])

    if app_user_col not in df.columns:
        df['app_user_id'] = 'all'
        app_user_col = 'app_user_id'

    df = df.dropna(subset=[timestamp_col]).copy()
    if df.empty:
        return pd.DataFrame(columns=[app_user_col, 'day', 'count', 'date'])

    # Filter to only include dates from November 15th, 2025 onwards and before December 31, 2025 (first study)
    start_cutoff_date = pd.Timestamp('2025-11-15')
    end_cutoff_date = pd.Timestamp('2025-12-31')
    df = df[(df[timestamp_col] >= start_cutoff_date) & (df[timestamp_col] < end_cutoff_date)]

    if df.empty:
        return pd.DataFrame(columns=[app_user_col, 'day', 'count', 'date'])

    # Set study_start to November 15th, 2024
    if study_start is None:
        study_start = pd.Timestamp('2025-11-15').normalize()
    else:
        study_start = pd.to_datetime(study_start).normalize()

    df['day'] = (df[timestamp_col].dt.normalize() - study_start).dt.days + 1

    # Calculate actual study duration from data
    max_day = df['day'].max()
    study_days = int(max_day) if max_day > 0 else 28

    df = df[df['day'] >= 1]  # Remove the upper limit check

    # Count records per day (not sum of values)
    grouped = df.groupby([app_user_col, 'day']).size().reset_index(name='count')

    # ensure all day rows exist for each user (fill missing days with 0) and attach date
    users = grouped[app_user_col].unique().tolist()
    all_rows = []
    for u in users:
        user_days = grouped[grouped[app_user_col] == u].set_index('day')['count']
        for d in range(1, study_days + 1):
            cnt = int(user_days.get(d, 0))
            date_for_d = (pd.to_datetime(study_start) + pd.Timedelta(days=d-1)).date()
            all_rows.append({app_user_col: u, 'day': d, 'count': cnt, 'date': date_for_d})
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
    """Plot timelines for three users with record counts displayed on the line.
    X-axis shows calendar date only (MM-DD format).
    """
    plt.figure(figsize=(20, 8))

    colors = ['#228B22', '#C71585', '#F18F01']  # Forest Green, Dark Pink, Orange
    labels = ['Most Complete', 'Median', 'Least Complete']

    for idx, u in enumerate(users):
        user_df = daily_counts[daily_counts[app_user_col] == u].sort_values('day')

        # Plot line with markers
        plt.plot(user_df['day'], user_df['count'],
                marker='o', markersize=6, linewidth=2,
                color=colors[idx], label=f'{labels[idx]}: {u}')

        # Add count labels above each point - increased vertical offset
        for _, row in user_df.iterrows():
            # Adaptive offset: min absolute + capped relative
            min_offset = 0.5
            rel_offset = min(row['count'] * 0.08, 3.0)
            y = row['count'] + min_offset + rel_offset

            plt.text(
                row['day'], y, str(int(row['count'])),
                ha='center', va='bottom', fontsize=8,
                color=colors[idx], fontweight='bold',
                zorder=5,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75)
            )

    plt.xlabel('Date', fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title('Daily Record Step Counts for Most/Median/Least Complete Users',
             fontsize=14, fontweight='bold', pad=20)

    plt.legend(title='User Completeness', fontsize=10, title_fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')

    # Build xtick labels with dates only (MM-DD format)
    max_day = int(daily_counts['day'].max())
    min_day = int(daily_counts['day'].min())

    # Show every Nth date to avoid overlap (adjust based on range)
    days_range = max_day - min_day + 1
    if days_range > 40:
        step = 5
    elif days_range > 20:
        step = 3
    else:
        step = 2

    days_to_show = list(range(min_day, max_day + 1, step))

    if 'date' in daily_counts.columns:
        date_map = daily_counts.drop_duplicates('day').set_index('day')['date'].to_dict()
        xtick_labels = [pd.to_datetime(date_map.get(d, '')).strftime('%m-%d') if d in date_map else '' for d in days_to_show]
    else:
        xtick_labels = [str(d) for d in days_to_show]

    plt.xticks(days_to_show, xtick_labels, rotation=90, ha='center', fontsize=9)
    plt.xlim(min_day - 1, max_day + 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def hourly_average_across_users(df: pd.DataFrame, timestamp_col: str, app_user_col='app_user_id'):
    """Return DataFrame with hour (0..23) and total_records_per_user and total_value_per_user (if numeric value present).
    Returns totals across all users for each hour.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['hour', 'total_records'])

    if app_user_col not in df.columns:
        df['app_user_id'] = 'all'

    df = df.dropna(subset=[timestamp_col]).copy()
    df['hour'] = df[timestamp_col].dt.hour

    # Count total records per hour
    hourly_counts = df.groupby('hour').size().reset_index(name='total_records')

    # Sum steps/value if numeric column exists
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ('hour',)]
    if numeric_cols:
        val_col = numeric_cols[0]
        hourly_values = df.groupby('hour')[val_col].sum().reset_index(name='total_value')
        hourly_counts = hourly_counts.merge(hourly_values, on='hour', how='left')

    hours = pd.DataFrame({'hour': list(range(24))})
    hourly_counts = hours.merge(hourly_counts, on='hour', how='left').fillna(0)
    return hourly_counts


def hourly_average_for_users(df: pd.DataFrame, users: list, timestamp_col: str, app_user_col='app_user_id'):
    """Return DataFrame with hour (0..23), user, total_records and total_steps for specified users.
    Returns one row per (user, hour) combination with totals across all days.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['user', 'hour', 'total_records', 'total_steps'])

    df = df[df[app_user_col].isin(users)].copy()

    if df.empty:
        return pd.DataFrame(columns=['user', 'hour', 'total_records', 'total_steps'])

    df = df.dropna(subset=[timestamp_col]).copy()
    df['hour'] = df[timestamp_col].dt.hour

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ('hour', 'app_user_id')]
    step_col = numeric_cols[0] if numeric_cols else None

    results = []

    for user in users:
        user_df = df[df[app_user_col] == user]

        # Sum records and steps per hour (across all days)
        per_hour = user_df.groupby('hour').agg({
            timestamp_col: 'count',  # total count of records
            step_col: 'sum' if step_col else lambda x: 0  # total steps
        }).reset_index()
        per_hour.columns = ['hour', 'total_records', 'total_steps']
        per_hour['user'] = user
        results.append(per_hour)

    if not results:
        return pd.DataFrame(columns=['user', 'hour', 'total_records', 'total_steps'])

    combined = pd.concat(results, ignore_index=True)

    # Ensure all hours 0..23 present for each user
    all_rows = []
    for user in users:
        for hour in range(24):
            user_hour = combined[(combined['user'] == user) & (combined['hour'] == hour)]
            if user_hour.empty:
                all_rows.append({'user': user, 'hour': hour, 'total_records': 0, 'total_steps': 0})
            else:
                all_rows.append(user_hour.iloc[0].to_dict())

    return pd.DataFrame(all_rows)


def plot_hourly_average(avg_df: pd.DataFrame, out_path: Path = Path('passive_hourly_average.png')):
    """Plot hourly totals with values displayed on bars."""
    plt.figure(figsize=(16, 8))

    bars = plt.bar(avg_df['hour'], avg_df['total_records'],
                   width=0.8, color='#2E86AB', alpha=0.7, edgecolor='black')

    for idx, (hour, val) in enumerate(zip(avg_df['hour'], avg_df['total_records'])):
        if val > 0:
            plt.text(hour, val, f'{int(val)}',
                    ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='#2E86AB')

    plt.xlabel('Hour of Day (24-hour format)', fontsize=12, fontweight='bold')
    plt.ylabel('Total Records', fontsize=12, fontweight='bold')
    plt.title('Total Number of Recordings per Hour Across All Users',
             fontsize=14, fontweight='bold', pad=20)
    plt.xticks(range(0, 24), [f'{h:02d}:00' for h in range(24)], rotation=45)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def plot_hourly_records_by_user(hourly_df: pd.DataFrame, users: list, out_path: Path = Path('passive_hourly_records_by_user.png')):
    """Plot hourly total record counts for three users."""
    fig, ax = plt.subplots(figsize=(18, 8))

    colors = ['#228B22', '#C71585', '#F18F01']
    labels = ['Most Complete', 'Median', 'Least Complete']

    for idx, user in enumerate(users):
        user_df = hourly_df[hourly_df['user'] == user].sort_values('hour')

        ax.plot(user_df['hour'], user_df['total_records'],
                marker='o', markersize=8, linewidth=2.5,
                color=colors[idx], label=f'{labels[idx]}: {user}',
                alpha=0.8, linestyle='-')

        for _, row in user_df.iterrows():
            if row['total_records'] > 0:
                min_offset = 0.5
                rel_offset = min(row['total_records'] * 0.08, 3.0)
                stagger = 0.5 * idx
                y = row['total_records'] + min_offset + rel_offset + stagger

                ax.text(
                    row['hour'], y, f"{int(row['total_records'])}",
                    ha='center', va='bottom', fontsize=8,
                    color=colors[idx], fontweight='bold',
                    alpha=0.85, zorder=5,
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.7)
                )

    ax.set_xlabel('Hour of Day', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Records per Hour', fontsize=12, fontweight='bold')

    plt.title('Hourly Total Record Counts by User Completeness Level',
             fontsize=14, fontweight='bold', pad=20)

    ax.set_xticks(range(0, 24))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right')

    ax.grid(axis='both', alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def plot_hourly_steps_by_user(hourly_df: pd.DataFrame, users: list, out_path: Path = Path('passive_hourly_steps_by_user.png')):
    """Plot hourly total steps for three users."""
    fig, ax = plt.subplots(figsize=(18, 8))

    colors = ['#228B22', '#C71585', '#F18F01']
    labels = ['Most Complete', 'Median', 'Least Complete']

    for idx, user in enumerate(users):
        user_df = hourly_df[hourly_df['user'] == user].sort_values('hour')

        ax.plot(user_df['hour'], user_df['total_steps'],
                marker='s', markersize=8, linewidth=2.5,
                color=colors[idx], label=f'{labels[idx]}: {user}',
                alpha=0.8, linestyle='-')

        for _, row in user_df.iterrows():
            if row['total_steps'] > 10:
                min_offset = 10
                rel_offset = min(row['total_steps'] * 0.05, 50)
                stagger = 30 * idx
                y = row['total_steps'] + min_offset + rel_offset + stagger

                ax.text(
                    row['hour'], y, f"{int(row['total_steps'])}",
                    ha='center', va='bottom', fontsize=8,
                    color=colors[idx], fontweight='bold',
                    alpha=0.85, zorder=5,
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.7)
                )

    ax.set_xlabel('Hour of Day', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Steps per Hour', fontsize=12, fontweight='bold')

    plt.title('Hourly Total Steps by User Completeness Level',
             fontsize=14, fontweight='bold', pad=20)

    ax.set_xticks(range(0, 24))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right')

    ax.grid(axis='both', alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def plot_all_users_timeline(daily_counts: pd.DataFrame, app_user_col='app_user_id', out_path: Path = Path('passive_all_users_timeline.png'), ylabel='Total Steps'):
    """Plot timelines for ALL users showing daily step counts.
    X-axis shows calendar date only (MM-DD format).
    """
    plt.figure(figsize=(22, 10))

    all_users = daily_counts[app_user_col].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_users)))

    for idx, user in enumerate(all_users):
        user_df = daily_counts[daily_counts[app_user_col] == user].sort_values('day')

        # Plot line with smaller markers for readability
        plt.plot(user_df['day'], user_df['count'],
                marker='o', markersize=3, linewidth=1.2,
                color=colors[idx], label=f'{user}', alpha=0.7)

        # Add count labels above each point using adaptive offset
        for _, row in user_df.iterrows():
            if row['count'] > 0:
                # Adaptive offset: min absolute + capped relative
                min_offset = 0.5
                rel_offset = min(row['count'] * 0.08, 3.0)
                y = row['count'] + min_offset + rel_offset

                plt.text(
                    row['day'], y, f"{int(row['count'])}",
                    ha='center', va='bottom',
                    fontsize=5,
                    color=colors[idx],
                    fontweight='bold',
                    alpha=0.7,
                    zorder=5,
                    bbox=dict(boxstyle='round,pad=0.1',
                              fc='white', ec='none', alpha=0.6)
                )

    plt.xlabel('Date', fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title('Daily Step Counts for All Participants',
             fontsize=14, fontweight='bold', pad=20)

    # Place legend outside plot area
    plt.legend(title='Participant ID', fontsize=8, title_fontsize=10,
              bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    plt.grid(True, alpha=0.3, linestyle='--')

    # Build xtick labels with dates only - show every Nth date
    max_day = int(daily_counts['day'].max())
    min_day = int(daily_counts['day'].min())

    days_range = max_day - min_day + 1
    if days_range > 40:
        step = 5
    elif days_range > 20:
        step = 3
    else:
        step = 2

    days_to_show = list(range(min_day, max_day + 1, step))

    if 'date' in daily_counts.columns:
        date_map = daily_counts.drop_duplicates('day').set_index('day')['date'].to_dict()
        xtick_labels = [pd.to_datetime(date_map.get(d, '')).strftime('%m-%d') if d in date_map else '' for d in days_to_show]
    else:
        xtick_labels = [str(d) for d in days_to_show]

    plt.xticks(days_to_show, xtick_labels, rotation=90, ha='center', fontsize=8)
    plt.xlim(min_day - 1, max_day + 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def plot_all_users_daily_records(daily_counts: pd.DataFrame, app_user_col='app_user_id', out_path: Path = Path('passive_all_users_daily_records.png')):
    """Plot timelines for ALL users showing daily record counts.
    X-axis shows calendar date only (MM-DD format).
    """
    plt.figure(figsize=(22, 10))

    all_users = daily_counts[app_user_col].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_users)))

    for idx, user in enumerate(all_users):
        user_df = daily_counts[daily_counts[app_user_col] == user].sort_values('day')

        # Plot line with smaller markers for readability
        plt.plot(user_df['day'], user_df['count'],
                marker='o', markersize=3, linewidth=1.2,
                color=colors[idx], label=f'{user}', alpha=0.7)

        # Add count labels above each point using adaptive offset
        for _, row in user_df.iterrows():
            if row['count'] > 0:
                # Adaptive offset: min absolute + capped relative
                min_offset = 0.5
                rel_offset = min(row['count'] * 0.08, 3.0)
                y = row['count'] + min_offset + rel_offset

                plt.text(
                    row['day'], y, f"{int(row['count'])}",
                    ha='center', va='bottom',
                    fontsize=5,
                    color=colors[idx],
                    fontweight='bold',
                    alpha=0.7,
                    zorder=5,
                    bbox=dict(boxstyle='round,pad=0.1',
                              fc='white', ec='none', alpha=0.6)
                )

    plt.xlabel('Date', fontsize=12, fontweight='bold')
    plt.ylabel('Total Records', fontsize=12, fontweight='bold')
    plt.title('Daily Record Counts for All Participants',
             fontsize=14, fontweight='bold', pad=20)

    # Place legend outside plot area
    plt.legend(title='Participant ID', fontsize=8, title_fontsize=10,
              bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    plt.grid(True, alpha=0.3, linestyle='--')

    # Build xtick labels with dates only - show every Nth date
    max_day = int(daily_counts['day'].max())
    min_day = int(daily_counts['day'].min())

    days_range = max_day - min_day + 1
    if days_range > 40:
        step = 5
    elif days_range > 20:
        step = 3
    else:
        step = 2

    days_to_show = list(range(min_day, max_day + 1, step))

    if 'date' in daily_counts.columns:
        date_map = daily_counts.drop_duplicates('day').set_index('day')['date'].to_dict()
        xtick_labels = [pd.to_datetime(date_map.get(d, '')).strftime('%m-%d') if d in date_map else '' for d in days_to_show]
    else:
        xtick_labels = [str(d) for d in days_to_show]

    plt.xticks(days_to_show, xtick_labels, rotation=90, ha='center', fontsize=8)
    plt.xlim(min_day - 1, max_day + 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def plot_all_users_hourly_records(hourly_df: pd.DataFrame, out_path: Path = Path('passive_all_users_hourly_records.png')):
    """Plot hourly total record counts for all users."""
    fig, ax = plt.subplots(figsize=(20, 10))

    all_users = hourly_df['user'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_users)))

    for idx, user in enumerate(all_users):
        user_df = hourly_df[hourly_df['user'] == user].sort_values('hour')

        ax.plot(user_df['hour'], user_df['total_records'],
                marker='o', markersize=4, linewidth=1.2,
                color=colors[idx], label=f'{user}', alpha=0.7)

        # Add count labels above each point
        for _, row in user_df.iterrows():
            if row['total_records'] > 0:
                min_offset = 0.3
                rel_offset = min(row['total_records'] * 0.05, 2.0)
                y = row['total_records'] + min_offset + rel_offset

                ax.text(
                    row['hour'], y, f"{int(row['total_records'])}",
                    ha='center', va='bottom',
                    fontsize=5,
                    color=colors[idx],
                    fontweight='bold',
                    alpha=0.7,
                    zorder=5,
                    bbox=dict(boxstyle='round,pad=0.1',
                              fc='white', ec='none', alpha=0.6)
                )

    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Records per Hour', fontsize=12, fontweight='bold')
    ax.set_title('Hourly Total Record Counts for All Participants',
                 fontsize=14, fontweight='bold', pad=20)

    ax.set_xticks(range(0, 24))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right')

    plt.legend(title='Participant ID', fontsize=7, title_fontsize=9,
               bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    ax.grid(axis='both', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def plot_all_users_hourly_steps(hourly_df: pd.DataFrame, out_path: Path = Path('passive_all_users_hourly_steps.png')):
    """Plot hourly total steps for all users."""
    fig, ax = plt.subplots(figsize=(20, 10))

    all_users = hourly_df['user'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_users)))

    for idx, user in enumerate(all_users):
        user_df = hourly_df[hourly_df['user'] == user].sort_values('hour')

        ax.plot(user_df['hour'], user_df['total_steps'],
                marker='s', markersize=4, linewidth=1.2,
                color=colors[idx], label=f'{user}', alpha=0.7)

        # Add count labels above each point
        for _, row in user_df.iterrows():
            if row['total_steps'] > 10:
                min_offset = 5
                rel_offset = min(row['total_steps'] * 0.03, 30)
                y = row['total_steps'] + min_offset + rel_offset

                ax.text(
                    row['hour'], y, f"{int(row['total_steps'])}",
                    ha='center', va='bottom',
                    fontsize=5,
                    color=colors[idx],
                    fontweight='bold',
                    alpha=0.7,
                    zorder=5,
                    bbox=dict(boxstyle='round,pad=0.1',
                              fc='white', ec='none', alpha=0.6)
                )

    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Steps per Hour', fontsize=12, fontweight='bold')
    ax.set_title('Hourly Total Steps for All Participants',
                 fontsize=14, fontweight='bold', pad=20)

    ax.set_xticks(range(0, 24))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right')

    plt.legend(title='Participant ID', fontsize=7, title_fontsize=9,
               bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    ax.grid(axis='both', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def calculate_completeness_score(daily_counts: pd.DataFrame, app_user_col='app_user_id') -> pd.DataFrame:
    """Calculate data completeness score for each user.
    Returns DataFrame with user, days_with_data, completeness_percentage, and total_records.
    """
    max_day = daily_counts['day'].max()
    study_days = int(max_day) if max_day > 0 else 28

    summary = []
    for user in daily_counts[app_user_col].unique():
        user_df = daily_counts[daily_counts[app_user_col] == user]
        days_with_data = (user_df['count'] > 0).sum()
        completeness_pct = (days_with_data / study_days) * 100
        total_records = user_df['count'].sum()

        summary.append({
            app_user_col: user,
            'days_with_data': days_with_data,
            'completeness_percentage': completeness_pct,
            'total_records': int(total_records)
        })

    return pd.DataFrame(summary).sort_values('completeness_percentage', ascending=False)


def plot_completeness_score(completeness_df: pd.DataFrame, app_user_col='app_user_id',
                            out_path: Path = Path('steps_completeness_score.png')):
    """Plot stacked bar chart showing data completeness for each user."""
    fig, ax = plt.subplots(figsize=(14, 8))

    users = completeness_df[app_user_col].tolist()
    completeness = completeness_df['completeness_percentage'].tolist()
    missing = [100 - c for c in completeness]

    # Create stacked bars
    bars1 = ax.barh(users, completeness, color='#228B22', label='Days with Data', alpha=0.8)
    bars2 = ax.barh(users, missing, left=completeness, color='#DC143C', label='Missing Days', alpha=0.8)

    # Add percentage labels inside bars
    for idx, (user, comp, miss) in enumerate(zip(users, completeness, missing)):
        # Label for days with data
        if comp > 5:
            ax.text(comp / 2, idx, f'{comp:.1f}%',
                    ha='center', va='center', fontweight='bold', color='white', fontsize=10)
        # Label for missing days
        if miss > 5:
            ax.text(comp + miss / 2, idx, f'{miss:.1f}%',
                    ha='center', va='center', fontweight='bold', color='white', fontsize=10)

    ax.set_xlabel('Completeness (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Participant ID', fontsize=12, fontweight='bold')
    ax.set_title('Data Completeness Score by Participant', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 100)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def plot_availability_heatmap(daily_counts: pd.DataFrame, app_user_col='app_user_id',
                              out_path: Path = Path('steps_availability_heatmap.png')):
    """Plot heatmap showing which days each participant has data."""
    # Create pivot table with users as rows and days as columns
    pivot = daily_counts.pivot(index=app_user_col, columns='day', values='count').fillna(0)

    # Convert to binary (has data = 1, no data = 0)
    binary_data = (pivot > 0).astype(int)

    # Calculate study days from data
    study_days = int(daily_counts['day'].max())

    fig, ax = plt.subplots(figsize=(18, max(8, len(binary_data) * 0.5)))

    # Create heatmap with custom colors
    cmap = plt.cm.colors.ListedColormap(['#DC143C', '#228B22'])  # Red for missing, Green for present
    im = ax.imshow(binary_data, cmap=cmap, aspect='auto', interpolation='nearest')

    # Set ticks and labels
    ax.set_xticks(range(study_days))
    ax.set_xticklabels(range(1, study_days + 1))
    ax.set_yticks(range(len(binary_data)))
    ax.set_yticklabels(binary_data.index)

    # Add date labels if available
    if 'date' in daily_counts.columns:
        date_map = daily_counts.drop_duplicates('day').set_index('day')['date'].to_dict()
        date_labels = [pd.to_datetime(date_map.get(d + 1, '')).strftime('%m-%d')
                       if date_map.get(d + 1) else '' for d in range(study_days)]
        ax.set_xticklabels(date_labels, rotation=45, ha='right')

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Participant ID', fontsize=12, fontweight='bold')
    ax.set_title('Daily Data Availability Heatmap', fontsize=14, fontweight='bold', pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['No Data', 'Has Data'])

    # Add grid
    ax.set_xticks([x - 0.5 for x in range(1, study_days)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(binary_data))], minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def plot_daily_participation(daily_counts: pd.DataFrame, app_user_col='app_user_id',
                            out_path: Path = Path('steps_daily_participation.png')):
    """Plot number of participants with data each day."""
    daily_participation = daily_counts[daily_counts['count'] > 0].groupby('day')[app_user_col].nunique()
    total_users = daily_counts[app_user_col].nunique()

    # Calculate study days from data
    study_days = int(daily_counts['day'].max())

    fig, ax = plt.subplots(figsize=(16, 8))

    bars = ax.bar(daily_participation.index, daily_participation.values,
                  color='#2E86AB', alpha=0.7, edgecolor='black')

    # Add count labels only (removed percentage)
    for day, count in zip(daily_participation.index, daily_participation.values):
        ax.text(day, count, f'{count}',
               ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Add total participants annotation in top-right corner
    ax.text(0.98, 0.98, f'Total Participants = {total_users}',
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.8))

    # Add date labels if available
    if 'date' in daily_counts.columns:
        date_map = daily_counts.drop_duplicates('day').set_index('day')['date'].to_dict()
        date_labels = [pd.to_datetime(date_map.get(d, '')).strftime('%m-%d')
                       if date_map.get(d) else '' for d in range(1, study_days + 1)]
        ax.set_xticks(range(1, study_days + 1))
        ax.set_xticklabels(date_labels, rotation=45, ha='right')

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Participants', fontsize=12, fontweight='bold')
    ax.set_title('Daily Participant Data Availability', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 6)  # Set y-axis to go up to 6
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def plot_completeness_distribution(completeness_df: pd.DataFrame,
                                   out_path: Path = Path('steps_completeness_distribution.png')):
    """Plot histogram of completeness scores."""
    fig, ax = plt.subplots(figsize=(12, 7))

    bins = [0, 25, 50, 75, 90, 100]
    counts, edges, patches = ax.hist(completeness_df['completeness_percentage'],
                                     bins=bins, edgecolor='black', alpha=0.7,
                                     color='#2E86AB')

    # Color code bars
    colors = ['#DC143C', '#FF6347', '#FFD700', '#90EE90', '#228B22']
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)

    # Add count labels
    for count, edge in zip(counts, edges[:-1]):
        if count > 0:
            ax.text(edge + (edges[1] - edges[0]) / 2, count,
                   f'{int(count)}', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')

    ax.set_xlabel('Completeness (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Participants', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Data Completeness Scores', fontsize=14, fontweight='bold')
    ax.set_xticks([0, 25, 50, 75, 90, 100])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def per_user_daily_survey_submissions(df: pd.DataFrame, timestamp_col: str, app_user_col: str = 'app_user_id', study_start=None):
    """Return DataFrame with rows (app_user_id, day_index) and survey_submitted flag for each day.
    Only includes data from November 15th, 2025 onwards and before December 31, 2025.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[app_user_col, 'day', 'survey_submitted', 'date'])

    if app_user_col not in df.columns:
        df['app_user_id'] = 'all'
        app_user_col = 'app_user_id'

    df = df.dropna(subset=[timestamp_col]).copy()
    if df.empty:
        return pd.DataFrame(columns=[app_user_col, 'day', 'survey_submitted', 'date'])

    # Filter to only include dates from November 15th, 2025 onwards and before December 31, 2025
    start_cutoff_date = pd.Timestamp('2025-11-15')
    end_cutoff_date = pd.Timestamp('2025-12-31')
    df = df[(df[timestamp_col] >= start_cutoff_date) & (df[timestamp_col] < end_cutoff_date)]

    if df.empty:
        return pd.DataFrame(columns=[app_user_col, 'day', 'survey_submitted', 'date'])

    # Set study_start to November 15th, 2025
    if study_start is None:
        study_start = pd.Timestamp('2025-11-15').normalize()
    else:
        study_start = pd.to_datetime(study_start).normalize()

    df['day'] = (df[timestamp_col].dt.normalize() - study_start).dt.days + 1
    max_day = df['day'].max()
    study_days = int(max_day) if max_day > 0 else 28

    df = df[df['day'] >= 1]

    # Get unique days per user (one submission per day counts as survey submitted)
    grouped = df.groupby([app_user_col, 'day']).size().reset_index(name='count')
    grouped['survey_submitted'] = 1

    # ensure all day rows exist for each user (fill missing days with 0)
    users = grouped[app_user_col].unique().tolist()
    all_rows = []
    for u in users:
        user_days = grouped[grouped[app_user_col] == u].set_index('day')['survey_submitted']
        for d in range(1, study_days + 1):
            submitted = int(user_days.get(d, 0))
            date_for_d = (pd.to_datetime(study_start) + pd.Timedelta(days=d-1)).date()
            all_rows.append({app_user_col: u, 'day': d, 'survey_submitted': submitted, 'date': date_for_d})
    result = pd.DataFrame(all_rows)
    return result


def plot_step_and_survey_heatmap(daily_steps: pd.DataFrame, daily_surveys: pd.DataFrame, app_user_col='app_user_id',
                                  out_path: Path = Path('steps_and_survey_heatmap.png')):
    """Plot combined heatmap showing step data and survey submissions for each user by day.
    
    Color coding:
    - Dark Green (#228B22): Has both step data and survey
    - Light Green (#90EE90): Has step data only
    - Red (#DC143C): Has survey only
    - White (#FFFFFF): Neither step data nor survey
    """
    # Create pivot tables
    steps_pivot = daily_steps.pivot(index=app_user_col, columns='day', values='count').fillna(0)
    surveys_pivot = daily_surveys.pivot(index=app_user_col, columns='day', values='survey_submitted').fillna(0)
    
    # Ensure both have the same shape and index
    all_users = sorted(set(steps_pivot.index) | set(surveys_pivot.index))
    all_days = sorted(set(steps_pivot.columns) | set(surveys_pivot.columns))
    
    steps_pivot = steps_pivot.reindex(all_users, fill_value=0)
    surveys_pivot = surveys_pivot.reindex(all_users, fill_value=0)
    steps_pivot = steps_pivot.reindex(columns=all_days, fill_value=0)
    surveys_pivot = surveys_pivot.reindex(columns=all_days, fill_value=0)
    
    # Create binary indicators
    has_steps = (steps_pivot > 0).astype(int)
    has_survey = (surveys_pivot > 0).astype(int)
    
    # Combine into single matrix with 4 states: 0=none, 1=steps only, 2=survey only, 3=both
    combined = has_steps + (has_survey * 2)
    
    fig, ax = plt.subplots(figsize=(20, max(8, len(all_users) * 0.4)))
    
    # Create custom colormap: white (none), light green (steps), red (survey), dark green (both)
    colors_list = ['#FFFFFF', '#90EE90', '#DC143C', '#228B22']
    cmap = plt.cm.colors.ListedColormap(colors_list)
    
    im = ax.imshow(combined, cmap=cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=3)
    
    # Set ticks and labels
    study_days = len(all_days)
    ax.set_xticks(range(study_days))
    ax.set_xticklabels(range(1, study_days + 1))
    ax.set_yticks(range(len(all_users)))
    ax.set_yticklabels(all_users)
    
    # Add date labels if available
    if 'date' in daily_steps.columns:
        date_map = daily_steps.drop_duplicates('day').set_index('day')['date'].to_dict()
        date_labels = [pd.to_datetime(date_map.get(d, '')).strftime('%m-%d')
                       if date_map.get(d) else '' for d in sorted(all_days)]
        ax.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=9)
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Participant ID', fontsize=12, fontweight='bold')
    ax.set_title('Daily Step Data and Survey Submission Status by Participant',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add key
    cbar = plt.colorbar(im, ax=ax, ticks=[0.375, 1.125, 1.875, 2.625])
    cbar.ax.set_yticklabels(['Neither', 'Steps Only', 'Survey Only', 'Both'])
    cbar.set_label('Data Status', fontsize=11, fontweight='bold')
    
    # Add grid
    ax.set_xticks([x - 0.5 for x in range(1, study_days)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(all_users))], minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def per_user_daily_screentime_submissions(df: pd.DataFrame, timestamp_col: str, app_user_col: str = 'app_user_id', study_start=None):
    """Return DataFrame with rows (app_user_id, day_index) and screentime_submitted flag for each day.
    Only includes data from November 15th, 2025 onwards and before December 31, 2025.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[app_user_col, 'day', 'screentime_submitted', 'date'])

    if app_user_col not in df.columns:
        df['app_user_id'] = 'all'
        app_user_col = 'app_user_id'

    df = df.dropna(subset=[timestamp_col]).copy()
    if df.empty:
        return pd.DataFrame(columns=[app_user_col, 'day', 'screentime_submitted', 'date'])

    # Filter to only include dates from November 15th, 2025 onwards and before December 31, 2025
    start_cutoff_date = pd.Timestamp('2025-11-15')
    end_cutoff_date = pd.Timestamp('2025-12-31')
    df = df[(df[timestamp_col] >= start_cutoff_date) & (df[timestamp_col] < end_cutoff_date)]

    if df.empty:
        return pd.DataFrame(columns=[app_user_col, 'day', 'screentime_submitted', 'date'])

    if study_start is None:
        study_start = pd.Timestamp('2025-11-15').normalize()
    else:
        study_start = pd.to_datetime(study_start).normalize()

    df['day'] = (df[timestamp_col].dt.normalize() - study_start).dt.days + 1
    max_day = df['day'].max()
    study_days = int(max_day) if max_day > 0 else 28

    df = df[df['day'] >= 1]

    grouped = df.groupby([app_user_col, 'day']).size().reset_index(name='count')
    grouped['screentime_submitted'] = 1

    users = grouped[app_user_col].unique().tolist()
    all_rows = []
    for u in users:
        user_days = grouped[grouped[app_user_col] == u].set_index('day')['screentime_submitted']
        for d in range(1, study_days + 1):
            submitted = int(user_days.get(d, 0))
            date_for_d = (pd.to_datetime(study_start) + pd.Timedelta(days=d - 1)).date()
            all_rows.append({app_user_col: u, 'day': d, 'screentime_submitted': submitted, 'date': date_for_d})

    return pd.DataFrame(all_rows)


def plot_screentime_and_survey_heatmap(daily_screentime: pd.DataFrame, daily_surveys: pd.DataFrame,
                                       app_user_col='app_user_id',
                                       out_path: Path = Path('screentime_and_survey_heatmap.png')):
    """Plot combined heatmap showing screentime submissions and survey submissions for each user by day."""
    screentime_pivot = daily_screentime.pivot(index=app_user_col, columns='day', values='screentime_submitted').fillna(0)
    surveys_pivot = daily_surveys.pivot(index=app_user_col, columns='day', values='survey_submitted').fillna(0)

    all_users = sorted(set(screentime_pivot.index) | set(surveys_pivot.index))
    all_days = sorted(set(screentime_pivot.columns) | set(surveys_pivot.columns))

    screentime_pivot = screentime_pivot.reindex(all_users, fill_value=0)
    surveys_pivot = surveys_pivot.reindex(all_users, fill_value=0)
    screentime_pivot = screentime_pivot.reindex(columns=all_days, fill_value=0)
    surveys_pivot = surveys_pivot.reindex(columns=all_days, fill_value=0)

    has_screentime = (screentime_pivot > 0).astype(int)
    has_survey = (surveys_pivot > 0).astype(int)
    combined = has_screentime + (has_survey * 2)

    fig, ax = plt.subplots(figsize=(20, max(8, len(all_users) * 0.4)))

    colors_list = ['#FFFFFF', '#90EE90', '#DC143C', '#228B22']
    cmap = plt.cm.colors.ListedColormap(colors_list)

    im = ax.imshow(combined, cmap=cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=3)

    study_days = len(all_days)
    ax.set_xticks(range(study_days))
    ax.set_xticklabels(range(1, study_days + 1))
    ax.set_yticks(range(len(all_users)))
    ax.set_yticklabels(all_users)

    if 'date' in daily_screentime.columns:
        date_map = daily_screentime.drop_duplicates('day').set_index('day')['date'].to_dict()
        date_labels = [pd.to_datetime(date_map.get(d, '')).strftime('%m-%d')
                       if date_map.get(d) else '' for d in sorted(all_days)]
        ax.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=9)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Participant ID', fontsize=12, fontweight='bold')
    ax.set_title('Daily Screentime Submission and Survey Status by Participant',
                 fontsize=14, fontweight='bold', pad=20)

    cbar = plt.colorbar(im, ax=ax, ticks=[0.375, 1.125, 1.875, 2.625])
    cbar.ax.set_yticklabels(['Neither', 'Screentime Only', 'Survey Only', 'Both'])
    cbar.set_label('Data Status', fontsize=11, fontweight='bold')

    ax.set_xticks([x - 0.5 for x in range(1, study_days)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(all_users))], minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def run_analysis(output_dir: Path = Path('analysis_outputs')):
    output_dir.mkdir(parents=True, exist_ok=True)
    service = DatabaseService()
    surveys = None
    screentime = None

    try:
        # Load steps data from database only
        print("Loading steps data from database...")
        steps = load_table_from_db(service, 'step')

        if steps is None or steps.empty:
            print('No steps data found in database.')
            return None

        print(f"Steps data columns: {steps.columns.tolist()}")
        print(f"Steps data shape: {steps.shape}")

        # Load screentime data for the parallel submission heatmap
        print("Loading screentime data from database...")
        screentime = load_table_from_db(service, 'screentime')

        if screentime is None or screentime.empty:
            print("Warning: Could not load screentime data - screentime visualization will be skipped")
        else:
            print(f"Screentime data columns: {screentime.columns.tolist()}")
            print(f"Screentime data shape: {screentime.shape}")
            if 'app_user_id' not in screentime.columns:
                print("Warning: screentime data missing 'app_user_id' column")
            ts_col_check = normalize_timestamp_column(screentime)
            if ts_col_check is None:
                print("Warning: screentime data has no recognized timestamp column")
            else:
                print(f"Screentime timestamp column identified: '{ts_col_check}'")

        # Load survey data
        print("Loading survey data from database...")
        surveys = None
        surveys = load_table_from_db(service, 'survey_response')
        if surveys is not None and not surveys.empty:
            print("Loaded survey data from survey_response table")
        
        if surveys is None or surveys.empty:
            print("Warning: Could not load survey data - combined visualization will be skipped")
        else:
            print(f"Survey data columns: {surveys.columns.tolist()}")
            print(f"Survey data shape: {surveys.shape}")
            # Verify required columns exist
            if 'app_user_id' not in surveys.columns:
                print("Warning: survey data missing 'app_user_id' column")
            ts_col_check = normalize_timestamp_column(surveys)
            if ts_col_check is None:
                print("Warning: survey data has no recognized timestamp column")
            else:
                print(f"Survey timestamp column identified: '{ts_col_check}'")

        # Load umass_id mapping table
        print("Loading umass_id mapping table...")
        id_mapping = load_table_from_db(service, 'umass_id')

        if id_mapping is not None and not id_mapping.empty:
            print(f"Loaded {len(id_mapping)} ID mappings")
            # Map app_user_id to umass_id in steps data
            steps = map_to_umass_ids(steps, id_mapping, 'app_user_id')
            # Also map surveys if available
            if surveys is not None and not surveys.empty:
                surveys = map_to_umass_ids(surveys, id_mapping, 'app_user_id')
            if screentime is not None and not screentime.empty:
                screentime = map_to_umass_ids(screentime, id_mapping, 'app_user_id')
            # DO NOT RENAME - keep as 'app_user_id' so all downstream functions work
            print("Mapped app_user_id to umass_id in data")
            print(f"Steps columns after mapping: {steps.columns.tolist()}")
        else:
            print("Warning: Could not load umass_id mapping, using app_user_id")

    finally:
        # Disconnect after loading all tables
        service.disconnect()

    results = {}

    # Process steps data
    print("\n" + "="*70)
    print("ANALYZING STEPS DATA")
    print("="*70)

    ts_col = normalize_timestamp_column(steps)
    if ts_col is None:
        print('No timestamp column found in steps data.')
        return None

    daily_counts = per_user_daily_counts(steps, ts_col, app_user_col='app_user_id')

    print(f"Daily counts columns: {daily_counts.columns.tolist()}")
    print(f"Daily counts shape: {daily_counts.shape}")

    if daily_counts.empty:
        print("No daily counts generated.")
        return None

    # Debug: Print all unique users found (now showing umass_id)
    all_unique_users = daily_counts['app_user_id'].unique().tolist()
    print(f"\nFound {len(all_unique_users)} unique participants (umass_id): {sorted(all_unique_users)}")

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

    # Calculate and plot completeness scores
    completeness_df = calculate_completeness_score(daily_counts)
    completeness_path = plot_completeness_score(
        completeness_df,
        out_path=output_dir / 'steps_completeness_score.png'
    )
    completeness_df.to_csv(output_dir / 'steps_completeness_scores.csv', index=False)

    # Plot availability heatmap
    heatmap_path = plot_availability_heatmap(
        daily_counts,
        out_path=output_dir / 'steps_availability_heatmap.png'
    )

    # Plot daily participation rate
    participation_path = plot_daily_participation(
        daily_counts,
        out_path=output_dir / 'steps_daily_participation.png'
    )

    # Plot completeness distribution
    distribution_path = plot_completeness_distribution(
        completeness_df,
        out_path=output_dir / 'steps_completeness_distribution.png'
    )

    print(f'Saved completeness score chart to {completeness_path}')
    print(f'Saved completeness scores CSV to {output_dir / "steps_completeness_scores.csv"}')
    print(f'Saved availability heatmap to {heatmap_path}')
    print(f'Saved daily participation chart to {participation_path}')
    print(f'Saved completeness distribution to {distribution_path}')

    # Generate ALL USERS plots
    print(f"\nGenerating plots for all {len(all_unique_users)} participants...")
    hourly_all = hourly_average_for_users(steps, all_unique_users, ts_col)

    # Generate daily record counts (counting records, not summing steps)
    daily_record_counts = per_user_daily_record_counts(steps, ts_col)

    all_daily_records_path = plot_all_users_daily_records(
        daily_record_counts,
        out_path=output_dir / 'steps_all_users_daily_records.png'
    )

    all_timeline_path = plot_all_users_timeline(
        daily_counts,
        out_path=output_dir / 'steps_all_users_timeline.png'
    )

    all_hourly_records_path = plot_all_users_hourly_records(
        hourly_all,
        out_path=output_dir / 'steps_all_users_hourly_records.png'
    )

    all_hourly_steps_path = plot_all_users_hourly_steps(
        hourly_all,
        out_path=output_dir / 'steps_all_users_hourly_steps.png'
    )

    print(f'Saved ALL USERS daily records to {all_daily_records_path}')
    print(f'Saved ALL USERS timeline to {all_timeline_path}')
    print(f'Saved ALL USERS hourly records to {all_hourly_records_path}')
    print(f'Saved ALL USERS hourly steps to {all_hourly_steps_path}')

    # Generate combined step and survey heatmap if survey data is available
    combined_heatmap_path = None
    if surveys is not None and not surveys.empty:
        print("\nGenerating combined step and survey heatmap...")
        ts_col_survey = normalize_timestamp_column(surveys)
        if ts_col_survey is not None:
            daily_surveys = per_user_daily_survey_submissions(surveys, ts_col_survey, app_user_col='app_user_id')
            if not daily_surveys.empty:
                combined_heatmap_path = plot_step_and_survey_heatmap(
                    daily_counts,
                    daily_surveys,
                    out_path=output_dir / 'steps_and_survey_heatmap.png'
                )
                print(f'Saved combined step and survey heatmap to {combined_heatmap_path}')
            else:
                print("Warning: Could not generate daily survey submissions data")
        else:
            print("Warning: Could not find timestamp column in survey data")
    else:
        print("\nSkipping combined step and survey heatmap - no survey data available")

    # Generate combined screentime and survey heatmap if screentime data is available
    screentime_heatmap_path = None
    if screentime is not None and not screentime.empty:
        print("\nGenerating combined screentime and survey heatmap...")
        ts_col_screentime = normalize_timestamp_column(screentime)
        if ts_col_screentime is not None:
            daily_screentime = per_user_daily_screentime_submissions(screentime, ts_col_screentime, app_user_col='app_user_id')
            if not daily_screentime.empty and surveys is not None and not surveys.empty:
                ts_col_survey = normalize_timestamp_column(surveys)
                if ts_col_survey is not None:
                    daily_surveys = per_user_daily_survey_submissions(surveys, ts_col_survey, app_user_col='app_user_id')
                    if not daily_surveys.empty:
                        screentime_heatmap_path = plot_screentime_and_survey_heatmap(
                            daily_screentime,
                            daily_surveys,
                            out_path=output_dir / 'screentime_and_survey_heatmap.png'
                        )
                        print(f'Saved combined screentime and survey heatmap to {screentime_heatmap_path}')
                    else:
                        print("Warning: Could not generate daily survey submissions data for screentime heatmap")
                else:
                    print("Warning: Could not find timestamp column in survey data")
            else:
                print("Warning: Could not generate daily screentime submissions data")
        else:
            print("Warning: Could not find timestamp column in screentime data")
    else:
        print("\nSkipping combined screentime and survey heatmap - no screentime data available")

    results['steps'] = {
        'daily_counts': daily_counts,
        'summary': summary,
        'timeline_path': timeline_path,
        'hourly_records_path': hourly_records_path,
        'hourly_steps_path': hourly_steps_path,
        'hourly_df': hourly,
        'completeness_path': completeness_path,
        'heatmap_path': heatmap_path,
        'participation_path': participation_path,
        'distribution_path': distribution_path,
        'all_daily_records_path': all_daily_records_path,
        'all_timeline_path': all_timeline_path,
        'all_hourly_records_path': all_hourly_records_path,
        'all_hourly_steps_path': all_hourly_steps_path,
        'combined_heatmap_path': combined_heatmap_path,
        'screentime_heatmap_path': screentime_heatmap_path
    }

    return results

if __name__ == '__main__':
    run_analysis()