"""
Test script to demonstrate the date range filtering feature in health_data_analysis.
"""
from src.health_data_analysis import daily_health_with_week, hourly_health_data, weekly_avg_health_data, steps_data
import pandas as pd

print("="*70)
print("DATE RANGE FILTERING TEST")
print("="*70)

# Test 1: Daily health data with date range
print("\n1. Testing daily_health_with_week() with date range")
print("-"*70)

# Get all data first to see the full range
all_daily = daily_health_with_week(app_user_id=20, null_method=None)
if not all_daily.empty:
    print(f"Full date range: {all_daily['date'].min()} to {all_daily['date'].max()}")
    print(f"Total days: {len(all_daily)}")

    # Now filter to a specific date range
    filtered_daily = daily_health_with_week(
        app_user_id=20,
        null_method=None,
        date_range=('2025-10-01', '2025-10-31')
    )
    if not filtered_daily.empty:
        print(f"\nFiltered date range (October 2025): {filtered_daily['date'].min()} to {filtered_daily['date'].max()}")
        print(f"Total days in October: {len(filtered_daily)}")
    else:
        print("\nNo data found in October 2025")
else:
    print("No data found for user 20")

# Test 2: Hourly health data with date range
print("\n\n2. Testing hourly_health_data() with date range")
print("-"*70)

all_hourly = hourly_health_data(app_user_id=20, null_method=None)
if not all_hourly.empty:
    print(f"Full date range: {all_hourly['datetime'].min()} to {all_hourly['datetime'].max()}")
    print(f"Total hours: {len(all_hourly)}")

    # Filter to a specific date range
    filtered_hourly = hourly_health_data(
        app_user_id=20,
        null_method=None,
        date_range=('2025-11-01', '2025-11-15')
    )
    if not filtered_hourly.empty:
        print(f"\nFiltered date range (Nov 1-15, 2025): {filtered_hourly['datetime'].min()} to {filtered_hourly['datetime'].max()}")
        print(f"Total hours in range: {len(filtered_hourly)}")
    else:
        print("\nNo data found in November 1-15, 2025")
else:
    print("No data found for user 20")

# Test 3: Weekly average with date range
print("\n\n3. Testing weekly_avg_health_data() with date range")
print("-"*70)

all_weekly = weekly_avg_health_data(
    steps_data,
    app_user_id=20
)
if not all_weekly.empty:
    print(f"Full date range: {all_weekly['week_start'].min()} to {all_weekly['week_start'].max()}")
    print(f"Total weeks: {len(all_weekly)}")

    # Filter to a specific date range
    filtered_weekly = weekly_avg_health_data(
        steps_data,
        app_user_id=20,
        date_range=('2025-09-01', '2025-09-30')
    )
    if not filtered_weekly.empty:
        print(f"\nFiltered date range (September 2025): {filtered_weekly['week_start'].min()} to {filtered_weekly['week_start'].max()}")
        print(f"Total weeks in September: {len(filtered_weekly)}")
        print(f"\nWeekly averages:\n{filtered_weekly}")
    else:
        print("\nNo data found in September 2025")
else:
    print("No data found for user 20")

# Test 4: Multiple users with date range
print("\n\n4. Testing with all users and date range")
print("-"*70)

all_users_daily = daily_health_with_week(
    app_user_id=-1,
    null_method=None,
    date_range=('2025-11-01', '2025-11-30')
)
if not all_users_daily.empty:
    print(f"Date range (November 2025): {all_users_daily['date'].min()} to {all_users_daily['date'].max()}")
    print(f"Total days (all users): {len(all_users_daily)}")
    print(f"Unique users: {sorted(all_users_daily['app_user_id'].unique())}")
    print(f"Days per user:\n{all_users_daily.groupby('app_user_id').size()}")
else:
    print("No data found for November 2025")

# Test 5: Edge cases
print("\n\n5. Testing edge cases")
print("-"*70)

# Empty date range (future dates)
future_data = daily_health_with_week(
    app_user_id=20,
    date_range=('2026-01-01', '2026-12-31')
)
print(f"Future date range (2026): {len(future_data)} rows (expected: 0)")

# Single day
single_day = daily_health_with_week(
    app_user_id=20,
    date_range=('2025-11-21', '2025-11-21')
)
print(f"Single day (2025-11-21): {len(single_day)} rows")

# Date range with datetime objects
from datetime import datetime
datetime_range = daily_health_with_week(
    app_user_id=20,
    date_range=(datetime(2025, 11, 1), datetime(2025, 11, 30))
)
print(f"Using datetime objects (November 2025): {len(datetime_range)} rows")

print("\n" + "="*70)
print("ALL TESTS COMPLETE")
print("="*70)
print("\nDate range filtering is now available for:")
print("  - daily_health_with_week(date_range=(start, end))")
print("  - hourly_health_data(date_range=(start, end))")
print("  - weekly_avg_health_data(date_range=(start, end))")
print("\nDate formats accepted:")
print("  - String: '2025-01-01'")
print("  - String with time: '2025-01-01 12:00:00'")
print("  - datetime objects: datetime(2025, 1, 1)")

