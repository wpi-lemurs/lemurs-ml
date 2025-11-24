"""
Test script to demonstrate linear interpolation functionality for health data.
"""
from src.health_data_analysis import daily_health_with_week, hourly_health_data
import pandas as pd

# Set display options to see all data
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)

print("=" * 80)
print("TESTING LINEAR INTERPOLATION FOR HEALTH DATA")
print("=" * 80)

# Test 1: Daily health data WITHOUT interpolation
print("\n1. Daily Health Data WITHOUT Interpolation (showing first 15 rows):")
print("-" * 80)
daily_without = daily_health_with_week(interpolate=False)
print(daily_without.head(15))
print(f"\nNull values per column:")
print(daily_without.isnull().sum())

# Test 2: Daily health data WITH interpolation
print("\n" + "=" * 80)
print("2. Daily Health Data WITH Interpolation (showing first 15 rows):")
print("-" * 80)
daily_with = daily_health_with_week(interpolate=True)
print(daily_with.head(15))
print(f"\nNull values per column:")
print(daily_with.isnull().sum())

# Test 3: Hourly health data WITHOUT interpolation
print("\n" + "=" * 80)
print("3. Hourly Health Data WITHOUT Interpolation (showing first 15 rows):")
print("-" * 80)
hourly_without = hourly_health_data(interpolate=False)
print(hourly_without.head(15))
print(f"\nNull values per column:")
print(hourly_without.isnull().sum())

# Test 4: Hourly health data WITH interpolation
print("\n" + "=" * 80)
print("4. Hourly Health Data WITH Interpolation (showing first 15 rows):")
print("-" * 80)
hourly_with = hourly_health_data(interpolate=True)
print(hourly_with.head(15))
print(f"\nNull values per column:")
print(hourly_with.isnull().sum())

# Show comparison for a specific user
print("\n" + "=" * 80)
print("5. Comparison for User 21 (Daily Data):")
print("-" * 80)
user_21_without = daily_without[daily_without['app_user_id'] == 21][['date', 'daily_steps', 'daily_distance', 'daily_calories', 'daily_avg_speed']]
user_21_with = daily_with[daily_with['app_user_id'] == 21][['date', 'daily_steps', 'daily_distance', 'daily_calories', 'daily_avg_speed']]

print("\nWithout Interpolation:")
print(user_21_without)
print("\nWith Interpolation:")
print(user_21_with)

print("\n" + "=" * 80)
print("INTERPOLATION TEST COMPLETE")
print("=" * 80)

