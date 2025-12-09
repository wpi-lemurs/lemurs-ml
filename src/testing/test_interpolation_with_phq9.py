"""
Test script to demonstrate linear interpolation with merged PHQ-9 data.
"""
from src.merge_passive_data_and_labels import merge_daily_health_with_phq9, merge_hourly_health_with_phq9
import pandas as pd

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)

print("=" * 80)
print("TESTING INTERPOLATION WITH PHQ-9 MERGED DATA")
print("=" * 80)

# Test 1: Daily merged data WITHOUT interpolation
print("\n1. Daily Health + PHQ-9 WITHOUT Interpolation:")
print("-" * 80)
daily_without = merge_daily_health_with_phq9(interpolate=False)
print(daily_without.head(10))
print(f"\nNull values per column:")
print(daily_without.isnull().sum())
print(f"\nTotal rows: {len(daily_without)}")

# Test 2: Daily merged data WITH interpolation
print("\n" + "=" * 80)
print("2. Daily Health + PHQ-9 WITH Interpolation:")
print("-" * 80)
daily_with = merge_daily_health_with_phq9(interpolate=True)
print(daily_with.head(10))
print(f"\nNull values per column:")
print(daily_with.isnull().sum())
print(f"\nTotal rows: {len(daily_with)}")

# Test 3: Hourly merged data WITHOUT interpolation
print("\n" + "=" * 80)
print("3. Hourly Health + PHQ-9 WITHOUT Interpolation:")
print("-" * 80)
hourly_without = merge_hourly_health_with_phq9(interpolate=False)
print(hourly_without.head(10))
print(f"\nNull values per column:")
print(hourly_without.isnull().sum())
print(f"\nTotal rows: {len(hourly_without)}")

# Test 4: Hourly merged data WITH interpolation
print("\n" + "=" * 80)
print("4. Hourly Health + PHQ-9 WITH Interpolation:")
print("-" * 80)
hourly_with = merge_hourly_health_with_phq9(interpolate=True)
print(hourly_with.head(10))
print(f"\nNull values per column:")
print(hourly_with.isnull().sum())
print(f"\nTotal rows: {len(hourly_with)}")

print("\n" + "=" * 80)
print("KEY BENEFIT: Interpolation significantly reduces null values!")
print("This provides more complete data for machine learning models.")
print("=" * 80)

