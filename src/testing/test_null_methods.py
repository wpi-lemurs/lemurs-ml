"""
Test script to verify the three null handling methods work correctly.
"""
from src.health_data_analysis import daily_health_with_week, hourly_health_data

print("Testing null handling methods...")
print("="*60)

# Test daily data with all three methods
print("\n1. Testing DAILY data with null_method=None (no handling)")
daily_none = daily_health_with_week(app_user_id=-1, null_method=None)
print(f"   Shape: {daily_none.shape}")
print(f"   Null counts:\n{daily_none[['daily_steps', 'daily_distance', 'daily_calories', 'daily_avg_speed']].isna().sum()}")

print("\n2. Testing DAILY data with null_method='linear' (linear interpolation)")
daily_linear = daily_health_with_week(app_user_id=-1, null_method='linear')
print(f"   Shape: {daily_linear.shape}")
print(f"   Null counts:\n{daily_linear[['daily_steps', 'daily_distance', 'daily_calories', 'daily_avg_speed']].isna().sum()}")

print("\n3. Testing DAILY data with null_method='fill' (forward/backward fill)")
daily_fill = daily_health_with_week(app_user_id=-1, null_method='fill')
print(f"   Shape: {daily_fill.shape}")
print(f"   Null counts:\n{daily_fill[['daily_steps', 'daily_distance', 'daily_calories', 'daily_avg_speed']].isna().sum()}")

print("\n" + "="*60)

# Test hourly data with all three methods
print("\n4. Testing HOURLY data with null_method=None (no handling)")
hourly_none = hourly_health_data(app_user_id=-1, null_method=None)
print(f"   Shape: {hourly_none.shape}")
print(f"   Null counts:\n{hourly_none[['hourly_steps', 'hourly_distance', 'hourly_calories', 'hourly_avg_speed']].isna().sum()}")

print("\n5. Testing HOURLY data with null_method='linear' (linear interpolation)")
hourly_linear = hourly_health_data(app_user_id=-1, null_method='linear')
print(f"   Shape: {hourly_linear.shape}")
print(f"   Null counts:\n{hourly_linear[['hourly_steps', 'hourly_distance', 'hourly_calories', 'hourly_avg_speed']].isna().sum()}")

print("\n6. Testing HOURLY data with null_method='fill' (forward/backward fill)")
hourly_fill = hourly_health_data(app_user_id=-1, null_method='fill')
print(f"   Shape: {hourly_fill.shape}")
print(f"   Null counts:\n{hourly_fill[['hourly_steps', 'hourly_distance', 'hourly_calories', 'hourly_avg_speed']].isna().sum()}")

print("\n" + "="*60)
print("All tests completed successfully!")
print("="*60)

# Test with specific user
print("\nTesting with specific user (user_id=20):")
daily_user = daily_health_with_week(app_user_id=20, null_method='linear')
print(f"Daily data for user 20 - Shape: {daily_user.shape}")
print(f"Unique users: {daily_user['app_user_id'].unique()}")

