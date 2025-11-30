"""
Test script to demonstrate the metrics selection feature in null_value_visualization.
"""
from src.visualization.null_value_visualization import visualize_steps_with_null_handling

print("="*70)
print("TESTING METRICS SELECTION FEATURE")
print("="*70)

# Example 1: Visualize all metrics (default behavior)
print("\n1. Visualize ALL metrics for user 20 (default)")
print("-"*70)
visualize_steps_with_null_handling(user_id=20, time_unit='D', metrics_to_plot=None)

# Example 2: Visualize only steps
print("\n2. Visualize ONLY STEPS for user 20")
print("-"*70)
visualize_steps_with_null_handling(user_id=20, time_unit='D', metrics_to_plot=['steps'])

# Example 3: Visualize steps and distance
print("\n3. Visualize STEPS and DISTANCE for user 20")
print("-"*70)
visualize_steps_with_null_handling(user_id=20, time_unit='D', metrics_to_plot=['steps', 'distance'])

# Example 4: Visualize calories and speed
print("\n4. Visualize CALORIES and SPEED for user 20")
print("-"*70)
visualize_steps_with_null_handling(user_id=20, time_unit='D', metrics_to_plot=['calories', 'speed'])

# Example 5: Visualize only speed
print("\n5. Visualize ONLY SPEED for user 20")
print("-"*70)
visualize_steps_with_null_handling(user_id=20, time_unit='D', metrics_to_plot=['speed'])

print("\n" + "="*70)
print("ALL TESTS COMPLETE")
print("="*70)

