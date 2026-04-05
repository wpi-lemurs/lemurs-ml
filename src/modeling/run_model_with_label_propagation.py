"""
Example script showing how to run the screentime time window models with label propagation.

Label propagation treats any user who has been labeled as 'at_risk' or 'depressed'
at least once as having that label for ALL of their entries. This helps address
severe class imbalance when there are very few positive cases.

Supports all daily labels:
- phq9 (depression), suicide_risk, self_harm, positive_emotion, negative_emotion
- social_stress, social_connection, minority_stress, emotion_regulation, sleep

Usage:
    # For PHQ-9 depression prediction WITH label propagation:
    python src/modeling/run_model_with_label_propagation.py phq9

    # For social connection prediction WITH label propagation:
    python src/modeling/run_model_with_label_propagation.py social_connection

    # For any other label WITH label propagation:
    python src/modeling/run_model_with_label_propagation.py negative_emotion

    # Without label propagation (default behavior):
    python src/modeling/model_screentime_time_windows.py <label_name>
"""

import sys
import os

# Add project root to path to ensure imports work from any directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.modeling.model_screentime_time_windows import main, AVAILABLE_LABELS

if __name__ == '__main__':
    import sys

    # Check if a target type is provided as a command line argument
    if len(sys.argv) > 1:
        target = sys.argv[1].lower()
        if target in AVAILABLE_LABELS.keys():
            # Enable label propagation
            main(target_type=target, propagate_labels=True)
        else:
            print(f"Invalid target type: {target}")
            print(f"Valid options: {', '.join(AVAILABLE_LABELS.keys())}")
            print("Using default: 'phq9' with label propagation")
            main(target_type='phq9', propagate_labels=True)
    else:
        # Default to PHQ-9 prediction with label propagation
        print("No target type specified. Using 'phq9' with label propagation.")
        print(f"Usage: python run_model_with_label_propagation.py [target_type]")
        print(f"Valid target types: {', '.join(AVAILABLE_LABELS.keys())}")
        main(target_type='phq9', propagate_labels=True)
