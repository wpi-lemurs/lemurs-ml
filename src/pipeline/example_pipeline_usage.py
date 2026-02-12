"""
Example script demonstrating the scikit-learn pipeline architecture.

This script shows several common use cases for the mental health prediction pipeline.

Usage:
    python example_pipeline_usage.py
"""

import sys
from pathlib import Path

# Add src to path if needed
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.pipeline.model_pipeline import ScreentimeModelPipeline, run_experiment


def example_1_basic_sleep():
    """Example 1: Basic sleep risk prediction with different time windows."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Sleep Risk Prediction")
    print("="*80)

    pipeline = ScreentimeModelPipeline(
        target_type='sleep',
        time_windows=[3, 6, 9],
        fill_method='zero',
        use_loocv=False,  # Use train/test split
        balanced_class_weight=False
    )

    results = pipeline.fit_predict()

    # Print summary
    print("\n--- Results Summary ---")
    for window, window_results in results.items():
        print(f"\nTime Window: {window} hours")
        print(f"  Total Samples: {window_results['total_samples']}")
        print(f"  Class Distribution: {window_results['class_distribution']}")

        for model_name, metrics in window_results['models'].items():
            print(f"\n  {model_name.replace('_', ' ').title()}:")
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            if metrics.get('f1_score') is not None:
                print(f"    F1 Score: {metrics['f1_score']:.4f}")


def example_2_with_label_propagation():
    """Example 2: Self-harm prediction with label propagation."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Self-Harm Prediction with Label Propagation")
    print("="*80)

    pipeline = ScreentimeModelPipeline(
        target_type='self_harm',
        time_windows=[6, 12],
        propagate_labels=True,  # Propagate positive labels
        balanced_class_weight=False,
        use_loocv=False
    )

    results = pipeline.fit_predict()
    return results


def example_3_with_class_balancing():
    """Example 3: Sleep risk with balanced class weights."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Sleep Risk with Balanced Class Weights")
    print("="*80)

    pipeline = ScreentimeModelPipeline(
        target_type='sleep',
        time_windows=[3, 6, 9],
        balanced_class_weight=True,  # Use balanced class weights
        use_loocv=False
    )

    results = pipeline.fit_predict()
    return results


def example_4_with_loocv():
    """Example 4: Depression prediction with LOOCV."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Depression (PHQ-9) with Leave-One-User-Out CV")
    print("="*80)

    pipeline = ScreentimeModelPipeline(
        target_type='phq9',
        time_windows=[6, 12],
        use_loocv=True,  # Use leave-one-user-out cross-validation
        balanced_class_weight=True
    )

    results = pipeline.fit_predict()
    return results


def example_5_convenience_function():
    """Example 5: Using the convenience function for quick experiments."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Quick Experiment Using Convenience Function")
    print("="*80)

    results = run_experiment(
        target_type='suicide_risk',
        time_windows=[3, 6],
        propagate_labels=True,
        balanced_class_weight=True,
        use_loocv=False
    )

    return results


def example_6_compare_configurations():
    """Example 6: Compare different configurations."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Comparing Different Configurations")
    print("="*80)

    configurations = [
        {'name': 'Baseline', 'propagate': False, 'balanced': False},
        {'name': 'With Label Propagation', 'propagate': True, 'balanced': False},
        {'name': 'With Class Balancing', 'propagate': False, 'balanced': True},
        {'name': 'Both Techniques', 'propagate': True, 'balanced': True},
    ]

    all_results = {}

    for config in configurations:
        print(f"\n--- Testing: {config['name']} ---")

        pipeline = ScreentimeModelPipeline(
            target_type='suicide_risk',
            time_windows=[6],  # Single window for quick comparison
            propagate_labels=config['propagate'],
            balanced_class_weight=config['balanced'],
            use_loocv=False,
            save_confusion_matrices=False  # Don't save for comparison
        )

        results = pipeline.fit_predict()
        all_results[config['name']] = results

    # Compare results
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    for name, results in all_results.items():
        if results and 6 in results:  # Check if we have results for 6h window
            window_results = results[6]
            print(f"\n{name}:")
            print(f"  Samples: {window_results['total_samples']}")

            for model_name, metrics in window_results['models'].items():
                f1 = metrics.get('f1_score')
                f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
                print(f"  {model_name}: Acc={metrics['accuracy']:.4f}, F1={f1_str}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("SCIKIT-LEARN PIPELINE EXAMPLES")
    print("Mental Health Prediction using Screentime Data")
    print("="*80)

    examples = [
        ("Basic Suicide Risk", example_1_basic_sleep),
        ("Label Propagation", example_2_with_label_propagation),
        ("Class Balancing", example_3_with_class_balancing),
        ("LOOCV", example_4_with_loocv),
        ("Convenience Function", example_5_convenience_function),
        ("Configuration Comparison", example_6_compare_configurations),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nRunning Example 1 (Basic Suicide Risk)...")
    print("To run other examples, modify the main() function.")
    print("-"*80)

    # Run first example by default
    example_1_basic_sleep()

    # Uncomment to run other examples:
    # example_2_with_label_propagation()
    # example_3_with_class_balancing()
    # example_4_with_loocv()
    # example_5_convenience_function()
    # example_6_compare_configurations()


if __name__ == "__main__":
    main()

