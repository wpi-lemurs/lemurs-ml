"""
Example script demonstrating how to save confusion matrices using the pipeline.

This shows different ways to save confusion matrices:
1. Automatically during fit_predict (when save_results=True)
2. Manually after training
3. Customizing which matrices to save
"""

from src.pipeline.model_pipeline import ScreentimeModelPipeline


def example_automatic_save():
    """
    Example 1: Automatic saving during fit_predict.

    When save_results=True, confusion matrices are automatically saved
    after model training completes.
    """
    print("="*80)
    print("EXAMPLE 1: Automatic Confusion Matrix Saving")
    print("="*80)

    pipeline = ScreentimeModelPipeline(
        target_type='sleep',
        time_windows=[3, 6, 9],
        balanced_class_weight=True,
        save_confusion_matrices=True  # This enables automatic saving
    )

    # Run pipeline - confusion matrices will be saved automatically
    results = pipeline.fit_predict()

    print("\nConfusion matrices saved automatically to:")
    print(f"  - confusion_matrices_suicide_risk_balanced.png (all windows)")
    print(f"  - confusion_matrices_suicide_risk_balanced_best.png (best only)")


def example_manual_save():
    """
    Example 2: Manual saving after training.

    Train the model first, then manually call save_confusion_matrices
    with custom options.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Manual Confusion Matrix Saving")
    print("="*80)

    pipeline = ScreentimeModelPipeline(
        target_type='sleep',
        time_windows=[3, 6, 9],
        save_confusion_matrices=False  # Disable automatic saving
    )

    # Run pipeline without automatic saving
    results = pipeline.fit_predict()

    # Manually save only the best models
    print("\nSaving only best models confusion matrices...")
    pipeline.save_confusion_matrices_plots(save_all_windows=False, save_best_only=True)

    print("Best models confusion matrix saved!")


def example_custom_save():
    """
    Example 3: Save both types of confusion matrices manually.

    Useful when you want full control over when matrices are saved.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Custom Confusion Matrix Saving")
    print("="*80)

    pipeline = ScreentimeModelPipeline(
        target_type='self_harm',
        time_windows=[3, 6, 9, 12],
        balanced_class_weight=True,
        use_loocv=True,
        save_confusion_matrices=False
    )

    # Run pipeline
    results = pipeline.fit_predict()

    # Save all windows
    print("\nSaving all windows confusion matrices...")
    pipeline.save_confusion_matrices_plots(save_all_windows=True, save_best_only=False)

    # Save best only
    print("\nSaving best models confusion matrices...")
    pipeline.save_confusion_matrices_plots(save_all_windows=False, save_best_only=True)

    print("\nBoth types saved!")
    print("Files created:")
    print("  - confusion_matrices_self_harm_balanced_loocv.png")
    print("  - confusion_matrices_self_harm_balanced_loocv_best.png")


if __name__ == "__main__":
    # Run example 1: Automatic saving (recommended for most use cases)
    example_automatic_save()

    # Uncomment to run other examples:
    # example_manual_save()
    # example_custom_save()

