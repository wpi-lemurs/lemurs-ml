"""
Model training script for mental health prediction using hourly screentime features.
Experiments with different time windows (n hours before survey) to find optimal prediction window.
Supports multiple prediction targets:
- PHQ-9 depression prediction (phq9)
- Suicide risk prediction (suicide_risk)
- Self-harm risk prediction (self_harm)
- Sleep risk prediction (sleep)

Usage:
    python model_screentime_time_windows.py phq9          # Depression prediction
    python model_screentime_time_windows.py suicide_risk  # Suicide risk prediction
    python model_screentime_time_windows.py self_harm     # Self-harm risk prediction
    python model_screentime_time_windows.py sleep         # Sleep risk prediction
"""

from src.data_processing.merge_passive_data_and_labels import (
    merge_daily_screentime_features_with_suicide_risk,
    merge_daily_screentime_features_with_phq9,
    merge_daily_screentime_features_with_risk_labels,
    export_as_csv,
    propagate_positive_labels
)
from src.config import DATA_DIR
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Set backend for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Use centralized data directory
data_dir = DATA_DIR

def train_and_evaluate_models(data, time_window, target_type='phq9', propagate_labels=False, balanced_class_weight=False):
    """
    Train and evaluate models for a specific time window.

    Parameters:
    - data: DataFrame with hourly screentime features and target labels
    - time_window: number of hours before survey (for reporting)
    - target_type: 'phq9' for depression prediction, 'suicide_risk' for suicide risk,
                   'self_harm' for self-harm risk, or 'sleep' for sleep risk prediction
    - propagate_labels: if True, propagate positive labels to all entries for users with at least one positive label
    - balanced_class_weight: if True, use the class_weight = 'balanced' hyperparameter for the RF and LR models

    Returns:
    - Dictionary with model performance metrics
    """
    if data.empty:
        print(f"  No data found for {time_window} hour window")
        return None

    # Determine label column based on target type
    if target_type == 'phq9':
        label_col = 'severity_label'
        positive_class = 'depressed'
        output_prefix = 'daily_screentime_phq9'
        prediction_task = 'Depression'
    elif target_type == 'suicide_risk':
        label_col = 'suicide_risk_label'
        positive_class = 'at_risk'
        output_prefix = 'daily_screentime_suicide_risk'
        prediction_task = 'Suicide Risk'
    elif target_type == 'self_harm':
        label_col = 'self_harm_risk_label'
        positive_class = 'at_risk'
        output_prefix = 'daily_screentime_self_harm'
        prediction_task = 'Self-Harm Risk'
    elif target_type == 'sleep':
        label_col = 'sleep_label'
        positive_class = 'at_risk'
        output_prefix = 'daily_screentime_sleep'
        prediction_task = 'Sleep Risk'
    else:
        raise ValueError(f"Invalid target_type: {target_type}. Must be 'phq9', 'suicide_risk', 'self_harm', or 'sleep'")

    class_weight = 'balanced' if balanced_class_weight else None

    print(f"\nData Summary:")
    print(f"  Total rows: {len(data)}")
    print(f"  Unique users: {data['app_user_id'].nunique()}")
    print(f"  Class balancing: {'ENABLED (class_weight=balanced)' if balanced_class_weight else 'DISABLED (class_weight=None)'}")
    print(f"  Label distribution (before propagation):")
    print(f"    {data[label_col].value_counts().to_dict()}")

    # Apply label propagation if requested
    if propagate_labels:
        print(f"\n  Applying label propagation for users with at least one '{positive_class}' label...")
        data = propagate_positive_labels(data, label_col, positive_class)

    # Export each time window separately
    export_as_csv(data, f'{output_prefix}_{time_window}h.csv')

    # Check if we have both classes
    if data[label_col].nunique() < 2:
        print(f"    WARNING: Only one class present ({data[label_col].unique()[0]}). Cannot train model.")
        return None

    # Check minimum samples per class
    class_counts = data[label_col].value_counts()
    min_class_count = class_counts.min()

    if min_class_count < 2:
        print(f"    WARNING: Minority class has only {min_class_count} sample(s). Need at least 2 per class for train/test split.")
        return None

    if len(data) < 10:
        print(f"    Insufficient data for modeling (need at least 10 samples, have {len(data)})")
        return None

    print(f"\n  Training models for {time_window}-hour window ({prediction_task} prediction)...")

    # Prepare features and labels
    hour_cols = [f'hour_{i}' for i in range(time_window)]
    X = data[hour_cols]
    y = data[label_col]

    # Split data with stratification
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    except ValueError as e:
        print(f"    WARNING: Cannot stratify split (likely too few samples in minority class). Using random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=None
        )

    # Final check: ensure both train and test have both classes
    if y_train.nunique() < 2:
        print(f"    WARNING: Training set only has one class. Cannot train model.")
        return None

    if y_test.nunique() < 2:
        print(f"    WARNING: Test set only has one class. Results may not be meaningful.")

    print(f"    Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    print(f"    Train labels: {y_train.value_counts().to_dict()}")
    print(f"    Test labels: {y_test.value_counts().to_dict()}")

    # Calculate and display class weights if balancing is enabled
    if balanced_class_weight:
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        print(f"\n    Class weights being applied:")
        for label, weight in class_weight_dict.items():
            count = (y_train == label).sum()
            print(f"      {label}: {weight:.4f} (n={count})")
        print(f"    Note: Higher weights are applied to minority class to balance the training")

    results = {
        'time_window': time_window,
        'total_samples': len(data),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'target_type': target_type
    }

    # Train Logistic Regression
    print(f"\n    Logistic Regression:")
    if balanced_class_weight:
        print(f"      Training with class_weight='balanced' to handle class imbalance...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weight)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    print(f"      Accuracy: {lr_acc:.4f}")
    print(f"      Classification Report:")
    print("      " + "\n      ".join(classification_report(y_test, lr_pred).split('\n')))

    # Generate confusion matrix with explicit label order
    # Determine label order based on target type (to match visualization)
    if target_type == 'phq9':
        label_order = ['not_depressed', 'depressed']
    else:
        label_order = ['not_at_risk', 'at_risk']

    lr_cm = confusion_matrix(y_test, lr_pred, labels=label_order)
    print(f"      Confusion Matrix:")
    print(f"      {lr_cm}")
    results['lr_confusion_matrix'] = lr_cm.tolist()

    results['lr_accuracy'] = lr_acc
    try:
        lr_prob = lr_model.predict_proba(X_test)[:, 1]
        y_test_binary = (y_test == positive_class).values.astype(int)
        results['lr_roc_auc'] = roc_auc_score(y_test_binary, lr_prob)
    except:
        results['lr_roc_auc'] = None

    # Train Random Forest
    print(f"\n    Random Forest:")
    if balanced_class_weight:
        print(f"      Training with class_weight='balanced' to handle class imbalance...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"      Accuracy: {rf_acc:.4f}")
    print(f"      Classification Report:")
    print("      " + "\n      ".join(classification_report(y_test, rf_pred).split('\n')))

    # Generate confusion matrix with explicit label order (same as LR above)
    rf_cm = confusion_matrix(y_test, rf_pred, labels=label_order)
    print(f"      Confusion Matrix:")
    print(f"      {rf_cm}")
    results['rf_confusion_matrix'] = rf_cm.tolist()

    results['rf_accuracy'] = rf_acc
    try:
        rf_prob = rf_model.predict_proba(X_test)[:, 1]
        y_test_binary = (y_test == positive_class).values.astype(int)
        results['rf_roc_auc'] = roc_auc_score(y_test_binary, rf_prob)
    except:
        results['rf_roc_auc'] = None

    # Feature importance for Random Forest
    print(f"\n    Top 5 Most Important Hours (Random Forest):")
    feature_importance = pd.DataFrame({
        'hour': hour_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    for idx, row in feature_importance.head(5).iterrows():
        print(f"      {row['hour']}: {row['importance']:.4f}")

    results['top_features'] = feature_importance.head(5).to_dict('records')

    # Prediction distribution analysis
    print(f"\n    Prediction Distribution Analysis:")
    lr_pred_counts = dict(zip(*np.unique(lr_pred, return_counts=True)))
    rf_pred_counts = dict(zip(*np.unique(rf_pred, return_counts=True)))
    print(f"      Logistic Regression predictions: {lr_pred_counts}")
    print(f"      Random Forest predictions: {rf_pred_counts}")
    print(f"      Actual test labels: {y_test.value_counts().to_dict()}")

    if balanced_class_weight:
        print(f"\n    Class Balancing Summary:")
        print(f"      Class weights were applied to help the model learn from minority class.")
        print(f"      This increases the cost of misclassifying the minority class during training.")
        print(f"      Effect: Models are encouraged to predict the minority class more often.")

    return results


def plot_confusion_matrices(all_results, target_type='phq9', balanced_class_weight=False):
    """
    Plot confusion matrices for all time windows.

    Parameters:
    - all_results: List of result dictionaries from train_and_evaluate_models
    - target_type: Type of target prediction
    - balanced_class_weight: Whether class balancing was used
    """
    if not all_results:
        return

    # Define balanced suffix early so it's available throughout the function
    balanced_suffix = '_balanced' if balanced_class_weight else ''

    # Determine label names based on target type
    if target_type == 'phq9':
        labels = ['not_depressed', 'depressed']
    else:
        labels = ['not_at_risk', 'at_risk']

    # Create figure with subplots for each time window
    n_windows = len(all_results)
    fig, axes = plt.subplots(n_windows, 2, figsize=(14, 5 * n_windows))

    # Handle case of single time window
    if n_windows == 1:
        axes = axes.reshape(1, -1)

    class_weight_title = " (Balanced Class Weights)" if balanced_class_weight else ""

    fig.suptitle(f'Confusion Matrices - {target_type.replace("_", " ").title()} Prediction{class_weight_title}',
                 fontsize=16, fontweight='bold', y=0.995)

    for idx, result in enumerate(all_results):
        time_window = result['time_window']

        # Plot Logistic Regression confusion matrix
        if 'lr_confusion_matrix' in result:
            lr_cm = np.array(result['lr_confusion_matrix'])
            sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels,
                       ax=axes[idx, 0], cbar=True)
            axes[idx, 0].set_title(f'Logistic Regression - {time_window}h window\n'
                                  f'Accuracy: {result["lr_accuracy"]:.3f}')
            axes[idx, 0].set_ylabel('True Label')
            axes[idx, 0].set_xlabel('Predicted Label')

        # Plot Random Forest confusion matrix
        if 'rf_confusion_matrix' in result:
            rf_cm = np.array(result['rf_confusion_matrix'])
            sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens',
                       xticklabels=labels, yticklabels=labels,
                       ax=axes[idx, 1], cbar=True)
            axes[idx, 1].set_title(f'Random Forest - {time_window}h window\n'
                                  f'Accuracy: {result["rf_accuracy"]:.3f}')
            axes[idx, 1].set_ylabel('True Label')
            axes[idx, 1].set_xlabel('Predicted Label')

    plt.tight_layout()

    # Save figure
    output_filename = f'confusion_matrices_{target_type}{balanced_suffix}.png'
    output_path = data_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrices visualization saved to: {output_path}")
    plt.close()  # Close the figure to free memory

    # Also create a summary plot showing only the best performing window
    if all_results:
        # Find best performing window by accuracy
        best_lr_idx = max(range(len(all_results)), key=lambda i: all_results[i]['lr_accuracy'])
        best_rf_idx = max(range(len(all_results)), key=lambda i: all_results[i]['rf_accuracy'])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Best Performing Models - {target_type.replace("_", " ").title()} Prediction',
                     fontsize=16, fontweight='bold')

        # Best Logistic Regression
        best_lr = all_results[best_lr_idx]
        lr_cm = np.array(best_lr['lr_confusion_matrix'])
        sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels,
                   ax=axes[0], cbar=True, annot_kws={'size': 14})
        axes[0].set_title(f'Best Logistic Regression\n{best_lr["time_window"]}h window - '
                         f'Accuracy: {best_lr["lr_accuracy"]:.3f}', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=11)
        axes[0].set_xlabel('Predicted Label', fontsize=11)

        # Best Random Forest
        best_rf = all_results[best_rf_idx]
        rf_cm = np.array(best_rf['rf_confusion_matrix'])
        sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens',
                   xticklabels=labels, yticklabels=labels,
                   ax=axes[1], cbar=True, annot_kws={'size': 14})
        axes[1].set_title(f'Best Random Forest\n{best_rf["time_window"]}h window - '
                         f'Accuracy: {best_rf["rf_accuracy"]:.3f}', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=11)
        axes[1].set_xlabel('Predicted Label', fontsize=11)

        plt.tight_layout()

        # Save best models figure
        best_output_filename = f'confusion_matrices_{target_type}{balanced_suffix}_best.png'
        best_output_path = data_dir / best_output_filename
        plt.savefig(best_output_path, dpi=300, bbox_inches='tight')
        print(f"Best models confusion matrices saved to: {best_output_path}")
        plt.close()  # Close the figure to free memory



def main(target_type='phq9', propagate_labels=False, balanced_class_weight=False):
    """
    Main function to experiment with different time windows.
    Trains models on screentime data from n hours before each survey to predict mental health outcomes.

    Parameters:
    - target_type: 'phq9' for depression prediction, 'suicide_risk' for suicide risk,
                   'self_harm' for self-harm risk, or 'sleep' for sleep risk prediction
    - propagate_labels: if True, propagate positive labels to all entries for users with at least one positive label
    - balanced_class_weight: if True, use the class_weight = 'balanced' hyperparameter for the RF and LR models
    """
    # Configure based on target type
    label_column = None  # Initialize to avoid reference before assignment
    if target_type == 'phq9':
        task_name = "DEPRESSION PREDICTION (PHQ-9)"
        merge_function = merge_daily_screentime_features_with_phq9
        use_generic = False
    elif target_type == 'suicide_risk':
        task_name = "SUICIDE RISK PREDICTION"
        merge_function = None
        label_column = 'suicide_risk_label'
        use_generic = True
    elif target_type == 'self_harm':
        task_name = "SELF-HARM RISK PREDICTION"
        merge_function = None
        label_column = 'self_harm_risk_label'
        use_generic = True
    elif target_type == 'sleep':
        task_name = "SLEEP RISK PREDICTION"
        merge_function = None
        label_column = 'sleep_label'
        use_generic = True
    else:
        raise ValueError(f"Invalid target_type: {target_type}. Must be 'phq9', 'suicide_risk', 'self_harm', or 'sleep'")

    print("="*80)
    print(f"{task_name} - HOURLY SCREENTIME FEATURES")
    print("Experimenting with different time windows for screentime before surveys...")
    if propagate_labels:
        print("NOTE: Label propagation is ENABLED - users with any positive label will have all entries labeled positive")
    print("="*80)

    # Time windows to experiment with (in hours)
    time_windows = [3,4,5,6,7,8,9]
    all_results = []

    for hours in time_windows:
        print(f"\n{'='*80}")
        print(f"TIME WINDOW: {hours} hours before survey")
        print(f"{'='*80}")

        # Get data for this time window using the appropriate merge function
        if use_generic:
            # Use the new generic function for risk labels
            screentime_data = merge_daily_screentime_features_with_risk_labels(
                screentime_df=None,
                risk_labels_df=None,
                label_column=label_column,
                fill_method='zero',
                hours_before_survey=hours,
                app_user_id=-1
            )
        else:
            # Use the PHQ-9 specific function
            screentime_data = merge_function(
                screentime_df=None,
                fill_method='zero',
                hours_before_survey=hours,
                app_user_id=-1
            )

        # Train and evaluate models
        results = train_and_evaluate_models(screentime_data, hours, target_type=target_type, propagate_labels=propagate_labels, balanced_class_weight=balanced_class_weight)
        if results:
            all_results.append(results)

    # Print comparison summary
    if all_results:
        print("\n" + "="*80)
        print("COMPARISON SUMMARY - ALL TIME WINDOWS")
        print("="*80)

        comparison_df = pd.DataFrame(all_results)
        print("\nModel Performance Comparison:")
        display_cols = ['time_window', 'total_samples', 'lr_accuracy', 'rf_accuracy']
        if 'lr_roc_auc' in comparison_df.columns:
            display_cols.extend(['lr_roc_auc', 'rf_roc_auc'])
        print(comparison_df[display_cols].to_string(index=False))

        # Find best performing window
        if 'lr_roc_auc' in comparison_df.columns and comparison_df['lr_roc_auc'].notna().any():
            best_lr_window = comparison_df.loc[comparison_df['lr_roc_auc'].idxmax()]
            best_rf_window = comparison_df.loc[comparison_df['rf_roc_auc'].idxmax()]

            print(f"\n" + "="*80)
            print("BEST PERFORMING TIME WINDOWS")
            print("="*80)
            print(f"\nLogistic Regression:")
            print(f"  Best window: {best_lr_window['time_window']}h")
            print(f"  ROC-AUC: {best_lr_window['lr_roc_auc']:.4f}")
            print(f"  Accuracy: {best_lr_window['lr_accuracy']:.4f}")

            print(f"\nRandom Forest:")
            print(f"  Best window: {best_rf_window['time_window']}h")
            print(f"  ROC-AUC: {best_rf_window['rf_roc_auc']:.4f}")
            print(f"  Accuracy: {best_rf_window['rf_accuracy']:.4f}")
        else:
            best_lr_window = comparison_df.loc[comparison_df['lr_accuracy'].idxmax()]
            best_rf_window = comparison_df.loc[comparison_df['rf_accuracy'].idxmax()]

            print(f"\n" + "="*80)
            print("BEST PERFORMING TIME WINDOWS (by accuracy)")
            print("="*80)
            print(f"\nLogistic Regression:")
            print(f"  Best window: {best_lr_window['time_window']}h")
            print(f"  Accuracy: {best_lr_window['lr_accuracy']:.4f}")

            print(f"\nRandom Forest:")
            print(f"  Best window: {best_rf_window['time_window']}h")
            print(f"  Accuracy: {best_rf_window['rf_accuracy']:.4f}")

        # Save results to CSV
        balanced_suffix = '_balanced' if balanced_class_weight else ''
        output_filename = f'time_window_comparison_{target_type}_results{balanced_suffix}.csv'
        output_path = data_dir / output_filename
        comparison_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

        # Generate confusion matrix visualizations
        print(f"\n{'='*80}")
        print("GENERATING CONFUSION MATRIX VISUALIZATIONS")
        print(f"{'='*80}")
        plot_confusion_matrices(all_results, target_type=target_type, balanced_class_weight=balanced_class_weight)
    else:
        print("\nNo results to compare. Insufficient data for all time windows.")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)



if __name__ == '__main__':
    import sys

    # Check if a target type is provided as a command line argument
    target_type = 'phq9'
    propagate_labels = False
    balanced_class_weight = False

    # Parse command line arguments
    if len(sys.argv) > 1:
        target = sys.argv[1].lower()
        if target in ['phq9', 'suicide_risk', 'self_harm', 'sleep']:
            target_type = target
        else:
            print(f"Invalid target type: {target}")
            print("Valid options: 'phq9', 'suicide_risk', 'self_harm', or 'sleep'")
            print("Using default: 'phq9'")

    # Check for optional flags
    if '--propagate' in sys.argv or '-p' in sys.argv:
        propagate_labels = True
        print("Label propagation enabled")

    if '--balanced' in sys.argv or '-b' in sys.argv:
        balanced_class_weight = True
        print("Class balancing enabled")

    # Run main with parsed arguments
    main(target_type=target_type, propagate_labels=propagate_labels, balanced_class_weight=balanced_class_weight)

