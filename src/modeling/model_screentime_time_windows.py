"""
Model training script for mental health prediction using hourly screentime features.
Experiments with different time windows (n hours before survey) to find optimal prediction window.
Supports both PHQ-9 depression prediction and suicide risk prediction.
"""

from src.data_processing.merge_passive_data_and_labels import (
    merge_daily_screentime_features_with_suicide_risk,
    merge_daily_screentime_features_with_phq9,
    export_as_csv,
    propagate_positive_labels
)
from src.config import DATA_DIR
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# Use centralized data directory
data_dir = DATA_DIR

def train_and_evaluate_models(data, time_window, target_type='phq9', propagate_labels=False):
    """
    Train and evaluate models for a specific time window.

    Parameters:
    - data: DataFrame with hourly screentime features and target labels
    - time_window: number of hours before survey (for reporting)
    - target_type: 'phq9' for depression prediction or 'suicide_risk' for suicide risk prediction
    - propagate_labels: if True, propagate positive labels to all entries for users with at least one positive label

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
    else:  # suicide_risk
        label_col = 'suicide_risk_label'
        positive_class = 'at_risk'
        output_prefix = 'daily_screentime_suicide_risk'
        prediction_task = 'Suicide Risk'

    print(f"\nData Summary:")
    print(f"  Total rows: {len(data)}")
    print(f"  Unique users: {data['app_user_id'].nunique()}")
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

    results = {
        'time_window': time_window,
        'total_samples': len(data),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'target_type': target_type
    }

    # Train Logistic Regression
    print(f"\n    Logistic Regression:")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    print(f"      Accuracy: {lr_acc:.4f}")
    print(f"      Classification Report:")
    print("      " + "\n      ".join(classification_report(y_test, lr_pred).split('\n')))

    results['lr_accuracy'] = lr_acc
    try:
        lr_prob = lr_model.predict_proba(X_test)[:, 1]
        y_test_binary = (y_test == positive_class).values.astype(int)
        results['lr_roc_auc'] = roc_auc_score(y_test_binary, lr_prob)
    except:
        results['lr_roc_auc'] = None

    # Train Random Forest
    print(f"\n    Random Forest:")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"      Accuracy: {rf_acc:.4f}")
    print(f"      Classification Report:")
    print("      " + "\n      ".join(classification_report(y_test, rf_pred).split('\n')))

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

    return results



def main(target_type='phq9', propagate_labels=False):
    """
    Main function to experiment with different time windows.
    Trains models on screentime data from n hours before each survey to predict mental health outcomes.

    Parameters:
    - target_type: 'phq9' for depression prediction or 'suicide_risk' for suicide risk prediction
    - propagate_labels: if True, propagate positive labels to all entries for users with at least one positive label
    """
    if target_type == 'phq9':
        task_name = "DEPRESSION PREDICTION (PHQ-9)"
        merge_function = merge_daily_screentime_features_with_phq9
    else:
        task_name = "SUICIDE RISK PREDICTION"
        merge_function = merge_daily_screentime_features_with_suicide_risk

    print("="*80)
    print(f"{task_name} - HOURLY SCREENTIME FEATURES")
    print("Experimenting with different time windows for screentime before surveys...")
    if propagate_labels:
        print("NOTE: Label propagation is ENABLED - users with any positive label will have all entries labeled positive")
    print("="*80)

    # Time windows to experiment with (in hours)
    time_windows = [6,7,8,9]
    all_results = []

    for hours in time_windows:
        print(f"\n{'='*80}")
        print(f"TIME WINDOW: {hours} hours before survey")
        print(f"{'='*80}")

        # Get data for this time window using the appropriate merge function
        screentime_data = merge_function(
            screentime_df=None,
            fill_method='zero',
            hours_before_survey=hours,
            app_user_id=-1
        )

        # Train and evaluate models
        results = train_and_evaluate_models(screentime_data, hours, target_type=target_type, propagate_labels=propagate_labels)
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
        output_filename = f'time_window_comparison_{target_type}_results.csv'
        output_path = data_dir / output_filename
        comparison_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    else:
        print("\nNo results to compare. Insufficient data for all time windows.")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)



if __name__ == '__main__':
    import sys

    # Check if a target type is provided as a command line argument
    if len(sys.argv) > 1:
        target = sys.argv[1].lower()
        if target in ['phq9', 'suicide_risk']:
            main(target_type=target)
        else:
            print(f"Invalid target type: {target}")
            print("Valid options: 'phq9' or 'suicide_risk'")
            print("Using default: 'phq9'")
            main(target_type='phq9')
    else:
        # Default to PHQ-9 prediction
        main(target_type='phq9')

