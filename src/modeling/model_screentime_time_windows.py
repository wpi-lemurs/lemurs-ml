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

Optional flags:
    --propagate, -p   Propagate positive labels to all entries for users with any positive label
    --balanced, -b    Use balanced class weights to handle class imbalance
    --loocv, -l       Use leave-one-user-out cross-validation instead of train/test split
"""

from src.data_processing.merge_passive_data_and_labels import (
    merge_daily_screentime_features_with_phq9,
    merge_daily_screentime_features_with_risk_labels,
    propagate_positive_labels
)
from src.config import DATA_DIR
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Set backend for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Use centralized data directory
data_dir = DATA_DIR

def train_and_evaluate_models(data, time_window, target_type='phq9', propagate_labels=False, balanced_class_weight=False, use_loocv=False):
    """
    Train and evaluate models for a specific time window.

    Parameters:
    - data: DataFrame with hourly screentime features and target labels
    - time_window: number of hours before survey (for reporting)
    - target_type: 'phq9' for depression prediction, 'suicide_risk' for suicide risk,
                   'self_harm' for self-harm risk, or 'sleep' for sleep risk prediction
    - propagate_labels: if True, propagate positive labels to all entries for users with at least one positive label
    - balanced_class_weight: if True, use the class_weight = 'balanced' hyperparameter for the RF and LR models
    - use_loocv: if True, use leave-one-out cross-validation by user (train on all users except one, test on held-out user)

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
        # drop N/A risk label rows with afternoon data
        data = data[data['sleep_label'] != 'N/A']
    else:
        raise ValueError(f"Invalid target_type: {target_type}. Must be 'phq9', 'suicide_risk', 'self_harm', or 'sleep'")

    class_weight = 'balanced' if balanced_class_weight else None

    # Apply label propagation if requested
    if propagate_labels:
        data = propagate_positive_labels(data, label_col, positive_class)

    # Check if we have both classes
    if data[label_col].nunique() < 2:
        return None

    # Check minimum samples per class
    class_counts = data[label_col].value_counts()
    min_class_count = class_counts.min()

    if min_class_count < 2:
        return None

    if len(data) < 10:
        return None


    # Prepare features and labels
    hour_cols = [f'hour_{i}' for i in range(time_window)]
    X = data[hour_cols]
    y = data[label_col]

    # Determine label order based on target type (for confusion matrix)
    if target_type == 'phq9':
        label_order = ['not_depressed', 'depressed']
    else:
        label_order = ['not_at_risk', 'at_risk']

    class_weight = 'balanced' if balanced_class_weight else None

    # Check if using LOOCV
    if use_loocv:

        # Initialize LOOCV
        logo = LeaveOneGroupOut()
        groups = data['app_user_id']

        # Store predictions and true labels across all folds
        all_lr_preds = []
        all_rf_preds = []
        all_lr_probs = []
        all_rf_probs = []
        all_y_test = []

        fold_num = 0
        successful_folds = 0

        for train_idx, test_idx in logo.split(X, y, groups):
            fold_num += 1
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            test_user = groups.iloc[test_idx].iloc[0]

            # Skip fold if training or test set has only one class
            if y_train.nunique() < 2:
                continue

            if y_test.nunique() < 1:
                continue

            successful_folds += 1

            # Scale features for Logistic Regression
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train models for this fold
            try:
                # Logistic Regression (with scaled data)
                lr_model = LogisticRegression(max_iter=10000, random_state=42, class_weight=class_weight)
                lr_model.fit(X_train_scaled, y_train)
                lr_pred = lr_model.predict(X_test_scaled)

                # Random Forest (without scaling)
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight)
                rf_model.fit(X_train, y_train)
                rf_pred = rf_model.predict(X_test)

                # Store predictions and true labels
                all_lr_preds.extend(lr_pred)
                all_rf_preds.extend(rf_pred)
                all_y_test.extend(y_test)

                # Store probabilities if possible
                try:
                    lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
                    all_lr_probs.extend(lr_prob)
                except:
                    pass

                try:
                    rf_prob = rf_model.predict_proba(X_test)[:, 1]
                    all_rf_probs.extend(rf_prob)
                except:
                    pass

            except Exception as e:
                continue

        if successful_folds == 0:
            return None


        # Convert to numpy arrays for evaluation
        all_y_test = np.array(all_y_test)
        all_lr_preds = np.array(all_lr_preds)
        all_rf_preds = np.array(all_rf_preds)

        # Calculate metrics across all folds
        lr_acc = accuracy_score(all_y_test, all_lr_preds)
        rf_acc = accuracy_score(all_y_test, all_rf_preds)

        # Generate confusion matrices
        lr_cm = confusion_matrix(all_y_test, all_lr_preds, labels=label_order)
        rf_cm = confusion_matrix(all_y_test, all_rf_preds, labels=label_order)


        # Store results
        results = {
            'time_window': time_window,
            'total_samples': len(data),
            'train_samples': len(data) - int(len(data) / data['app_user_id'].nunique()),  # Approx
            'test_samples': int(len(data) / data['app_user_id'].nunique()),  # Approx
            'target_type': target_type,
            'cv_method': 'LOOCV',
            'successful_folds': successful_folds,
            'lr_accuracy': lr_acc,
            'rf_accuracy': rf_acc,
            'lr_confusion_matrix': lr_cm.tolist(),
            'rf_confusion_matrix': rf_cm.tolist()
        }

        # Calculate F1 score
        try:
            results['lr_f1_score'] = f1_score(all_y_test, all_lr_preds, pos_label=positive_class, average='binary')
        except:
            results['lr_f1_score'] = None

        try:
            results['rf_f1_score'] = f1_score(all_y_test, all_rf_preds, pos_label=positive_class, average='binary')
        except:
            results['rf_f1_score'] = None

        # Feature importance (train on full dataset)
        rf_full = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight)
        rf_full.fit(X, y)
        feature_importance = pd.DataFrame({
            'hour': hour_cols,
            'importance': rf_full.feature_importances_
        }).sort_values('importance', ascending=False)

        results['top_features'] = feature_importance.head(5).to_dict('records')


        return results

    else:
        # Standard train/test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        except ValueError as e:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=None
            )

        # Final check: ensure both train and test have both classes
        if y_train.nunique() < 2:
            return None

        if y_test.nunique() < 2:
            pass  # Continue but note that results may not be meaningful

        results = {
            'time_window': time_window,
            'total_samples': len(data),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'target_type': target_type
        }

        # Train baseline model using previous survey label

        baseline_df = data.sort_values(['app_user_id', 'survey_timestamp'])

        # get label for each user's previous survey
        baseline_df['baseline_pred'] = baseline_df.groupby('app_user_id')[label_col].shift(1)
        # remove first survey (no data before it to use)
        baseline_df = baseline_df.dropna(subset=['baseline_pred'])

        # ONLY evaluate on X_test set
        test_indices = X_test.index
        baseline_df = baseline_df[baseline_df.index.isin(test_indices)]

        baseline_true = baseline_df[label_col].values
        baseline_pred = baseline_df['baseline_pred'].values
        baseline_acc = accuracy_score(baseline_true, baseline_pred)

        # Generate confusion matrix with explicit label order
        baseline_cm = confusion_matrix(baseline_true, baseline_pred, labels=label_order)
        results['baseline_confusion_matrix'] = baseline_cm.tolist()
        results['baseline_accuracy'] = baseline_acc

        try:
            results['baseline_f1_score'] = f1_score(baseline_true, baseline_pred, pos_label=positive_class, average='binary')
        except Exception:
            results['baseline_f1_score'] = None

        # Scale features for Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Logistic Regression (with scaled data)
        lr_model = LogisticRegression(max_iter=10000, random_state=42, class_weight=class_weight)
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        lr_acc = accuracy_score(y_test, lr_pred)

        # Generate confusion matrix with explicit label order
        lr_cm = confusion_matrix(y_test, lr_pred, labels=label_order)
        results['lr_confusion_matrix'] = lr_cm.tolist()

        results['lr_accuracy'] = lr_acc
        try:
            results['lr_f1_score'] = f1_score(y_test, lr_pred, pos_label=positive_class, average='binary')
        except:
            results['lr_f1_score'] = None

        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)

        # Generate confusion matrix with explicit label order (same as LR above)
        rf_cm = confusion_matrix(y_test, rf_pred, labels=label_order)
        results['rf_confusion_matrix'] = rf_cm.tolist()

        results['rf_accuracy'] = rf_acc
        try:
            results['rf_f1_score'] = f1_score(y_test, rf_pred, pos_label=positive_class, average='binary')
        except:
            results['rf_f1_score'] = None

        # Feature importance for Random Forest
        feature_importance = pd.DataFrame({
            'hour': hour_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        results['top_features'] = feature_importance.head(5).to_dict('records')


        return results


def plot_confusion_matrices(all_results, target_type='phq9', balanced_class_weight=False, use_loocv=False):
    """
    Plot confusion matrices for all time windows.

    Parameters:
    - all_results: List of result dictionaries from train_and_evaluate_models
    - target_type: Type of target prediction
    - balanced_class_weight: Whether class balancing was used
    - use_loocv: Whether leave-one-out cross-validation was used
    """
    if not all_results:
        return

    # Define balanced suffix early so it's available throughout the function
    balanced_suffix = '_balanced' if balanced_class_weight else ''
    loocv_suffix = '_loocv' if use_loocv else ''

    # Determine label names based on target type
    if target_type == 'phq9':
        labels = ['not_depressed', 'depressed']
    else:
        labels = ['not_at_risk', 'at_risk']

    # Create figure with subplots for each time window
    n_windows = len(all_results)
    fig, axes = plt.subplots(n_windows, 3, figsize=(14, 5 * n_windows))

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
            # Build title with F1 score if available
            title = f'Logistic Regression - {time_window}h window\nAccuracy: {result["lr_accuracy"]:.3f}'
            if 'lr_f1_score' in result and result['lr_f1_score'] is not None:
                title += f' | F1: {result["lr_f1_score"]:.3f}'
            axes[idx, 0].set_title(title)
            axes[idx, 0].set_ylabel('True Label')
            axes[idx, 0].set_xlabel('Predicted Label')

        # Plot Random Forest confusion matrix
        if 'rf_confusion_matrix' in result:
            rf_cm = np.array(result['rf_confusion_matrix'])
            sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens',
                       xticklabels=labels, yticklabels=labels,
                       ax=axes[idx, 1], cbar=True)
            # Build title with F1 score if available
            title = f'Random Forest - {time_window}h window\nAccuracy: {result["rf_accuracy"]:.3f}'
            if 'rf_f1_score' in result and result['rf_f1_score'] is not None:
                title += f' | F1: {result["rf_f1_score"]:.3f}'
            axes[idx, 1].set_title(title)
            axes[idx, 1].set_ylabel('True Label')
            axes[idx, 1].set_xlabel('Predicted Label')
        
        # Plot Baseline confusion matrix
        if 'baseline_confusion_matrix' in result:
            baseline_cm = np.array(result['baseline_confusion_matrix'])
            sns.heatmap(baseline_cm, annot=True, fmt='d', cmap='Oranges',
                       xticklabels=labels, yticklabels=labels,
                       ax=axes[idx, 2], cbar=True)
            # Build title with F1 score if available
            title = f'Baseline - {time_window}h window\nAccuracy: {result["baseline_accuracy"]:.3f}'
            if 'baseline_f1_score' in result and result['baseline_f1_score'] is not None:
                title += f' | F1: {result["baseline_f1_score"]:.3f}'
            axes[idx, 2].set_title(title)
            axes[idx, 2].set_ylabel('True Label')
            axes[idx, 2].set_xlabel('Predicted Label')

    plt.tight_layout()

    # Save figure
    output_filename = f'confusion_matrices_{target_type}{balanced_suffix}{loocv_suffix}.png'
    output_path = data_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    # Also create a summary plot showing only the best performing window
    if all_results:
        # Find best performing window by f1 score if available, otherwise by accuracy
        best_lr_idx = max(range(len(all_results)), key=lambda i: all_results[i]['lr_f1_score'] if 'lr_f1_score' in all_results[i] and all_results[i]['lr_f1_score'] is not None else all_results[i]['lr_accuracy'])
        best_rf_idx = max(range(len(all_results)), key=lambda i: all_results[i]['rf_f1_score'] if 'rf_f1_score' in all_results[i] and all_results[i]['rf_f1_score'] is not None else all_results[i]['rf_accuracy'])
        best_baseline_idx = max(range(len(all_results)), key=lambda i: all_results[i]['baseline_f1_score'] if 'baseline_f1_score' in all_results[i] and all_results[i]['baseline_f1_score'] is not None else all_results[i]['baseline_accuracy'])

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle(f'Best Performing Models - {target_type.replace("_", " ").title()} Prediction',
                     fontsize=16, fontweight='bold')

        # Best Logistic Regression
        best_lr = all_results[best_lr_idx]
        lr_cm = np.array(best_lr['lr_confusion_matrix'])
        sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels,
                   ax=axes[0], cbar=True, annot_kws={'size': 14})
        # Build title with F1 score if available
        title = f'Best Logistic Regression\n{best_lr["time_window"]}h window - Accuracy: {best_lr["lr_accuracy"]:.3f}'
        if 'lr_f1_score' in best_lr and best_lr['lr_f1_score'] is not None:
            title += f' | F1: {best_lr["lr_f1_score"]:.3f}'
        axes[0].set_title(title, fontsize=12, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=11)
        axes[0].set_xlabel('Predicted Label', fontsize=11)

        # Best Random Forest
        best_rf = all_results[best_rf_idx]
        rf_cm = np.array(best_rf['rf_confusion_matrix'])
        sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens',
                   xticklabels=labels, yticklabels=labels,
                   ax=axes[1], cbar=True, annot_kws={'size': 14})
        # Build title with F1 score if available
        title = f'Best Random Forest\n{best_rf["time_window"]}h window - Accuracy: {best_rf["rf_accuracy"]:.3f}'
        if 'rf_f1_score' in best_rf and best_rf['rf_f1_score'] is not None:
            title += f' | F1: {best_rf["rf_f1_score"]:.3f}'
        axes[1].set_title(title, fontsize=12, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=11)
        axes[1].set_xlabel('Predicted Label', fontsize=11)

        # Best Baseline
        best_baseline = all_results[best_baseline_idx]
        baseline_cm = np.array(best_baseline['baseline_confusion_matrix'])
        sns.heatmap(baseline_cm, annot=True, fmt='d', cmap='Oranges',
                   xticklabels=labels, yticklabels=labels,
                   ax=axes[2], cbar=True, annot_kws={'size': 14})
        # Build title with F1 score if available
        title = f'Baseline\n{best_baseline["time_window"]}h window - Accuracy: {best_baseline["baseline_accuracy"]:.3f}'
        if 'baseline_f1_score' in best_baseline and best_baseline['baseline_f1_score'] is not None:
            title += f' | F1: {best_baseline["baseline_f1_score"]:.3f}'
        axes[2].set_title(title, fontsize=12, fontweight='bold')
        axes[2].set_ylabel('True Label', fontsize=11)
        axes[2].set_xlabel('Predicted Label', fontsize=11)

        plt.tight_layout()

        # Save best models figure
        best_output_filename = f'confusion_matrices_{target_type}{balanced_suffix}{loocv_suffix}_best.png'
        best_output_path = data_dir / best_output_filename
        plt.savefig(best_output_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory



def main(target_type='phq9', propagate_labels=False, balanced_class_weight=False, use_loocv=False):
    """
    Main function to experiment with different time windows.
    Trains models on screentime data from n hours before each survey to predict mental health outcomes.

    Parameters:
    - target_type: 'phq9' for depression prediction, 'suicide_risk' for suicide risk,
                   'self_harm' for self-harm risk, or 'sleep' for sleep risk prediction
    - propagate_labels: if True, propagate positive labels to all entries for users with at least one positive label
    - balanced_class_weight: if True, use the class_weight = 'balanced' hyperparameter for the RF and LR models
    - use_loocv: if True, use leave-one-out cross-validation by user (train on all users except one, test on held-out user)
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
        results = train_and_evaluate_models(screentime_data, hours, target_type=target_type, propagate_labels=propagate_labels, balanced_class_weight=balanced_class_weight, use_loocv=use_loocv)
        if results:
            all_results.append(results)

    # Print comparison summary
    if all_results:
        print("\n" + "="*80)
        print("COMPARISON SUMMARY - ALL TIME WINDOWS")
        print("="*80)

        comparison_df = pd.DataFrame(all_results)
        print("\nModel Performance Comparison:")
        display_cols = ['time_window', 'total_samples', 'lr_accuracy', 'rf_accuracy', 'baseline_accuracy']
        if 'lr_f1_score' in comparison_df.columns:
            display_cols.extend(['lr_f1_score', 'rf_f1_score', 'baseline_f1_score'])
        print(comparison_df[display_cols].to_string(index=False))

        # Find best performing window
        if 'lr_f1_score' in comparison_df.columns and comparison_df['lr_f1_score'].notna().any():
            best_lr_window = comparison_df.loc[comparison_df['lr_f1_score'].idxmax()]
            best_rf_window = comparison_df.loc[comparison_df['rf_f1_score'].idxmax()]

            print(f"\n" + "="*80)
            print("BEST PERFORMING TIME WINDOWS")
            print("="*80)
            print(f"\nLogistic Regression:")
            print(f"  Best window: {best_lr_window['time_window']}h")
            print(f"  F1 Score: {best_lr_window['lr_f1_score']:.4f}")
            print(f"  Accuracy: {best_lr_window['lr_accuracy']:.4f}")

            print(f"\nRandom Forest:")
            print(f"  Best window: {best_rf_window['time_window']}h")
            print(f"  F1 Score: {best_rf_window['rf_f1_score']:.4f}")
            print(f"  Accuracy: {best_rf_window['rf_accuracy']:.4f}")

            print(f"\nBaseline Model:")
            print(f"  F1 Score: {comparison_df['baseline_f1_score'].max()}")
            print(f"  Accuracy: {comparison_df['baseline_accuracy'].max()}")
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

            print(f"\nBaseline Model:")
            print(f"  Accuracy: {comparison_df['baseline_accuracy'].max()}")


        # Generate confusion matrix visualizations
        plot_confusion_matrices(all_results, target_type=target_type, balanced_class_weight=balanced_class_weight, use_loocv=use_loocv)
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
    use_loocv = False

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

    if '--loocv' in sys.argv or '-l' in sys.argv:
        use_loocv = True
        print("Leave-One-Out Cross-Validation enabled")
        
    # Run main with parsed arguments
    main(target_type=target_type, propagate_labels=propagate_labels, balanced_class_weight=balanced_class_weight, use_loocv=use_loocv)

