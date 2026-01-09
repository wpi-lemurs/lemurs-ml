"""
Model training script for suicide risk prediction using hourly screentime features.
Experiments with different time windows (n hours before survey) to find optimal prediction window.
"""

from src.data_processing.merge_passive_data_and_labels import merge_daily_screentime_features_with_suicide_risk, export_as_csv
from src.categorization.suicide_risk_labels import get_suicide_risk_dataframe
from src.config import DATA_DIR
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# Use centralized data directory
data_dir = DATA_DIR

def train_and_evaluate_models(data, time_window):
    """
    Train and evaluate models for a specific time window.

    Parameters:
    - data: DataFrame with hourly screentime features and suicide risk labels
    - time_window: number of hours before survey (for reporting)

    Returns:
    - Dictionary with model performance metrics
    """
    if data.empty:
        print(f"  No data found for {time_window} hour window")
        return None

    print(f"\nData Summary:")
    print(f"  Total rows: {len(data)}")
    print(f"  Unique users: {data['app_user_id'].nunique()}")
    print(f"  Label distribution:")
    print(f"    {data['suicide_risk_label'].value_counts().to_dict()}")

    # Export each time window separately
    export_as_csv(data, f'daily_screentime_suicide_risk_{time_window}h.csv')

    # Try to train a simple model if we have enough data
    if len(data) < 10:  # Need at least 10 samples
        print(f"    Insufficient data for modeling (need at least 10 samples, have {len(data)})")
        return None

    print(f"\n  Training models for {time_window}-hour window...")

    # Prepare features and labels
    hour_cols = [f'hour_{i}' for i in range(time_window)]
    X = data[hour_cols]
    y = data['suicide_risk_label']

    # Check if we have both classes
    if y.nunique() < 2:
        print(f"    WARNING: Only one class present ({y.unique()[0]}). Cannot train model.")
        return None

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if y.value_counts().min() >= 2 else None
    )

    print(f"    Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    print(f"    Train labels: {y_train.value_counts().to_dict()}")
    print(f"    Test labels: {y_test.value_counts().to_dict()}")

    results = {
        'time_window': time_window,
        'total_samples': len(data),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
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
        y_test_binary = (y_test == 'at_risk').values.astype(int)
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
        y_test_binary = (y_test == 'at_risk').values.astype(int)
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



def main():
    """
    Main function to experiment with different time windows.
    Trains models on screentime data from n hours before each survey to predict suicide risk.
    """
    print("="*80)
    print("SUICIDE RISK PREDICTION - HOURLY SCREENTIME FEATURES")
    print("Experimenting with different time windows for screentime before surveys...")
    print("="*80)

    # Get suicide risk data
    suicide_risk_data = get_suicide_risk_dataframe()

    # Time windows to experiment with (in hours)
    time_windows = [3, 6, 9, 12, 24]
    all_results = []

    for hours in time_windows:
        print(f"\n{'='*80}")
        print(f"TIME WINDOW: {hours} hours before survey")
        print(f"{'='*80}")

        # Get data for this time window
        screentime_risk_data = merge_daily_screentime_features_with_suicide_risk(
            screentime_df=None,
            suicide_risk_df=suicide_risk_data,
            fill_method='zero',
            hours_before_survey=hours,
            app_user_id=-1
        )

        # Train and evaluate models
        results = train_and_evaluate_models(screentime_risk_data, hours)
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
        output_path = data_dir / 'time_window_comparison_results.csv'
        comparison_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    else:
        print("\nNo results to compare. Insufficient data for all time windows.")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)



if __name__ == '__main__':
    main()

