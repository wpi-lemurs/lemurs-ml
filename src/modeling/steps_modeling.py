import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from src.data_processing.passive_data_analysis import daily_health_with_week
from src.categorization.daily_questions_categorization import get_daily_labels_dataframe
from src.config import DATA_DIR

data_dir = DATA_DIR

# Available daily labels to model
AVAILABLE_LABELS = {
    'suicide_risk': 'suicide_risk_label',
    'self_harm': 'self_harm_risk_label',
    'positive_emotion': 'positive_emotion_label',
    'negative_emotion': 'negative_emotion_label',
    'social_stress': 'social_stress_label',
    'social_connection': 'social_connection_label',
    'minority_stress': 'minority_stress_label',
    'emotion_regulation': 'emotion_regulation_label',
    'sleep': 'sleep_label'
}


def prepare_daily_steps_with_labels(target_label='social_connection', week_anchor='MON'):
    """
    Merge daily steps data with daily risk labels.

    Parameters:
    - target_label: which label to model (key from AVAILABLE_LABELS)
    - week_anchor: weekday anchor for weekly grouping (default 'MON')

    Returns:
    - DataFrame with daily steps and the target label
    """
    if target_label not in AVAILABLE_LABELS:
        raise ValueError(f"Invalid target_label. Choose from: {list(AVAILABLE_LABELS.keys())}")

    # Get daily health data (steps, distance, calories, speed)
    daily_health = daily_health_with_week(week_anchor=week_anchor, fill_method=None)

    if daily_health.empty:
        raise ValueError("No daily health data available")

    # Get daily labels
    daily_labels = get_daily_labels_dataframe()

    if daily_labels.empty:
        raise ValueError("No daily labels available")

    # Parse timestamps to datetime
    daily_health['date'] = pd.to_datetime(daily_health['date'])
    daily_labels['timestamp'] = pd.to_datetime(daily_labels['timestamp'])

    # Extract date from daily_labels timestamp for matching
    # Convert to datetime.date for consistent matching
    daily_labels['date'] = daily_labels['timestamp'].dt.date
    daily_health['date'] = daily_health['date'].dt.date

    # Merge on app_user_id and date
    merged = pd.merge(
        daily_health,
        daily_labels[['app_user_id', 'date', AVAILABLE_LABELS[target_label]]],
        on=['app_user_id', 'date'],
        how='inner'
    )

    # Filter out N/A labels (mainly for sleep data which has N/A for afternoon surveys)
    label_col = AVAILABLE_LABELS[target_label]
    merged = merged[merged[label_col] != 'N/A'].copy()

    if merged.empty:
        raise ValueError(f"No valid data for label: {target_label}")

    # Check if we have both classes
    if merged[label_col].nunique() < 2:
        raise ValueError(f"Insufficient class diversity for label: {target_label}")

    return merged, label_col


def run_model_for_label(target_label='social_connection'):
    """
    Complete pipeline: prepare data, train models, evaluate, and visualize results.
    """
    print(f"\n{'='*80}")
    print(f"MODELING: {target_label.upper()}")
    print(f"{'='*80}\n")

    # Prepare data
    modeling_data, label_col = prepare_daily_steps_with_labels(target_label)

    print("="*80)
    print(f"{target_label.upper().replace('_', ' ')} PREDICTION MODEL - Steps Data")
    print("="*80)
    print(f"\nDataset shape before cleaning: {modeling_data.shape}")

    # Prepare features and target
    # Use all available health metrics: steps, distance, calories, speed
    feature_cols = ['daily_steps', 'daily_distance', 'daily_calories', 'daily_avg_speed']
    available_features = [col for col in feature_cols if col in modeling_data.columns]

    # Drop rows with NaN values in features or label
    modeling_data = modeling_data.dropna(subset=available_features + [label_col])

    if modeling_data.empty:
        raise ValueError(f"No valid data after removing NaN values for label: {target_label}")

    print(f"Dataset shape after cleaning: {modeling_data.shape}")
    print(f"\nTarget distribution:")
    print(modeling_data[label_col].value_counts())
    print(f"\nPercentage:")
    print(modeling_data[label_col].value_counts(normalize=True) * 100)

    X = modeling_data[available_features].values
    y = modeling_data[label_col].values

    # Split data into train and test sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression model
    print("\n" + "="*80)
    print("LOGISTIC REGRESSION MODEL")
    print("="*80)

    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)

    # Evaluate Logistic Regression
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print("\nAccuracy:", acc_lr)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_lr))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_lr))

    # Calculate F1 score
    f1_lr = f1_score(y_test, y_pred_lr, average='weighted')
    print(f"\nWeighted F1 Score: {f1_lr:.4f}")

    # Calculate ROC AUC if we have both classes in test set
    if len(np.unique(y_test)) > 1:
        try:
            # Determine positive class
            positive_class = 'at_risk' if 'at_risk' in lr_model.classes_ else lr_model.classes_[1]
            pos_idx = list(lr_model.classes_).index(positive_class)
            roc_auc_lr = roc_auc_score(y_test == positive_class, y_pred_proba_lr[:, pos_idx])
            print(f"ROC AUC Score: {roc_auc_lr:.4f}")
        except Exception as e:
            print(f"Could not calculate ROC AUC: {e}")

    # Train Random Forest model
    print("\n" + "="*80)
    print("RANDOM FOREST MODEL")
    print("="*80)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf_model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred_rf = rf_model.predict(X_test_scaled)
    y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)

    # Evaluate Random Forest
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print("\nAccuracy:", acc_rf)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))

    # Calculate F1 score
    f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
    print(f"\nWeighted F1 Score: {f1_rf:.4f}")

    # Calculate ROC AUC
    if len(np.unique(y_test)) > 1:
        try:
            positive_class = 'at_risk' if 'at_risk' in rf_model.classes_ else rf_model.classes_[1]
            pos_idx = list(rf_model.classes_).index(positive_class)
            roc_auc_rf = roc_auc_score(y_test == positive_class, y_pred_proba_rf[:, pos_idx])
            print(f"ROC AUC Score: {roc_auc_rf:.4f}")
        except Exception as e:
            print(f"Could not calculate ROC AUC: {e}")

    # Feature importance for Random Forest
    print("\nFeature Importance:")
    if hasattr(rf_model, 'feature_importances_'):
        for name, importance in zip(available_features, rf_model.feature_importances_):
            print(f"  {name}: {importance:.4f}")

    # Summary statistics by predicted class
    print("\n" + "="*80)
    print("PREDICTION ANALYSIS")
    print("="*80)

    test_df = pd.DataFrame({
        'daily_steps': X_test[:, 0],  # First feature is steps
        'actual': y_test,
        'predicted_lr': y_pred_lr,
        'predicted_rf': y_pred_rf
    })

    print("\nAverage steps by actual label:")
    print(test_df.groupby('actual')['daily_steps'].agg(['mean', 'std', 'min', 'max']))

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n{target_label.upper()} Prediction Results:")
    print(f"Logistic Regression - Test Accuracy: {acc_lr:.2%}, F1: {f1_lr:.4f}")
    print(f"Random Forest - Test Accuracy: {acc_rf:.2%}, F1: {f1_rf:.4f}")
    print(f"Best Model: {'Logistic Regression' if acc_lr >= acc_rf else 'Random Forest'}")

    # Confusion Matrix Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.heatmap(confusion_matrix(y_test, y_pred_lr),
                annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title("Logistic Regression – Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(confusion_matrix(y_test, y_pred_rf),
                annot=True, fmt='d', cmap='Purples', ax=axes[1])
    axes[1].set_title("Random Forest – Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    filename = data_dir / f"confusion_matrices_{target_label}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {filename}")
    plt.show()

    # ROC Curve Plot
    if len(np.unique(y_test)) > 1:
        try:
            positive_class_lr = 'at_risk' if 'at_risk' in lr_model.classes_ else lr_model.classes_[1]
            pos_idx_lr = list(lr_model.classes_).index(positive_class_lr)

            positive_class_rf = 'at_risk' if 'at_risk' in rf_model.classes_ else rf_model.classes_[1]
            pos_idx_rf = list(rf_model.classes_).index(positive_class_rf)

            fpr_lr, tpr_lr, _ = roc_curve(y_test == positive_class_lr,
                                          y_pred_proba_lr[:, pos_idx_lr])
            auc_lr = auc(fpr_lr, tpr_lr)

            fpr_rf, tpr_rf, _ = roc_curve(y_test == positive_class_rf,
                                          y_pred_proba_rf[:, pos_idx_rf])
            auc_rf = auc(fpr_rf, tpr_rf)

            plt.figure(figsize=(7, 5))
            plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression AUC = {auc_lr:.3f}")
            plt.plot(fpr_rf, tpr_rf, color="purple", label=f"Random Forest AUC = {auc_rf:.3f}")
            plt.plot([0, 1], [0, 1], 'k--')

            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve – {target_label.replace('_', ' ').title()}")
            plt.legend()
            plt.tight_layout()
            filename = data_dir / f"roc_curve_{target_label}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.show()
        except Exception as e:
            print(f"Could not plot ROC curve: {e}")

    # F1 Score Bar Chart
    plt.figure(figsize=(6, 4))
    f1_bars = plt.bar(['Logistic Regression', 'Random Forest'], [f1_lr, f1_rf])
    plt.ylabel("Weighted F1 Score")
    plt.title(f"F1 Score Comparison – {target_label.replace('_', ' ').title()}")
    plt.ylim(0, 1)
    for bar, val in zip(f1_bars, [f1_lr, f1_rf]):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                 f"{val:.2%}", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    filename = data_dir / f"f1_score_{target_label}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.show()

    # Accuracy Bar Chart
    plt.figure(figsize=(6, 4))
    bars = plt.bar(['Logistic Regression', 'Random Forest'],
                   [acc_lr, acc_rf])
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Comparison – {target_label.replace('_', ' ').title()}")
    plt.ylim(0, 1)
    for bar, val in zip(bars, [acc_lr, acc_rf]):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                 f"{val:.2%}", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    filename = data_dir / f"accuracy_{target_label}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.show()

    # Steps Distribution by Class
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=test_df, x='actual', y='daily_steps')
    plt.title(f"Distribution of Steps by {target_label.replace('_', ' ').title()}")
    plt.xlabel(f"{target_label.replace('_', ' ').title()}")
    plt.ylabel("Daily Steps")
    plt.tight_layout()
    filename = data_dir / f"steps_distribution_{target_label}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.show()

    return {
        'target_label': target_label,
        'lr_accuracy': acc_lr,
        'rf_accuracy': acc_rf,
        'lr_f1': f1_lr,
        'rf_f1': f1_rf,
    }


if __name__ == "__main__":
    # Configuration: Set to None to run all labels, or specify a single label
    SINGLE_LABEL = 'social_connection'  # Change this to run a different label, or set to None to run all

    results = []

    if SINGLE_LABEL:
        # Run for a single label
        try:
            result = run_model_for_label(SINGLE_LABEL)
            results.append(result)
        except Exception as e:
            print(f"\nError modeling {SINGLE_LABEL}: {e}\n")
    else:
        # Run for all available labels
        for label_name in AVAILABLE_LABELS.keys():
            try:
                result = run_model_for_label(label_name)
                results.append(result)
            except Exception as e:
                print(f"\nError modeling {label_name}: {e}\n")
                continue

    # Summary of all labels
    if results:
        print("\n\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))

        # Save results
        results_file = data_dir / "daily_labels_modeling_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\nSaved results to: {results_file}")
