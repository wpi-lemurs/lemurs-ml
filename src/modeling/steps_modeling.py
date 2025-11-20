import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from src.merge_weekly_health_with_phq9 import merge_weekly_health_with_phq9

modeling_data = merge_weekly_health_with_phq9(steps_only=False)

print("="*80)
print("DEPRESSION PREDICTION MODEL - Steps Data")
print("="*80)
print(f"\nDataset shape: {modeling_data.shape}")
print(f"\nTarget distribution:")
print(modeling_data['severity_label'].value_counts())
print(f"\nPercentage:")
print(modeling_data['severity_label'].value_counts(normalize=True) * 100)

# Prepare features and target
X = modeling_data[['avg_daily_steps']].values
y = modeling_data['severity_label'].values

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
print("\nAccuracy:", accuracy_score(y_test, y_pred_lr))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

# Calculate ROC AUC if we have both classes in test set
if len(np.unique(y_test)) > 1:
    # Get the probability for the positive class (depressed)
    if 'depressed' in lr_model.classes_:
        depressed_idx = list(lr_model.classes_).index('depressed')
        roc_auc_lr = roc_auc_score(y_test == 'depressed', y_pred_proba_lr[:, depressed_idx])
        print(f"\nROC AUC Score: {roc_auc_lr:.4f}")

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
print("\nAccuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# Calculate ROC AUC
if len(np.unique(y_test)) > 1:
    if 'depressed' in rf_model.classes_:
        depressed_idx = list(rf_model.classes_).index('depressed')
        roc_auc_rf = roc_auc_score(y_test == 'depressed', y_pred_proba_rf[:, depressed_idx])
        print(f"\nROC AUC Score: {roc_auc_rf:.4f}")

# Feature importance for Random Forest
print("\nFeature Importance:")
print(f"avg_daily_steps: {rf_model.feature_importances_[0]:.4f}")

# Summary statistics by predicted class
print("\n" + "="*80)
print("PREDICTION ANALYSIS")
print("="*80)

test_df = pd.DataFrame({
    'avg_daily_steps': X_test.flatten(),
    'actual': y_test,
    'predicted_lr': y_pred_lr,
    'predicted_rf': y_pred_rf
})

print("\nAverage steps by actual label:")
print(test_df.groupby('actual')['avg_daily_steps'].agg(['mean', 'std', 'min', 'max']))

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nBoth models trained successfully!")
print(f"Logistic Regression - Test Accuracy: {accuracy_score(y_test, y_pred_lr):.2%}")
print(f"Random Forest - Test Accuracy: {accuracy_score(y_test, y_pred_rf):.2%}")

