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
from src.merge_passive_data_and_labels import merge_weekly_health_with_phq9

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

# Confusion Matrix Heatmaps
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_lr),
            annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression – Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf),
            annot=True, fmt='d', cmap='Purples')
plt.title("Random Forest – Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ROC Curve Plot
if 'depressed' in lr_model.classes_:
    depressed_idx = list(lr_model.classes_).index('depressed')

    # Logistic Regression
    fpr_lr, tpr_lr, _ = roc_curve(y_test == "depressed",
                                  y_pred_proba_lr[:, depressed_idx])
    auc_lr = auc(fpr_lr, tpr_lr)

    # Random Forest
    fpr_rf, tpr_rf, _ = roc_curve(y_test == "depressed",
                                  y_pred_proba_rf[:, depressed_idx])
    auc_rf = auc(fpr_rf, tpr_rf)

    plt.figure(figsize=(7,5))
    plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression AUC = {auc_lr:.3f}")
    plt.plot(fpr_rf, tpr_rf, color="purple", label=f"Random Forest AUC = {auc_rf:.3f}")
    plt.plot([0,1], [0,1], 'k--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Depression vs Not Depressed")
    plt.legend()
    plt.tight_layout()
    plt.show()


# F1 Score Bar Chart
f1_lr = f1_score(y_test, y_pred_lr, average='weighted')
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

plt.figure(figsize=(6,4))
f1_bars = plt.bar(['Logistic Regression', 'Random Forest'], [f1_lr, f1_rf])
plt.ylabel("Weighted F1 Score")
plt.title("F1 Score Comparison")
plt.ylim(0, 1)
for bar, val in zip(f1_bars, [f1_lr, f1_rf]):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
             f"{val:.2%}", ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.show()

# Step Accuracy Graph
acc_lr = accuracy_score(y_test, y_pred_lr)
acc_rf = accuracy_score(y_test, y_pred_rf)

plt.figure(figsize=(6,4))
bars = plt.bar(['Logistic Regression', 'Random Forest'],
               [acc_lr, acc_rf])
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.ylim(0, 1)
for bar, val in zip(bars, [acc_lr, acc_rf]):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
             f"{val:.2%}", ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.show()

# Steps Distribution by Class
plt.figure(figsize=(6,4))
sns.boxplot(data=test_df, x='actual', y='avg_daily_steps')
plt.title("Distribution of Steps by Actual Depression Label")
plt.xlabel("Depression Label")
plt.ylabel("Average Daily Steps")
plt.tight_layout()
plt.show()
