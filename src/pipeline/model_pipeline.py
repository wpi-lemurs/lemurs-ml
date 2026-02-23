"""
Complete modeling pipeline combining data processing and ML models.

This module provides end-to-end pipelines that include data extraction,
processing, feature engineering, and model training/evaluation.

Example usage:
    # Create and run a complete modeling pipeline
    from src.pipeline.model_pipeline import ScreentimeModelPipeline

    pipeline = ScreentimeModelPipeline(
        target_type='suicide_risk',
        time_windows=[3, 6, 9, 12],
        use_loocv=True,
        balanced_class_weight=True
    )

    # Run the entire pipeline
    results = pipeline.fit_predict()

    # Access results
    for window, window_results in results.items():
        print(f"Window {window}h: {window_results}")
"""

import numpy as np
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.pipeline.mental_health_pipeline import (
    create_screentime_risk_pipeline,
    create_screentime_phq9_pipeline,
    create_subwindow_pipeline
)
from src.config import DATA_DIR


class ScreentimeModelPipeline:
    """
    Complete pipeline for screentime-based mental health prediction.

    This class wraps the data processing pipeline and adds model training,
    evaluation, and cross-validation capabilities.

    Parameters:
    -----------
    target_type : str, default='suicide_risk'
        Target to predict: 'phq9', 'suicide_risk', 'self_harm', or 'sleep'
    time_windows : list of int, default=[3, 6, 9, 12]
        Hours before survey to use for features (ignored if use_subwindows=True)
    models : list of str or dict, default=['logistic_regression', 'random_forest']
        Models to train. Can be list of names or dict of {name: model_instance}
    fill_method : str or None, default='zero'
        Method for handling missing values (only applies to traditional hourly features)
    propagate_labels : bool, default=False
        Whether to propagate positive labels
    balanced_class_weight : bool, default=False
        Whether to use balanced class weights
    use_loocv : bool, default=False
        Whether to use leave-one-user-out cross-validation
    use_subwindows : bool, default=False
        Whether to ADD sub-window category features to traditional hourly features.
        When True, combines BOTH hourly screentime (hour_0, hour_1, etc.) AND
        sub-window app category features (most_used_category, time, num_apps).
        This provides richer information than either approach alone.
    lookback_hours : int, default=12
        Total lookback hours for sub-window features (only if use_subwindows=True)
    subwindow_hours : int, default=3
        Size of each sub-window in hours (only if use_subwindows=True)
    test_size : float, default=0.3
        Test set size for train/test split (ignored if use_loocv=True)
    random_state : int, default=42
        Random seed for reproducibility
    save_confusion_matrices : bool, default=True
        Whether to save confusion matrix plots
    use_accurate_method : bool, default=False
        If True, uses calculate_accurate_screentime_from_app_table for more precise calculations.
        If False, uses the original method based on screentime table.

    Methods:
    --------
    fit_predict() : dict
        Run the complete pipeline and return results
    save_confusion_matrices_plots(save_all_windows=True, save_best_only=True) : None
        Save confusion matrix visualizations
        - save_all_windows: Creates grid of confusion matrices for all time windows
        - save_best_only: Creates plot showing only best performing models
        Files are saved to DATA_DIR with naming pattern:
        confusion_matrices_{target_type}[_balanced][_loocv][_best].png

    Example:
    --------
    >>> pipeline = ScreentimeModelPipeline(
    ...     target_type='suicide_risk',
    ...     time_windows=[3, 6, 9],
    ...     balanced_class_weight=True
    ... )
    >>> results = pipeline.fit_predict()  # Automatically saves confusion matrices
    >>> # Or manually save:
    >>> pipeline.save_confusion_matrices_plots(save_all_windows=True, save_best_only=True)
    """

    def __init__(
        self,
        target_type='suicide_risk',
        time_windows=None,
        models=None,
        fill_method='zero',
        propagate_labels=False,
        balanced_class_weight=False,
        use_loocv=False,
        use_subwindows=False,
        lookback_hours=12,
        subwindow_hours=3,
        test_size=0.3,
        random_state=42,
        save_confusion_matrices=True,
        use_accurate_method=False
    ):
        self.target_type = target_type
        self.time_windows = time_windows or [3, 6, 9, 12]
        self.fill_method = fill_method
        self.propagate_labels = propagate_labels
        self.balanced_class_weight = balanced_class_weight
        self.use_loocv = use_loocv
        self.use_subwindows = use_subwindows
        self.lookback_hours = lookback_hours
        self.subwindow_hours = subwindow_hours
        self.test_size = test_size
        self.random_state = random_state
        self.save_confusion_matrices = save_confusion_matrices
        self.use_accurate_method = use_accurate_method

        # Initialize models
        if models is None:
            models = ['logistic_regression', 'random_forest']
        self.models = self._initialize_models(models)

        # Create data processing pipeline
        if use_subwindows:
            # Use sub-window pipeline
            self.data_pipeline = create_subwindow_pipeline(
                target_type=self.target_type,
                lookback_hours=self.lookback_hours,
                subwindow_hours=self.subwindow_hours,
                propagate_labels=self.propagate_labels,
                use_accurate_method=self.use_accurate_method
            )
        elif target_type == 'phq9':
            self.data_pipeline = create_screentime_phq9_pipeline(
                time_windows=self.time_windows,
                fill_method=self.fill_method,
                propagate_labels=self.propagate_labels,
                use_accurate_method=self.use_accurate_method
            )
        else:
            self.data_pipeline = create_screentime_risk_pipeline(
                target_type=self.target_type,
                time_windows=self.time_windows,
                fill_method=self.fill_method,
                propagate_labels=self.propagate_labels,
                use_accurate_method=self.use_accurate_method
            )

        # Store results
        self.results_ = {}
        self.processed_data_ = None

    def _initialize_models(self, models):
        """Initialize model instances."""
        class_weight = 'balanced' if self.balanced_class_weight else None

        model_map = {
            'logistic_regression': LogisticRegression(
                max_iter=10000,
                random_state=self.random_state,
                class_weight=class_weight,
                solver='lbfgs'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight=class_weight
            )
        }

        if isinstance(models, dict):
            return models

        # Convert list of names to dict of instances
        return {name: model_map[name] for name in models if name in model_map}

    def _get_label_config(self):
        """Get label configuration based on target type."""
        if self.target_type == 'phq9':
            return {
                'label_col': 'severity_label',
                'positive_class': 'depressed',
                'negative_class': 'not_depressed',
                'output_prefix': 'screentime_phq9'
            }
        else:
            label_col_map = {
                'suicide_risk': 'suicide_risk_label',
                'self_harm': 'self_harm_risk_label',
                'sleep': 'sleep_label'
            }
            return {
                'label_col': label_col_map[self.target_type],
                'positive_class': 'at_risk',
                'negative_class': 'not_at_risk',
                'output_prefix': f'screentime_{self.target_type}'
            }

    def _print_feature_importance(self, model, feature_names, top_n=100):
        """
        Print top N most important features from a tree-based model.

        Parameters:
        -----------
        model : trained model with feature_importances_ attribute
        feature_names : list of feature names
        top_n : int, number of top features to display (default 20)
        """
        if not hasattr(model, 'feature_importances_'):
            return

        import pandas as pd

        # Get feature importances
        importances = model.feature_importances_

        # Create DataFrame for easy sorting
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        # Filter out features with zero importance
        # feature_importance_df = feature_importance_df[feature_importance_df['importance'] > 0]

        print(f"\n  Top {min(top_n, len(feature_importance_df))} Most Important Features:")
        print("  " + "-" * 60)

        for idx, row in feature_importance_df.head(top_n).iterrows():
            importance_pct = row['importance'] * 100
            print(f"  {row['feature']:45s} {importance_pct:6.2f}%")

        # Print summary statistics
        print("  " + "-" * 60)
        total_importance = feature_importance_df.head(top_n)['importance'].sum() * 100
        print(f"  Total importance of top {min(top_n, len(feature_importance_df))} features: {total_importance:.2f}%")
        print(f"  Total features: {len(feature_names)}, Non-zero: {len(feature_importance_df)}")

    def fit_predict(self):
        """
        Run the complete pipeline: extract data, process, train, and evaluate.

        Returns:
        --------
        dict : Results for each time window
        """
        # Step 1: Process data through pipeline
        print("\n" + "="*80)
        print("EXTRACTING AND PROCESSING DATA")
        print("="*80)

        self.data_pipeline.fit(None)
        self.processed_data_ = self.data_pipeline.transform(None)

        # Check if we have data (could be dict or DataFrame)
        if self.processed_data_ is None:
            print("ERROR: No data produced by pipeline")
            return {}

        if isinstance(self.processed_data_, dict) and len(self.processed_data_) == 0:
            print("ERROR: No data produced by pipeline")
            return {}

        if hasattr(self.processed_data_, 'empty') and self.processed_data_.empty:
            print("ERROR: No data produced by pipeline")
            return {}

        # Step 2: Train and evaluate models for each time window
        label_config = self._get_label_config()

        for window, data in self.processed_data_.items():
            print(f"\n{'='*80}")
            print(f"MODELING TIME WINDOW: {window} HOURS")
            print(f"{'='*80}")

            if data.empty:
                print(f"No data for window {window}h")
                continue

            # Check if we have both classes
            label_col = label_config['label_col']
            if label_col not in data.columns:
                print(f"Label column '{label_col}' not found in data")
                continue

            if data[label_col].nunique() < 2:
                print(f"Only one class present: {data[label_col].unique()}")
                continue

            # Train and evaluate
            window_results = self._train_evaluate_window(
                data, window, label_config
            )

            if window_results:
                self.results_[window] = window_results

        # Step 3: Save confusion matrices
        if self.save_confusion_matrices and self.results_:
            self.save_confusion_matrices_plots(save_all_windows=True, save_best_only=True)

        # Step 4: Print summary
        self._print_summary()

        return self.results_

    def _train_evaluate_window(self, data, window, label_config):
        """Train and evaluate models for a single time window."""
        label_col = label_config['label_col']
        positive_class = label_config['positive_class']
        negative_class = label_config['negative_class']

        # Filter out N/A labels for sleep (afternoon surveys without sleep data)
        if self.target_type == 'sleep' and label_col in data.columns:
            original_size = len(data)
            data = data[data[label_col] != 'N/A'].copy()
            filtered_count = original_size - len(data)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} rows with N/A sleep labels")

        # Prepare features and labels
        exclude_cols = [
            'app_user_id', 'survey_response_id', 'timestamp', 'survey_timestamp',
            'week_start', 'time_key', 'date', 'phq9_response_id',
            label_col, 'phq9_total_score'
        ]

        # Also exclude any datetime columns automatically
        datetime_cols = data.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
        exclude_cols.extend(datetime_cols)

        # Remove duplicates
        exclude_cols = list(set(exclude_cols))

        feature_cols = [col for col in data.columns if col not in exclude_cols]

        if not feature_cols:
            print(f"No feature columns found")
            return None

        # Ensure all features are numeric
        X = data[feature_cols].copy()

        # Apply one-hot encoding to categorical columns
        non_numeric_cols = X.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric_cols:
            print(f"Applying one-hot encoding to categorical columns: {non_numeric_cols}")
            # Use pd.get_dummies for one-hot encoding
            X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=True, dummy_na=False)
            # Convert boolean columns to int (0/1)
            bool_cols = X.select_dtypes(include=['bool']).columns
            X[bool_cols] = X[bool_cols].astype(int)
            print(f"After encoding: {X.shape[1]} features")

        X = X.fillna(0)
        y = data[label_col]

        # Check minimum samples
        class_counts = y.value_counts()
        if class_counts.min() < 2:
            print(f"Insufficient samples for class: {class_counts.to_dict()}")
            return None

        results = {
            'window': window,
            'total_samples': len(data),
            'class_distribution': class_counts.to_dict(),
            'models': {}
        }

        # Train and evaluate each model
        if self.use_loocv:
            results['models'] = self._train_loocv(
                X, y, data, label_config
            )
        else:
            results['models'] = self._train_test_split(
                X, y, label_config
            )

        return results

    def _train_test_split(self, X, y, label_config):
        """Train models using train/test split."""
        positive_class = label_config['positive_class']

        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state
            )

        model_results = {}

        # Scale features (important for Logistic Regression convergence)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for model_name, model in self.models.items():
            # Use scaled data for Logistic Regression, original for Random Forest
            if 'logistic' in model_name.lower():
                X_train_model = X_train_scaled
                X_test_model = X_test_scaled
            else:
                # Random Forest doesn't need scaling
                X_train_model = X_train
                X_test_model = X_test

            # Train
            model.fit(X_train_model, y_train)

            # Predict
            y_pred = model.predict(X_test_model)

            # Get probability predictions for AUC if available
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test_model)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_proba = model.decision_function(X_test_model)
                else:
                    y_proba = None
            except:
                y_proba = None

            # Evaluate
            acc = accuracy_score(y_test, y_pred)

            # Calculate F1 score
            try:
                f1 = f1_score(y_test, y_pred, pos_label=positive_class, average='binary')
            except Exception as e:
                print(f"  Warning: Could not calculate F1 score: {e}")
                print(f"  Unique values in y_test: {set(y_test)}")
                print(f"  Unique values in y_pred: {set(y_pred)}")
                print(f"  Expected positive_class: {positive_class}")
                f1 = None

            # Calculate AUC score
            auc = None
            if y_proba is not None:
                try:
                    # Create binary labels for AUC calculation
                    y_test_binary = (y_test == positive_class).astype(int)
                    auc = roc_auc_score(y_test_binary, y_proba)
                except Exception as e:
                    print(f"  Warning: Could not calculate AUC: {e}")

            cm = confusion_matrix(
                y_test, y_pred,
                labels=[label_config['negative_class'], positive_class]
            )

            model_results[model_name] = {
                'accuracy': acc,
                'f1_score': f1,
                'roc_auc': auc,
                'confusion_matrix': cm.tolist(),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'trained_model': model,
                'feature_names': X.columns.tolist()
            }

            print(f"\n{model_name.replace('_', ' ').title()}:")
            print(f"  Accuracy: {acc:.4f}")
            if f1 is not None:
                print(f"  F1 Score: {f1:.4f}")
            if auc is not None:
                print(f"  AUC: {auc:.4f}")

            # Print feature importance for Random Forest
            if model_name == 'random_forest' and hasattr(model, 'feature_importances_'):
                self._print_feature_importance(model, X.columns.tolist())

        return model_results

    def _train_loocv(self, X, y, data, label_config):
        """Train models using leave-one-user-out cross-validation."""
        positive_class = label_config['positive_class']
        negative_class = label_config['negative_class']

        if 'app_user_id' not in data.columns:
            print("app_user_id not found, falling back to train/test split")
            return self._train_test_split(X, y, label_config)

        groups = data['app_user_id'].values
        logo = LeaveOneGroupOut()

        model_results = {}

        for model_name, model in self.models.items():
            all_y_test = []
            all_y_pred = []
            all_y_proba = []
            successful_folds = 0

            for train_idx, test_idx in logo.split(X, y, groups):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Check if train set has both classes
                if y_train.nunique() < 2:
                    continue

                try:
                    # Scale features for Logistic Regression
                    if 'logistic' in model_name.lower():
                        scaler = StandardScaler()
                        X_train_model = scaler.fit_transform(X_train)
                        X_test_model = scaler.transform(X_test)
                    else:
                        X_train_model = X_train
                        X_test_model = X_test

                    # Train and predict
                    model.fit(X_train_model, y_train)
                    y_pred = model.predict(X_test_model)

                    # Get probabilities for AUC if available
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_test_model)[:, 1]
                    elif hasattr(model, 'decision_function'):
                        y_proba = model.decision_function(X_test_model)
                    else:
                        y_proba = None

                    all_y_test.extend(y_test)
                    all_y_pred.extend(y_pred)
                    if y_proba is not None:
                        all_y_proba.extend(y_proba)
                    successful_folds += 1
                except:
                    continue

            if successful_folds == 0:
                print(f"No successful folds for {model_name}")
                continue

            # Evaluate aggregated predictions
            all_y_test = np.array(all_y_test)
            all_y_pred = np.array(all_y_pred)

            acc = accuracy_score(all_y_test, all_y_pred)

            # Calculate F1 score
            try:
                f1 = f1_score(all_y_test, all_y_pred, pos_label=positive_class, average='binary')
            except Exception as e:
                print(f"  Warning: Could not calculate F1 score: {e}")
                print(f"  Unique values in all_y_test: {set(all_y_test)}")
                print(f"  Unique values in all_y_pred: {set(all_y_pred)}")
                print(f"  Expected positive_class: {positive_class}")
                f1 = None

            # Calculate AUC score
            auc = None
            if len(all_y_proba) > 0:
                try:
                    all_y_proba = np.array(all_y_proba)
                    # Create binary labels for AUC calculation
                    y_test_binary = (all_y_test == positive_class).astype(int)
                    auc = roc_auc_score(y_test_binary, all_y_proba)
                except Exception as e:
                    print(f"  Warning: Could not calculate AUC: {e}")

            cm = confusion_matrix(
                all_y_test, all_y_pred,
                labels=[negative_class, positive_class]
            )

            model_results[model_name] = {
                'accuracy': acc,
                'f1_score': f1,
                'roc_auc': auc,
                'confusion_matrix': cm.tolist(),
                'successful_folds': successful_folds,
                'cv_method': 'LOOCV'
            }

            print(f"\n{model_name.replace('_', ' ').title()}:")
            print(f"  Accuracy: {acc:.4f}")
            if f1 is not None:
                print(f"  F1 Score: {f1:.4f}")
            if auc is not None:
                print(f"  AUC: {auc:.4f}")
            print(f"  Successful folds: {successful_folds}")

        return model_results

    def _plot_confusion_matrices(self, filename, label_config):
        """Plot confusion matrices for all windows and models."""
        if not self.results_:
            return

        n_windows = len(self.results_)
        n_models = len(self.models)

        fig, axes = plt.subplots(n_models, n_windows, figsize=(5*n_windows, 5*n_models))

        if n_windows == 1:
            axes = axes.reshape(-1, 1)
        if n_models == 1:
            axes = axes.reshape(1, -1)

        for col_idx, (window, window_results) in enumerate(self.results_.items()):
            for row_idx, (model_name, model_results) in enumerate(window_results['models'].items()):
                ax = axes[row_idx, col_idx]

                cm = np.array(model_results['confusion_matrix'])

                sns.heatmap(
                    cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[label_config['negative_class'], label_config['positive_class']],
                    yticklabels=[label_config['negative_class'], label_config['positive_class']],
                    ax=ax
                )

                f1 = model_results.get('f1_score')
                f1_str = f", F1: {f1:.3f}" if f1 is not None else ""

                ax.set_title(
                    f"{model_name.replace('_', ' ').title()}\n"
                    f"Window: {window}h{f1_str}"
                )
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')

        plt.tight_layout()
        plt.savefig(f"{DATA_DIR}/{filename}", dpi=150, bbox_inches='tight')
        plt.close()

    def save_confusion_matrices_plots(self, save_all_windows=True, save_best_only=True):
        """
        Save confusion matrices as PNG files.

        Parameters:
        -----------
        save_all_windows : bool, default=True
            If True, saves confusion matrices for all time windows
        save_best_only : bool, default=True
            If True, saves confusion matrices for only the best performing windows
        """
        if not self.results_:
            print("No results to plot")
            return

        label_config = self._get_label_config()
        labels = [label_config['negative_class'], label_config['positive_class']]

        # Build filename suffix
        balanced_suffix = '_balanced' if self.balanced_class_weight else ''
        loocv_suffix = '_loocv' if self.use_loocv else ''

        # Save all windows
        if save_all_windows:
            n_windows = len(self.results_)
            n_models = len(self.models)

            fig, axes = plt.subplots(n_windows, n_models, figsize=(7 * n_models, 5 * n_windows))

            # Handle single window or single model case
            if n_windows == 1 and n_models == 1:
                axes = np.array([[axes]])
            elif n_windows == 1:
                axes = axes.reshape(1, -1)
            elif n_models == 1:
                axes = axes.reshape(-1, 1)

            class_weight_title = " (Balanced Class Weights)" if self.balanced_class_weight else ""
            fig.suptitle(
                f'Confusion Matrices - {self.target_type.replace("_", " ").title()} Prediction{class_weight_title}',
                fontsize=16, fontweight='bold', y=0.995
            )

            # Plot each window and model
            model_names = list(self.models.keys())
            windows = sorted(self.results_.keys())

            for row_idx, window in enumerate(windows):
                window_results = self.results_[window]

                for col_idx, model_name in enumerate(model_names):
                    if model_name in window_results['models']:
                        model_results = window_results['models'][model_name]
                        cm = model_results.get('confusion_matrix')

                        if cm is not None:
                            cm_array = np.array(cm)

                            # Choose colormap based on model
                            cmap = 'Blues' if 'logistic' in model_name.lower() else 'Greens'

                            sns.heatmap(
                                cm_array, annot=True, fmt='d', cmap=cmap,
                                xticklabels=labels, yticklabels=labels,
                                ax=axes[row_idx, col_idx], cbar=True
                            )

                            # Build title with metrics
                            title = f'{model_name.replace("_", " ").title()} - {window}h window\n'
                            title += f'Accuracy: {model_results.get("accuracy", 0):.3f}'

                            if model_results.get('f1_score') is not None:
                                title += f' | F1: {model_results.get("f1_score"):.3f}'

                            axes[row_idx, col_idx].set_title(title)
                            axes[row_idx, col_idx].set_ylabel('True Label')
                            axes[row_idx, col_idx].set_xlabel('Predicted Label')

            plt.tight_layout()
            output_filename = f'confusion_matrices_{self.target_type}{balanced_suffix}{loocv_suffix}.png'
            output_path = DATA_DIR / output_filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved all windows confusion matrices to: {output_path}")

        # Save best only
        if save_best_only:
            n_models = len(self.models)
            fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5))

            # Handle single model case
            if n_models == 1:
                axes = [axes]

            fig.suptitle(
                f'Best Performing Models - {self.target_type.replace("_", " ").title()} Prediction',
                fontsize=16, fontweight='bold'
            )

            model_names = list(self.models.keys())

            for col_idx, model_name in enumerate(model_names):
                # Find best window for this model by F1 score
                best_f1 = -1
                best_window = None
                best_results = None

                for window, window_results in self.results_.items():
                    if model_name in window_results['models']:
                        model_results = window_results['models'][model_name]
                        f1 = model_results.get('f1_score')

                        if f1 is not None and f1 > best_f1:
                            best_f1 = f1
                            best_window = window
                            best_results = model_results

                if best_results is not None:
                    cm = best_results.get('confusion_matrix')

                    if cm is not None:
                        cm_array = np.array(cm)

                        # Choose colormap based on model
                        cmap = 'Blues' if 'logistic' in model_name.lower() else 'Greens'

                        sns.heatmap(
                            cm_array, annot=True, fmt='d', cmap=cmap,
                            xticklabels=labels, yticklabels=labels,
                            ax=axes[col_idx], cbar=True, annot_kws={'size': 14}
                        )

                        # Build title with metrics
                        title = f'Best {model_name.replace("_", " ").title()}\n'
                        title += f'{best_window}h window - Accuracy: {best_results.get("accuracy", 0):.3f}'

                        if best_results.get('f1_score') is not None:
                            title += f' | F1: {best_results.get("f1_score"):.3f}'

                        axes[col_idx].set_title(title, fontsize=12, fontweight='bold')
                        axes[col_idx].set_ylabel('True Label', fontsize=11)
                        axes[col_idx].set_xlabel('Predicted Label', fontsize=11)

            plt.tight_layout()
            best_output_filename = f'confusion_matrices_{self.target_type}{balanced_suffix}{loocv_suffix}_best.png'
            best_output_path = DATA_DIR / best_output_filename
            plt.savefig(best_output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved best models confusion matrices to: {best_output_path}")

    def _print_summary(self):
        """Print summary of results."""
        if not self.results_:
            print("\nNo results to summarize")
            return

        print("\n" + "="*80)
        print("SUMMARY - ALL TIME WINDOWS")
        print("="*80)

        # Get all model names
        model_names = list(self.models.keys())

        # Display results for each model
        for model_name in model_names:
            print(f"\n{model_name.replace('_', ' ').title()}:")
            print(f"{'Window':>8} {'F1 Score':>12} {'Accuracy':>12} {'AUC':>12}")
            print("-" * 48)

            best_f1 = -1
            best_window = None
            best_cm = None

            # Collect and display all windows
            for window in sorted(self.results_.keys()):
                window_results = self.results_[window]
                if model_name in window_results['models']:
                    model_results = window_results['models'][model_name]
                    f1 = model_results.get('f1_score')
                    acc = model_results.get('accuracy')
                    auc = model_results.get('roc_auc')

                    # Format metrics (handle None values)
                    f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
                    acc_str = f"{acc:.4f}" if acc is not None else "N/A"
                    auc_str = f"{auc:.4f}" if auc is not None else "N/A"

                    print(f"{window:>6}h {f1_str:>12} {acc_str:>12} {auc_str:>12}")

                    # Track best window by F1 score
                    if f1 is not None and f1 > best_f1:
                        best_f1 = f1
                        best_window = window
                        best_cm = model_results.get('confusion_matrix')

            # Print best window and its confusion matrix
            if best_window is not None:
                print(f"\n  Best Window: {best_window}h (F1={best_f1:.4f})")

                if best_cm is not None:
                    cm = np.array(best_cm)
                    label_config = self._get_label_config()
                    labels = [label_config['negative_class'], label_config['positive_class']]

                    print(f"\n  Confusion Matrix (Best Window):")
                    print(f"  {'':>15} Predicted")
                    print(f"  {'':>15} {labels[0]:>12} {labels[1]:>12}")
                    print(f"  Actual")
                    print(f"  {labels[0]:>15} {cm[0,0]:>12} {cm[0,1]:>12}")
                    print(f"  {labels[1]:>15} {cm[1,0]:>12} {cm[1,1]:>12}")
            print()


# Example usage function
def run_experiment(
    target_type='suicide_risk',
    time_windows=None,
    propagate_labels=False,
    balanced_class_weight=False,
    use_loocv=False
):
    """
    Convenience function to run a complete experiment.

    Parameters:
    -----------
    target_type : str
        Target to predict
    time_windows : list of int or None
        Time windows to experiment with
    propagate_labels : bool
        Whether to propagate positive labels
    balanced_class_weight : bool
        Whether to use balanced class weights
    use_loocv : bool
        Whether to use LOOCV

    Returns:
    --------
    dict : Experiment results
    """
    pipeline = ScreentimeModelPipeline(
        target_type=target_type,
        time_windows=time_windows,
        propagate_labels=propagate_labels,
        balanced_class_weight=balanced_class_weight,
        use_loocv=use_loocv
    )

    return pipeline.fit_predict()

