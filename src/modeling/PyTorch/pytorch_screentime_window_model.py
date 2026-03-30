import os
import random
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.pipeline.mental_health_pipeline import (
    create_screentime_phq9_pipeline,
    create_screentime_risk_pipeline,
    create_subwindow_pipeline,
)
from src.pipeline.transformers import FeatureSelector, LabelEncoder

# Fixed seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Using the prexisting scikitlearn pipeline class
def _select_pipeline(
    target_type: str,
    time_windows: Iterable[int],
    propagate_labels: bool,
    use_accurate_method: bool,
    use_subwindows: bool,
    lookback_hours: int,
    subwindow_hours: int,
    standardized: bool,
):
    if use_subwindows:
        return create_subwindow_pipeline(
            target_type=target_type,
            lookback_hours=lookback_hours,
            subwindow_hours=subwindow_hours,
            propagate_labels=propagate_labels,
            use_accurate_method=use_accurate_method,
            standardized=standardized,
        )

    if target_type == "phq9":
        return create_screentime_phq9_pipeline(
            time_windows=list(time_windows),
            propagate_labels=propagate_labels,
            use_accurate_method=use_accurate_method,
        )
    return create_screentime_risk_pipeline(
        target_type=target_type,
        time_windows=list(time_windows),
        propagate_labels=propagate_labels,
        use_accurate_method=use_accurate_method,
    )


def _prepare_features_labels(df, target_type: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], pd.Index]:
    # Drop rows with missing target
    df = df.copy()
    label_encoder = LabelEncoder(target_type=target_type)
    label_encoder.fit(df)
    label_series = label_encoder.transform(df)
    if label_series is None:
        return np.array([]), np.array([]), [], [], pd.Index([])
    df = df.loc[label_series.notna()]
    label_series = label_series.loc[label_series.notna()]

    selector = FeatureSelector()
    selector.fit(df)
    feature_df = selector.transform(df)

    # Separate column types
    datetime_cols = feature_df.select_dtypes(include=["datetime64[ns]"]).columns
    cat_cols = feature_df.select_dtypes(include=["object", "category", "bool"]).columns

    # Remove datetimes (they cannot be fed directly to the model)
    feature_df = feature_df.drop(columns=list(datetime_cols), errors="ignore")

    # One-hot encode categoricals (keep all levels, fill missing with explicit token)
    if len(cat_cols) > 0:
        feature_df[cat_cols] = feature_df[cat_cols].fillna("<missing>")
        feature_df = pd.get_dummies(feature_df, columns=list(cat_cols), dtype=float)

    # Ensure numeric types and drop any remaining non-numeric columns
    numeric_df = feature_df.select_dtypes(include=[np.number]).copy()
    dropped_cols = set(feature_df.columns) - set(numeric_df.columns)
    if dropped_cols:
        print(f"Dropping non-numeric feature columns: {sorted(dropped_cols)}")

    if numeric_df.empty:
        return np.array([]), np.array([]), [], [], pd.Index([])

    numeric_df = numeric_df.fillna(0)

    X = numeric_df.to_numpy(dtype=float)
    classes = list(label_encoder.classes_ or [])
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    y = label_series.map(label_map).to_numpy(dtype=int)
    return X, y, list(numeric_df.columns), classes, numeric_df.index


def _train_val_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, scale: bool = True):
    if len(np.unique(y)) < 2:
        raise ValueError("Need at least two classes for training")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y if len(np.unique(y)) > 1 else None, random_state=42
    )
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
    return X_train, X_val, y_train, y_val, scaler


class ScreentimeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: Iterable[int] = (128, 64), dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _make_loaders(
    X_train,
    X_val,
    y_train,
    y_val,
    batch_size: int = 64,
    use_weighted_sampler: bool = False,
):
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    if use_weighted_sampler:
        classes, counts = np.unique(y_train, return_counts=True)
        class_weights = {cls: counts.sum() / (len(classes) * cnt) for cls, cnt in zip(classes, counts)}
        sample_weights = np.array([class_weights[int(lbl)] for lbl in y_train], dtype=np.float32)
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _compute_class_weights(y: np.ndarray) -> Optional[torch.Tensor]:
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return None
    weights = counts.sum() / (len(classes) * counts)
    full_weights = np.ones(int(classes.max()) + 1)
    for cls, w in zip(classes, weights):
        full_weights[int(cls)] = w
    return torch.tensor(full_weights, dtype=torch.float32)


def _binary_metrics_from_cm(cm: np.ndarray) -> Tuple[float, float]:
    """Return (sensitivity, specificity) from a 2x2 confusion matrix."""
    if cm is None or cm.shape != (2, 2):
        return float("nan"), float("nan")

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity, specificity


def _train_evaluate_split(
    X_train,
    X_val,
    y_train,
    y_val,
    hidden_layers,
    dropout,
    lr,
    weight_decay,
    use_weighted_sampler,
    epochs,
    batch_size,
    device,
):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = ScreentimeMLP(input_dim=X_train_scaled.shape[1], hidden_layers=hidden_layers, dropout=dropout).to(device)
    class_weights = _compute_class_weights(y_train)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader, val_loader = _make_loaders(
        X_train_scaled,
        X_val_scaled,
        y_train,
        y_val,
        batch_size=batch_size,
        use_weighted_sampler=use_weighted_sampler,
    )

    train_losses, val_losses = [], []
    preds, targets = np.array([]), np.array([])

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        train_losses.append(epoch_loss / len(train_loader.dataset))

        model.eval()
        fold_preds, fold_targets = [], []
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss = criterion(logits, yb)
                val_loss_sum += val_loss.item() * xb.size(0)
                fold_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                fold_targets.append(yb.cpu().numpy())
        preds = np.concatenate(fold_preds)
        targets = np.concatenate(fold_targets)
        val_losses.append(val_loss_sum / len(val_loader.dataset))

    return {
        "model": model,
        "scaler": scaler,
        "preds": preds,
        "targets": targets,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


def train_one_window(
    df,
    target_type: str,
    hidden_layers: Iterable[int] = (128, 64),
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    use_weighted_sampler: bool = True,
    epochs: int = 20,
    batch_size: int = 64,
    window_id: Optional[int] = None,
    save_plots: bool = True,
    plots_dir: str = "data",
    use_loocv: bool = False,
    group_col: str = "app_user_id",
):
    X, y, feature_names, classes, row_index = _prepare_features_labels(df, target_type)
    if X.size == 0 or y.size == 0:
        raise ValueError("No data available after preprocessing")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = None
    scaler = None
    train_losses, val_losses = [], []
    cm = None
    sensitivity = float("nan")
    specificity = float("nan")
    successful_folds = 1
    acc = float("nan")
    f1 = float("nan")

    if use_loocv:
        if group_col not in df.columns:
            raise ValueError(f"LOOCV requires '{group_col}' column in the input dataframe")

        groups = df.loc[row_index, group_col]
        valid_group_mask = groups.notna().to_numpy()
        X_cv = X[valid_group_mask]
        y_cv = y[valid_group_mask]
        groups_cv = groups.loc[groups.notna()]

        if len(np.unique(groups_cv)) < 2:
            raise ValueError("LOOCV requires at least two distinct users/groups")

        logo = LeaveOneGroupOut()
        all_preds = []
        all_targets = []
        fold_train_losses = []
        fold_val_losses = []
        successful_folds = 0

        for train_idx, test_idx in logo.split(X_cv, y_cv, groups_cv):
            y_train_fold = y_cv[train_idx]
            y_test_fold = y_cv[test_idx]

            if np.unique(y_train_fold).size < 2:
                continue
            if y_test_fold.size == 0:
                continue

            fold_result = _train_evaluate_split(
                X_train=X_cv[train_idx],
                X_val=X_cv[test_idx],
                y_train=y_train_fold,
                y_val=y_test_fold,
                hidden_layers=hidden_layers,
                dropout=dropout,
                lr=lr,
                weight_decay=weight_decay,
                use_weighted_sampler=use_weighted_sampler,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
            )

            successful_folds += 1
            model = fold_result["model"]
            scaler = fold_result["scaler"]
            all_preds.append(fold_result["preds"])
            all_targets.append(fold_result["targets"])
            fold_train_losses.append(fold_result["train_losses"])
            fold_val_losses.append(fold_result["val_losses"])

            fold_acc = accuracy_score(fold_result["targets"], fold_result["preds"])
            fold_f1 = f1_score(fold_result["targets"], fold_result["preds"], zero_division=0)
            print(
                f"LOOCV fold {successful_folds} - "
                f"acc={fold_acc:.3f} f1={fold_f1:.3f}"
            )

        if successful_folds == 0:
            raise ValueError("No valid LOOCV folds were available for training")

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, zero_division=0)
        cm = confusion_matrix(targets, preds, labels=[0, 1])
        sensitivity, specificity = _binary_metrics_from_cm(cm)

        train_losses = np.mean(np.asarray(fold_train_losses), axis=0).tolist()
        val_losses = np.mean(np.asarray(fold_val_losses), axis=0).tolist()
        print(
            f"LOOCV aggregate - acc={acc:.3f} f1={f1:.3f} "
            f"sens={sensitivity:.3f} spec={specificity:.3f} folds={successful_folds}"
        )
    else:
        X_train, X_val, y_train, y_val, _ = _train_val_split(X, y)
        split_result = _train_evaluate_split(
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            hidden_layers=hidden_layers,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            use_weighted_sampler=use_weighted_sampler,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
        )

        model = split_result["model"]
        scaler = split_result["scaler"]
        train_losses = split_result["train_losses"]
        val_losses = split_result["val_losses"]
        preds = split_result["preds"]
        targets = split_result["targets"]

        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, zero_division=0)
        cm = confusion_matrix(targets, preds, labels=[0, 1])
        sensitivity, specificity = _binary_metrics_from_cm(cm)
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1):
            print(
                f"Epoch {epoch}/{epochs} - "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"acc={acc:.3f} f1={f1:.3f} sens={sensitivity:.3f} spec={specificity:.3f}"
            )

    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)
        loocv_suffix = "_loocv" if use_loocv else ""
        tag = f"{target_type}_{window_id if window_id is not None else 'val'}{loocv_suffix}"
        # Confusion matrix plot
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {target_type} ({tag})", fontsize=12)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["class0", "class1"] if not classes else classes, fontsize=9)
        ax.set_yticklabels(["class0", "class1"] if not classes else classes, fontsize=9)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.text(
            0.5,
            0.02,
            f"F1: {f1:.3f}   Sensitivity: {sensitivity:.3f}   Specificity: {specificity:.3f}",
            ha="center",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0.05, 1, 0.97])
        plt.savefig(os.path.join(plots_dir, f"confusion_matrix_{tag}.png"), dpi=200)
        plt.close(fig)

        # Learning curve plot
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(train_losses, label="train")
        ax.plot(val_losses, label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Learning Curve")
        ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"learning_curve_{tag}.png"), dpi=200)
        plt.close(fig)

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "classes": classes,
        "val_accuracy": acc,
        "val_f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "use_loocv": use_loocv,
        "successful_folds": successful_folds,
        "confusion_matrix": cm,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


def run_experiment(
    target_type: str = "social_connection",
    time_windows: Iterable[int] = (3, 6, 9, 12),
    propagate_labels: bool = False,
    use_accurate_method: bool = False,
    hidden_layers: Iterable[int] = (128, 64),
    weight_decay: float = 1e-4,
    use_weighted_sampler: bool = True,
    epochs: int = 15,
    use_subwindows: bool = False,
    lookback_hours: int = 12,
    subwindow_hours: int = 3,
    standardized: bool = True,
    use_loocv: bool = False,
):
    pipeline = _select_pipeline(
        target_type,
        time_windows,
        propagate_labels,
        use_accurate_method,
        use_subwindows,
        lookback_hours,
        subwindow_hours,
        standardized,
    )
    pipeline.fit(None)
    merged_by_window: Dict[int, np.ndarray] = pipeline.transform(None)

    results = {}
    for window, df in merged_by_window.items():
        if df is None or df.empty:
            print(f"Window {window}h skipped: no data")
            continue
        try:
            print(f"\n=== Training MLP for {target_type} | window {window}h ===")
            results[window] = train_one_window(
                df,
                target_type=target_type,
                hidden_layers=hidden_layers,
                weight_decay=weight_decay,
                use_weighted_sampler=use_weighted_sampler,
                epochs=epochs,
                window_id=window,
                use_loocv=use_loocv,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Window {window}h failed: {exc}")
    return results


if __name__ == "__main__":
    run_experiment(
        target_type=os.getenv("TARGET_TYPE", "sleep"),
        time_windows=[int(x) for x in os.getenv("TIME_WINDOWS", "15,16,17,18,19,20,21,21,23,24,25").split(",")],
        propagate_labels=os.getenv("PROPAGATE_LABELS", "false").lower() == "true",
        use_accurate_method=os.getenv("USE_ACCURATE_METHOD", "false").lower() == "true",
        hidden_layers=(128, 64),
        weight_decay=float(os.getenv("WEIGHT_DECAY", "0.0001")),
        use_weighted_sampler=os.getenv("USE_WEIGHTED_SAMPLER", "true").lower() == "true",
        epochs=int(os.getenv("EPOCHS", "20")),
        use_subwindows=os.getenv("USE_SUBWINDOWS", "true").lower() == "true",
        lookback_hours=int(os.getenv("LOOKBACK_HOURS", "24")),
        subwindow_hours=int(os.getenv("SUBWINDOW_HOURS", "3")),
        standardized=os.getenv("STANDARDIZED", "true").lower() == "true",
        use_loocv=os.getenv("USE_LOOCV", "true").lower() == "true",
    )
