import os
import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
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


def _prepare_features_labels(df, target_type: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    # Drop rows with missing target
    df = df.copy()
    label_encoder = LabelEncoder(target_type=target_type)
    label_encoder.fit(df)
    label_series = label_encoder.transform(df)
    if label_series is None:
        return np.array([]), np.array([]), [], []
    df = df.loc[label_series.notna()]
    label_series = label_series.loc[label_series.notna()]

    selector = FeatureSelector()
    selector.fit(df)
    feature_df = selector.transform(df)

    # Keep only numeric columns (datetime/object columns cause float cast errors)
    numeric_df = feature_df.select_dtypes(include=[np.number]).copy()
    dropped_cols = set(feature_df.columns) - set(numeric_df.columns)
    if dropped_cols:
        print(f"Dropping non-numeric feature columns: {sorted(dropped_cols)}")

    if numeric_df.empty:
        return np.array([]), np.array([]), [], []

    numeric_df = numeric_df.fillna(0)

    X = numeric_df.to_numpy(dtype=float)
    classes = list(label_encoder.classes_ or [])
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    y = label_series.map(label_map).to_numpy()
    return X, y, list(numeric_df.columns), classes


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


def _make_loaders(X_train, X_val, y_train, y_val, batch_size: int = 64):
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def _compute_class_weights(y: np.ndarray) -> Optional[torch.Tensor]:
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return None
    weights = counts.sum() / (len(classes) * counts)
    full_weights = np.ones(int(classes.max()) + 1)
    for cls, w in zip(classes, weights):
        full_weights[int(cls)] = w
    return torch.tensor(full_weights, dtype=torch.float32)


def train_one_window(
    df,
    target_type: str,
    hidden_layers: Iterable[int] = (128, 64),
    dropout: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 20,
    batch_size: int = 64,
):
    X, y, feature_names, classes = _prepare_features_labels(df, target_type)
    if X.size == 0 or y.size == 0:
        raise ValueError("No data available after preprocessing")
    X_train, X_val, y_train, y_val, scaler = _train_val_split(X, y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = ScreentimeMLP(input_dim=X_train.shape[1], hidden_layers=hidden_layers, dropout=dropout).to(device)
    class_weights = _compute_class_weights(y_train)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader, val_loader = _make_loaders(X_train, X_val, y_train, y_val, batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            # 1: Zero the gradients
            optimizer.zero_grad()
            # 2: Make predictions
            logits = model(xb)
            # 3: Calculate loss
            loss = criterion(logits, yb)
            # 4: Backward pass
            loss.backward()
            # 5: Update weights
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        # Evaluate model
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                targets.append(yb.numpy())
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, zero_division=0)
        print(f"Epoch {epoch+1}/{epochs} - loss={epoch_loss/len(train_loader.dataset):.4f} acc={acc:.3f} f1={f1:.3f}")

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "classes": classes,
        "val_accuracy": acc,
        "val_f1": f1,
    }


def run_experiment(
    target_type: str = "suicide_risk",
    time_windows: Iterable[int] = (3, 6, 9, 12),
    propagate_labels: bool = False,
    use_accurate_method: bool = False,
    hidden_layers: Iterable[int] = (128, 64),
    epochs: int = 15,
    use_subwindows: bool = False,
    lookback_hours: int = 12,
    subwindow_hours: int = 3,
    standardized: bool = True,
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
                epochs=epochs,
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
        epochs=int(os.getenv("EPOCHS", "15")),
        use_subwindows=os.getenv("USE_SUBWINDOWS", "true").lower() == "true",
        lookback_hours=int(os.getenv("LOOKBACK_HOURS", "24")),
        subwindow_hours=int(os.getenv("SUBWINDOW_HOURS", "3")),
        standardized=os.getenv("STANDARDIZED", "true").lower() == "true",
    )
