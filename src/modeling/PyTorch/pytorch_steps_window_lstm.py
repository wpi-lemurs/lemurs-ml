import os
import random
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.pipeline.mental_health_pipeline import (
    create_step_phq9_pipeline,
    create_step_risk_pipeline,
)
from src.pipeline.transformers import LabelEncoder


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def _select_pipeline(
    target_type: str,
    time_windows: Iterable[int],
    propagate_labels: bool,
    average_shared_labels: bool = True,
):
    if target_type == "phq9":
        return create_step_phq9_pipeline(
            time_windows=list(time_windows),
            propagate_labels=propagate_labels,
        )
    # Sleep labels are derived from morning survey sleep items, so shared-label
    # averaging must be disabled to preserve consistent target construction.
    effective_average_shared_labels = False if target_type == "sleep" else average_shared_labels

    return create_step_risk_pipeline(
        target_type=target_type,
        time_windows=list(time_windows),
        propagate_labels=propagate_labels,
        average_shared_labels=effective_average_shared_labels,
    )


def _prepare_step_sequences_labels(
    df: pd.DataFrame,
    target_type: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.Index, List[str]]:
    df = df.copy()

    label_encoder = LabelEncoder(target_type=target_type)
    label_encoder.fit(df)
    y_series = label_encoder.transform(df)
    if y_series is None:
        return np.array([]), np.array([]), [], pd.Index([]), []

    valid_mask = y_series.notna()
    df = df.loc[valid_mask].copy()
    y_series = y_series.loc[valid_mask]

    hour_cols = [c for c in df.columns if c.startswith("hour_")]
    if not hour_cols:
        return np.array([]), np.array([]), [], pd.Index([]), []

    hour_cols = sorted(hour_cols, key=lambda x: int(x.split("_")[1]))
    seq_cols = list(reversed(hour_cols))  # oldest -> newest

    X_2d = df[seq_cols].fillna(0).to_numpy(dtype=np.float32)
    X = np.expand_dims(X_2d, axis=-1)

    classes = list(label_encoder.classes_ or [])
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    y = y_series.map(label_map).to_numpy(dtype=np.int64)

    return X, y, classes, df.index, seq_cols


def _make_loaders(X_train, X_val, y_train, y_val, batch_size=32, use_weighted_sampler=True):
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
    if cm is None or cm.shape != (2, 2):
        return float("nan"), float("nan")
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity, specificity


def _balanced_accuracy_from_sens_spec(sensitivity: float, specificity: float) -> float:
    return (sensitivity + specificity) / 2.0


def _mean_variable_length_sequences(sequences: List[List[float]]) -> List[float]:
    valid = [seq for seq in sequences if seq]
    if not valid:
        return []

    max_len = max(len(seq) for seq in valid)
    padded = np.full((len(valid), max_len), np.nan, dtype=float)
    for i, seq in enumerate(valid):
        padded[i, : len(seq)] = seq

    return np.nanmean(padded, axis=0).tolist()


class StepLSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, dropout=0.2, debug_shapes=True):
        super().__init__()
        self.debug_shapes = debug_shapes
        self._shape_debug_logged = False
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x):
        if self.debug_shapes and not self._shape_debug_logged:
            print(f"[LSTM DEBUG] Input shape before LSTM: {tuple(x.shape)}")
        out, _ = self.lstm(x)
        if self.debug_shapes and not self._shape_debug_logged:
            print(f"[LSTM DEBUG] Output shape after LSTM: {tuple(out.shape)}")
            self._shape_debug_logged = True
        last_hidden = out[:, -1, :]
        return self.classifier(last_hidden)


def _train_evaluate_split(
    X_train,
    X_eval,
    y_train,
    y_eval,
    hidden_size,
    num_layers,
    dropout,
    lr,
    weight_decay,
    use_weighted_sampler,
    epochs,
    batch_size,
    device,
    X_early_stop=None,
    y_early_stop=None,
    use_early_stopping=True,
    early_stopping_patience=5,
    early_stopping_min_delta=1e-4,
    min_epochs=5,
    debug_shapes=True,
):
    if X_early_stop is None:
        X_early_stop = X_eval
        y_early_stop = y_eval

    model = StepLSTMClassifier(
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        debug_shapes=debug_shapes,
    ).to(device)

    class_weights = _compute_class_weights(y_train)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader, val_loader = _make_loaders(
        X_train,
        X_early_stop,
        y_train,
        y_early_stop,
        batch_size=batch_size,
        use_weighted_sampler=use_weighted_sampler,
    )

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve_count = 0
    stopped_early = False
    best_state = deepcopy(model.state_dict())

    for epoch in range(epochs):
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
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss = criterion(logits, yb)
                val_loss_sum += val_loss.item() * xb.size(0)

        current_val_loss = val_loss_sum / len(val_loader.dataset)
        val_losses.append(current_val_loss)

        improved = current_val_loss < (best_val_loss - early_stopping_min_delta)
        if improved:
            best_val_loss = current_val_loss
            best_epoch = epoch + 1
            no_improve_count = 0
            best_state = deepcopy(model.state_dict())
        else:
            no_improve_count += 1

        can_stop = use_early_stopping and early_stopping_patience and early_stopping_patience > 0
        if can_stop and (epoch + 1) >= max(1, min_epochs) and no_improve_count >= early_stopping_patience:
            stopped_early = True
            break

    model.load_state_dict(best_state)

    eval_ds = TensorDataset(torch.tensor(X_eval, dtype=torch.float32), torch.tensor(y_eval, dtype=torch.long))
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
    model.eval()
    eval_preds, eval_targets = [], []

    with torch.no_grad():
        for xb, yb in eval_loader:
            xb = xb.to(device)
            logits = model(xb)
            eval_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            eval_targets.append(yb.numpy())

    preds = np.concatenate(eval_preds)
    targets = np.concatenate(eval_targets)

    return {
        "model": model,
        "preds": preds,
        "targets": targets,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "epochs_ran": len(train_losses),
        "stopped_early": stopped_early,
    }


def train_one_window(
    df,
    target_type,
    hidden_size=64,
    num_layers=1,
    dropout=0.2,
    lr=1e-3,
    weight_decay=1e-4,
    use_weighted_sampler=True,
    epochs=20,
    batch_size=32,
    window_id=None,
    save_plots=True,
    plots_dir="data",
    use_loocv=False,
    group_col="app_user_id",
    inner_val_size=0.2,
    use_early_stopping=True,
    early_stopping_patience=5,
    early_stopping_min_delta=1e-4,
    min_epochs=5,
    debug_shapes=True,
):
    X, y, classes, row_index, seq_cols = _prepare_step_sequences_labels(df, target_type)
    if X.size == 0 or y.size == 0:
        raise ValueError("No data available after preprocessing")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = None
    cm = None
    train_losses, val_losses = [], []
    successful_folds = 1
    acc = float("nan")
    f1 = float("nan")
    sensitivity = float("nan")
    specificity = float("nan")
    balanced_accuracy = float("nan")

    if use_loocv:
        if group_col not in df.columns:
            raise ValueError(f"LOOCV requires '{group_col}' column in input dataframe")

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
        fold_accs = []
        fold_f1s = []
        fold_sens = []
        fold_spec = []
        fold_bal_acc = []
        successful_folds = 0

        for train_idx, test_idx in logo.split(X_cv, y_cv, groups_cv):
            X_train_fold = X_cv[train_idx]
            y_train_fold = y_cv[train_idx]
            X_test_fold = X_cv[test_idx]
            y_test_fold = y_cv[test_idx]

            if np.unique(y_train_fold).size < 2 or y_test_fold.size == 0:
                continue

            stratify_labels = y_train_fold if np.unique(y_train_fold).size > 1 else None
            try:
                X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
                    X_train_fold,
                    y_train_fold,
                    test_size=inner_val_size,
                    random_state=42,
                    stratify=stratify_labels,
                )
            except ValueError:
                continue

            if np.unique(y_inner_train).size < 2 or len(y_inner_val) == 0:
                continue

            fold_result = _train_evaluate_split(
                X_train=X_inner_train,
                X_eval=X_test_fold,
                y_train=y_inner_train,
                y_eval=y_test_fold,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                lr=lr,
                weight_decay=weight_decay,
                use_weighted_sampler=use_weighted_sampler,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
                X_early_stop=X_inner_val,
                y_early_stop=y_inner_val,
                use_early_stopping=use_early_stopping,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
                min_epochs=min_epochs,
                debug_shapes=debug_shapes,
            )

            successful_folds += 1
            model = fold_result["model"]
            fold_preds = fold_result["preds"]
            fold_targets = fold_result["targets"]
            all_preds.append(fold_preds)
            all_targets.append(fold_targets)
            fold_train_losses.append(fold_result["train_losses"])
            fold_val_losses.append(fold_result["val_losses"])

            fold_acc = accuracy_score(fold_targets, fold_preds)
            fold_f1 = f1_score(fold_targets, fold_preds, zero_division=0)
            fold_cm = confusion_matrix(fold_targets, fold_preds, labels=[0, 1])
            fold_sens, fold_spec_val = _binary_metrics_from_cm(fold_cm)
            fold_bal = _balanced_accuracy_from_sens_spec(fold_sens, fold_spec_val)

            fold_accs.append(fold_acc)
            fold_f1s.append(fold_f1)
            fold_sens.append(fold_sens)
            fold_spec.append(fold_spec_val)
            fold_bal_acc.append(fold_bal)

        if successful_folds == 0:
            raise ValueError("No valid LOOCV folds were available for training")

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        acc = float(np.mean(fold_accs))
        f1 = float(np.mean(fold_f1s))
        sensitivity = float(np.mean(fold_sens))
        specificity = float(np.mean(fold_spec))
        balanced_accuracy = float(np.mean(fold_bal_acc))

        cm = confusion_matrix(targets, preds, labels=[0, 1])
        train_losses = _mean_variable_length_sequences(fold_train_losses)
        val_losses = _mean_variable_length_sequences(fold_val_losses)
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None,
        )

        split_result = _train_evaluate_split(
            X_train=X_train,
            X_eval=X_val,
            y_train=y_train,
            y_eval=y_val,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            use_weighted_sampler=use_weighted_sampler,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            X_early_stop=X_val,
            y_early_stop=y_val,
            use_early_stopping=use_early_stopping,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            min_epochs=min_epochs,
            debug_shapes=debug_shapes,
        )

        model = split_result["model"]
        preds = split_result["preds"]
        targets = split_result["targets"]
        train_losses = split_result["train_losses"]
        val_losses = split_result["val_losses"]

        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, zero_division=0)
        cm = confusion_matrix(targets, preds, labels=[0, 1])
        sensitivity, specificity = _binary_metrics_from_cm(cm)
        balanced_accuracy = _balanced_accuracy_from_sens_spec(sensitivity, specificity)

    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)
        loocv_suffix = "_loocv" if use_loocv else ""
        tag = f"{target_type}_{window_id if window_id is not None else 'val'}{loocv_suffix}"

        fig = plt.figure(figsize=(5.5, 5.5))
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[5.0, 1.3], hspace=0.25)
        ax = fig.add_subplot(gs[0, 0])
        metrics_ax = fig.add_subplot(gs[1, 0])

        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Confusion Matrix: LSTM {target_type} ({tag})", fontsize=12, pad=10, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["class0", "class1"] if not classes else classes, fontsize=9)
        ax.set_yticklabels(["class0", "class1"] if not classes else classes, fontsize=9)

        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=10)

        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

        metrics_ax.axis("off")
        metrics_text = (
            f"F1: {f1:.3f}    Sensitivity: {sensitivity:.3f}\n"
            f"Specificity: {specificity:.3f}    Balanced Accuracy: {balanced_accuracy:.3f}"
        )
        metrics_ax.text(0.5, 0.5, metrics_text, ha="center", va="center", fontsize=10)

        fig.subplots_adjust(left=0.10, right=0.86, top=0.90, bottom=0.09)
        plt.savefig(os.path.join(plots_dir, f"lstm_confusion_matrix_{tag}.png"), dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.plot(train_losses, label="train")
        ax.plot(val_losses, label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("LSTM Learning Curve", fontweight="bold")
        ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"lstm_learning_curve_{tag}.png"), dpi=200)
        plt.close(fig)

    return {
        "model": model,
        "val_accuracy": acc,
        "val_f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "confusion_matrix": cm,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "successful_folds": successful_folds,
        "sequence_features": seq_cols,
    }


def run_experiment(
    target_type="sleep",
    time_windows=(24),
    propagate_labels=False,
    average_shared_labels=False,
    hidden_size=64,
    num_layers=1,
    dropout=0.2,
    lr=1e-3,
    weight_decay=1e-4,
    use_weighted_sampler=True,
    epochs=20,
    batch_size=32,
    use_loocv=False,
    inner_val_size=0.2,
    use_early_stopping=True,
    early_stopping_patience=5,
    early_stopping_min_delta=1e-4,
    min_epochs=5,
    debug_shapes=False,
):
    effective_average_shared_labels = False if target_type == "sleep" else average_shared_labels
    if target_type == "sleep" and average_shared_labels:
        print("[INFO] Ignoring average_shared_labels=True for sleep target; using morning-only sleep labels.")

    pipeline = _select_pipeline(
        target_type=target_type,
        time_windows=time_windows,
        propagate_labels=propagate_labels,
        average_shared_labels=effective_average_shared_labels,
    )

    pipeline.fit(None)
    merged_by_window: Dict[int, pd.DataFrame] = pipeline.transform(None)

    results = {}
    for window, df in merged_by_window.items():
        if df is None or df.empty:
            print(f"Window {window}h skipped: no data")
            continue

        try:
            print(f"\n=== Training LSTM for {target_type} | window {window}h ===")
            results[window] = train_one_window(
                df=df,
                target_type=target_type,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                lr=lr,
                weight_decay=weight_decay,
                use_weighted_sampler=use_weighted_sampler,
                epochs=epochs,
                batch_size=batch_size,
                window_id=window,
                use_loocv=use_loocv,
                inner_val_size=inner_val_size,
                use_early_stopping=use_early_stopping,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
                min_epochs=min_epochs,
                debug_shapes=debug_shapes,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Window {window}h failed: {exc}")

    return results


def print_summary(results: Dict[int, dict]):
    """Print compact per-window metrics and the best-performing window."""
    print("\n" + "=" * 80)
    print("LSTM SUMMARY - ALL TIME WINDOWS")
    print("=" * 80)

    if not results:
        print("No successful windows to summarize.")
        return

    print(f"{'Window':>8} {'F1':>10} {'Accuracy':>10} {'Sensitivity':>12} {'Specificity':>12} {'Bal Acc':>10}")
    print("-" * 80)

    best_window = None
    best_f1 = float("-inf")
    best_acc = float("-inf")

    for window in sorted(results.keys()):
        metrics = results[window]
        f1 = metrics.get("val_f1")
        acc = metrics.get("val_accuracy")
        sens = metrics.get("sensitivity")
        spec = metrics.get("specificity")
        bal_acc = metrics.get("balanced_accuracy")

        print(
            f"{str(window) + 'h':>8} "
            f"{f1:>10.4f} {acc:>10.4f} {sens:>12.4f} {spec:>12.4f} {bal_acc:>10.4f}"
        )

        if f1 is not None and not np.isnan(f1):
            if f1 > best_f1:
                best_f1 = f1
                best_acc = acc if acc is not None else float("-inf")
                best_window = window
        elif best_window is None and acc is not None and not np.isnan(acc):
            if acc > best_acc:
                best_acc = acc
                best_window = window

    print("\nBest window:")
    if best_window is None:
        print("  No valid best window found.")
        return

    best_metrics = results[best_window]
    print(
        f"  {best_window}h | "
        f"F1={best_metrics.get('val_f1', float('nan')):.4f}, "
        f"Accuracy={best_metrics.get('val_accuracy', float('nan')):.4f}, "
        f"Sensitivity={best_metrics.get('sensitivity', float('nan')):.4f}, "
        f"Specificity={best_metrics.get('specificity', float('nan')):.4f}, "
        f"Balanced Accuracy={best_metrics.get('balanced_accuracy', float('nan')):.4f}"
    )


if __name__ == "__main__":
    results = run_experiment(
        target_type=os.getenv("TARGET_TYPE", "suicide_risk"),
        time_windows=[int(x) for x in os.getenv("TIME_WINDOWS", "24").split(",")],
        propagate_labels=os.getenv("PROPAGATE_LABELS", "false").lower() == "true",
        average_shared_labels=os.getenv("AVERAGE_SHARED_LABELS", "true").lower() == "true",
        hidden_size=int(os.getenv("HIDDEN_SIZE", "16")),
        num_layers=int(os.getenv("NUM_LAYERS", "2")),
        dropout=float(os.getenv("DROPOUT", "0.2")),
        lr=float(os.getenv("LEARNING_RATE", "0.001")),
        weight_decay=float(os.getenv("WEIGHT_DECAY", "0.0001")),
        use_weighted_sampler=os.getenv("USE_WEIGHTED_SAMPLER", "false").lower() == "true",
        epochs=int(os.getenv("EPOCHS", "70")),
        batch_size=int(os.getenv("BATCH_SIZE", "32")),
        use_loocv=os.getenv("USE_LOOCV", "false").lower() == "true",
        use_early_stopping=os.getenv("USE_EARLY_STOPPING", "false").lower() == "true",
        debug_shapes=os.getenv("DEBUG_SHAPES", "true").lower() == "true",
    )
    print_summary(results)


