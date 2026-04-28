"""Run a sweep of the step-based LSTM experiment across multiple risk labels.

This script reuses the existing `run_experiment` helper from
`pytorch_steps_window_lstm.py` and collects a compact results table for:
- sleep
- suicide_risk
- self_harm
- social_connection

Each target is run with and without LOOCV. The resulting table includes the
metrics that matter most for comparison:
- F1 score
- balanced accuracy
- recall (mapped from the model's sensitivity metric)

The script prints both a detailed per-window table and a best-window summary
for each target / LOOCV setting.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.PyTorch.pytorch_steps_window_lstm import run_experiment

TARGETS = ("sleep", "suicide_risk", "self_harm", "social_connection")
DEFAULT_TIME_WINDOWS = (24,)


def _to_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _collect_window_rows(
    target_type: str,
    use_loocv: bool,
    experiment_results: dict,
) -> List[dict]:
    rows: List[dict] = []

    for time_window, metrics in experiment_results.items():
        if not isinstance(metrics, dict):
            continue

        rows.append(
            {
                "target_type": target_type,
                "loocv": use_loocv,
                "time_window": int(time_window),
                "f1_score": _to_float(metrics.get("val_f1")),
                "balanced_accuracy": _to_float(metrics.get("balanced_accuracy")),
                "recall": _to_float(metrics.get("sensitivity")),
                "accuracy": _to_float(metrics.get("val_accuracy")),
                "successful_folds": metrics.get("successful_folds"),
                "train_loss_last": _to_float(metrics.get("train_losses", [None])[-1] if metrics.get("train_losses") else None),
                "val_loss_last": _to_float(metrics.get("val_losses", [None])[-1] if metrics.get("val_losses") else None),
            }
        )

    return rows


def _best_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if df.empty:
        return None

    ranked = df.sort_values(
        by=["f1_score", "balanced_accuracy", "recall", "accuracy"],
        ascending=[False, False, False, False],
        na_position="last",
    )
    return ranked.iloc[0]


def run_lstm_risk_experiment_suite(
    targets: Iterable[str] = TARGETS,
    time_windows: Iterable[int] = DEFAULT_TIME_WINDOWS,
    propagate_labels: bool = False,
    average_shared_labels: bool = True,
    hidden_size: int = 16,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    use_weighted_sampler: bool = False,
    epochs: int = 50,
    batch_size: int = 32,
    use_early_stopping: bool = False,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 1e-4,
    min_epochs: int = 5,
    debug_shapes: bool = False,
    one_survey_per_day: bool = True,
) -> pd.DataFrame:
    """Run the LSTM experiment for every target with and without LOOCV.

    Parameters:
    -----------
    one_survey_per_day : bool, default=True
        If True, prevents data leakage by ensuring only one survey per user per day
        is used for merging with health data. For sleep (morning-only), keeps only
        morning surveys. For other labels, keeps first survey of day.
        Default True to prevent data leakage.
    """
    all_rows: List[dict] = []

    for target_type in targets:
        for use_loocv in (False, True):
            print("=" * 80)
            print(f"RUNNING TARGET: {target_type} | LOOCV: {use_loocv}")
            print("=" * 80)

            experiment_results = run_experiment(
                target_type=target_type,
                time_windows=list(time_windows),
                propagate_labels=propagate_labels,
                average_shared_labels=average_shared_labels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                lr=lr,
                weight_decay=weight_decay,
                use_weighted_sampler=use_weighted_sampler,
                epochs=epochs,
                batch_size=batch_size,
                use_loocv=use_loocv,
                inner_val_size=0.2,
                use_early_stopping=use_early_stopping,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
                min_epochs=min_epochs,
                debug_shapes=debug_shapes,
                one_survey_per_day=one_survey_per_day,
            )

            if not experiment_results:
                print(f"No successful results for {target_type} | LOOCV={use_loocv}")
                continue

            all_rows.extend(_collect_window_rows(target_type, use_loocv, experiment_results))

    results_df = pd.DataFrame(all_rows)
    if results_df.empty:
        return results_df

    results_df = results_df.sort_values(
        by=["target_type", "loocv", "f1_score", "balanced_accuracy", "recall", "accuracy"],
        ascending=[True, True, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return results_df


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create one best-row summary per target / LOOCV setting."""
    if results_df.empty:
        return results_df

    summary_rows: List[dict] = []
    for (target_type, loocv), group_df in results_df.groupby(["target_type", "loocv"], sort=True):
        best = _best_row(group_df)
        if best is None:
            continue
        summary_rows.append(
            {
                "target_type": target_type,
                "loocv": loocv,
                "best_window": int(best["time_window"]),
                "f1_score": _to_float(best["f1_score"]),
                "balanced_accuracy": _to_float(best["balanced_accuracy"]),
                "recall": _to_float(best["recall"]),
                "accuracy": _to_float(best["accuracy"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["f1_score", "balanced_accuracy", "recall", "accuracy"],
            ascending=[False, False, False, False],
            na_position="last",
        ).reset_index(drop=True)
    return summary_df


def main() -> None:
    results_df = run_lstm_risk_experiment_suite()

    print("\n" + "=" * 80)
    print("LSTM RESULTS TABLE")
    print("=" * 80)

    if results_df.empty:
        print("No results were produced.")
        return

    display_columns = [
        "target_type",
        "loocv",
        "time_window",
        "f1_score",
        "balanced_accuracy",
        "recall",
        "accuracy",
        "successful_folds",
    ]
    print(results_df[display_columns].to_string(index=False))

    summary_df = summarize_results(results_df)
    print("\n" + "=" * 80)
    print("BEST WINDOW BY TARGET / LOOCV")
    print("=" * 80)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()


