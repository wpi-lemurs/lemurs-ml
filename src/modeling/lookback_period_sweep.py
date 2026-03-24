"""
Lookback Period Sweep Analysis (Optimized)

Sweeps over lookback periods 1-30 hours and, for each lookback period,
tries every valid subwindow size (all divisors of the lookback).

OPTIMIZATION: This script categorizes apps ONCE at the beginning and reuses
the categorized data for all parameter combinations, making it dramatically
faster than the naive approach.

Results are visualized as:
- Heatmaps – F1 / Accuracy across the full lookback × subwindow grid
- Line graphs – best subwindow per lookback period
- Subwindow comparison – F1 vs subwindow size for selected lookbacks
- Summary table – top-10 configurations by F1

Usage:
    python src/modeling/lookback_period_sweep.py

Configuration:
    Edit the constants at the top of the main() function to change:
    - TARGET_TYPE: 'suicide_risk', 'self_harm', 'sleep', or 'phq9'
    - MAX_LOOKBACK: maximum lookback period in hours (default: 30)
    - PROPAGATE_LABELS: whether to propagate positive labels (default: True)
    - BALANCED_WEIGHT: use balanced class weights (default: False)
    - USE_LOOCV: leave-one-user-out CV (disabled for sweep - too slow)

Performance:
    With optimization, a 30-hour sweep with ~120 combinations takes minutes
    instead of hours, since app categorization happens only once.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script execution
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.transformers import ScreentimeAppCategorizer
from src.pipeline.mental_health_pipeline import create_subwindow_pipeline
from src.config import DATA_DIR


def get_subwindow_sizes(lookback: int) -> list:
    """
    Return all divisors of lookback in descending order.

    These are the valid subwindow sizes for a given lookback period.
    For example, lookback=12 returns [12, 6, 4, 3, 2, 1].

    Parameters:
    -----------
    lookback : int
        Lookback period in hours

    Returns:
    --------
    list : Valid subwindow sizes in descending order
    """
    return sorted(
        [sw for sw in range(1, lookback + 1) if lookback % sw == 0],
        reverse=True,
    )


def run_sweep(
    target_type: str,
    max_lookback: int,
    propagate_labels: bool = True,
    balanced_weight: bool = False,
    use_loocv: bool = False,
) -> pd.DataFrame:
    """
    Run the lookback period sweep with optimized categorization.

    OPTIMIZATION: Categorizes apps ONCE at the beginning, then reuses the
    categorized data for all (lookback, subwindow) combinations. This makes
    the sweep dramatically faster!

    For every (lookback, subwindow) pair, creates features using the pre-categorized
    data and trains models, recording F1 score and accuracy for each model.

    Parameters:
    -----------
    target_type : str
        Target to predict: 'suicide_risk', 'self_harm', 'sleep', or 'phq9'
    max_lookback : int
        Maximum lookback period in hours
    propagate_labels : bool, default=True
        Whether to propagate positive labels across a user
    balanced_weight : bool, default=False
        Whether to use balanced class weights
    use_loocv : bool, default=False
        Whether to use leave-one-user-out cross-validation

    Returns:
    --------
    pd.DataFrame : Results with columns [lookback, subwindow, model, f1_score, accuracy, n_samples]
    """
    records = []

    lookbacks = list(range(1, max_lookback + 1))
    total_combinations = sum(len(get_subwindow_sizes(lb)) for lb in lookbacks)

    print(f'\n{"=" * 80}')
    print(f'STARTING OPTIMIZED LOOKBACK SWEEP')
    print(f'{"=" * 80}')
    print(f'Target type      : {target_type}')
    print(f'Max lookback     : {max_lookback} h')
    print(f'Propagate labels : {propagate_labels}')
    print(f'Balanced weights : {balanced_weight}')
    print(f'LOOCV            : {use_loocv}')
    print(f'Total combos     : {total_combinations}')
    print(f'{"=" * 80}\n')

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: CATEGORIZE APPS ONCE (this is the slow part)
    # ═══════════════════════════════════════════════════════════════════════
    print("STEP 1: Categorizing apps...")
    print("-" * 80)

    categorizer = ScreentimeAppCategorizer()
    categorizer.fit(None)  # This does the expensive categorization
    categorized_data = categorizer.transform(None)

    print(f"[OK] Categorization complete! Now running {total_combinations} experiments...\n")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: RUN EXPERIMENTS REUSING CATEGORIZED DATA (fast!)
    # ═══════════════════════════════════════════════════════════════════════
    done = 0

    for lookback in lookbacks:
        for subwindow in get_subwindow_sizes(lookback):
            done += 1
            print(f'[{done}/{total_combinations}]  lookback={lookback:2d}h  subwindow={subwindow:2d}h', end='  ')

            try:
                # Create a pipeline for this specific configuration
                # The categorized_data will be passed through the pipeline
                pipeline = create_subwindow_pipeline(
                    target_type=target_type,
                    lookback_hours=lookback,
                    subwindow_hours=subwindow,
                    propagate_labels=propagate_labels,
                    use_accurate_method=True,
                    standardized=False
                )

                # Fit the pipeline (but categorization is skipped since data is pre-categorized)
                # The ScreentimeAppCategorizer step will use the cached data
                pipeline.named_steps['categorize_apps'].categorized_data_ = categorized_data

                # Transform to get features (fast - uses pre-categorized data)
                processed_data = pipeline.transform(categorized_data)

                if not processed_data or (isinstance(processed_data, dict) and not processed_data):
                    print('→ no data')
                    continue

                # Now train models on the processed data
                from sklearn.model_selection import train_test_split, LeaveOneGroupOut
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LogisticRegression
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import f1_score, accuracy_score

                # Get the dataframe from the dict
                df = processed_data[lookback]

                if df.empty:
                    print('→ empty dataframe')
                    continue

                # Determine label column
                if target_type == 'phq9':
                    label_col = 'severity_label'
                    positive_class = 'depressed'
                else:
                    label_col_map = {
                        'suicide_risk': 'suicide_risk_label',
                        'self_harm': 'self_harm_risk_label',
                        'sleep': 'sleep_label'
                    }
                    label_col = label_col_map[target_type]
                    positive_class = 'at_risk'

                # Filter out N/A for sleep
                if target_type == 'sleep':
                    df = df[df[label_col] != 'N/A'].copy()

                # Check if we have data and both classes
                if df.empty or df[label_col].nunique() < 2:
                    print('→ insufficient data')
                    continue

                # Prepare features and labels
                exclude_cols = [
                    'app_user_id', 'survey_response_id', 'timestamp', 'survey_timestamp',
                    'week_start', 'time_key', 'date', 'phq9_response_id',
                    label_col, 'phq9_total_score'
                ]

                # Also exclude datetime columns
                datetime_cols = df.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
                exclude_cols.extend(datetime_cols)
                exclude_cols = list(set(exclude_cols))

                feature_cols = [col for col in df.columns if col not in exclude_cols]

                if not feature_cols:
                    print('→ no features')
                    continue

                X = df[feature_cols].copy()

                # One-hot encode categorical columns
                non_numeric_cols = X.select_dtypes(exclude=['number']).columns.tolist()
                if non_numeric_cols:
                    X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=True, dummy_na=False)
                    bool_cols = X.select_dtypes(include=['bool']).columns
                    X[bool_cols] = X[bool_cols].astype(int)

                X = X.fillna(0)
                y = df[label_col]

                # Check minimum samples per class
                class_counts = y.value_counts()
                if class_counts.min() < 2:
                    print('→ too few samples per class')
                    continue

                n_samples = len(df)

                # Train models
                class_weight = 'balanced' if balanced_weight else None

                models = {
                    'logistic_regression': LogisticRegression(
                        max_iter=10000, random_state=42, class_weight=class_weight, solver='lbfgs'
                    ),
                    'random_forest': RandomForestClassifier(
                        n_estimators=100, random_state=42, class_weight=class_weight
                    )
                }

                # Simple train/test split for speed (LOOCV would be too slow for sweep)
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42, stratify=y
                    )
                except:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42
                    )

                for model_name, model in models.items():
                    # Scale for Logistic Regression
                    if 'logistic' in model_name:
                        scaler = StandardScaler()
                        X_train_model = scaler.fit_transform(X_train)
                        X_test_model = scaler.transform(X_test)
                    else:
                        X_train_model = X_train
                        X_test_model = X_test

                    # Train and evaluate
                    model.fit(X_train_model, y_train)
                    y_pred = model.predict(X_test_model)

                    acc = accuracy_score(y_test, y_pred)

                    try:
                        f1 = f1_score(y_test, y_pred, pos_label=positive_class, average='binary')
                    except:
                        f1 = None

                    records.append({
                        'lookback': lookback,
                        'subwindow': subwindow,
                        'model': model_name,
                        'f1_score': f1 if f1 is not None else np.nan,
                        'accuracy': acc if acc is not None else np.nan,
                        'n_samples': n_samples,
                    })

                    tag = f'{model_name[:2].upper()} f1={f1:.3f}' if f1 is not None else f'{model_name[:2].upper()} acc={acc:.3f}'
                    print(tag, end='  ')

            except Exception as e:
                print(f'→ ERROR: {e}', end='')

            print()

    print(f'\n{"=" * 80}')
    print(f'SWEEP COMPLETE - {len(records)} result rows collected')
    print(f'{"=" * 80}\n')

    return pd.DataFrame(records)


def plot_heatmaps(df: pd.DataFrame, metric: str, target_type: str) -> plt.Figure:
    """
    Create heatmap of metric across the lookback × subwindow grid.

    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe
    metric : str
        Metric to plot ('f1_score' or 'accuracy')
    target_type : str
        Target type for labeling

    Returns:
    --------
    plt.Figure : The created figure
    """
    models = df['model'].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(10 * len(models), 8))
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        sub = df[df['model'] == model_name]
        pivot = sub.pivot_table(
            index='subwindow', columns='lookback', values=metric, aggfunc='mean'
        ).sort_index(ascending=False)

        sns.heatmap(
            pivot,
            ax=ax,
            annot=True, fmt='.2f',
            cmap='RdYlGn', vmin=0, vmax=1,
            linewidths=0.3, linecolor='grey',
            cbar_kws={'label': metric.replace('_', ' ').title()},
            annot_kws={'size': 7},
        )
        ax.set_title(
            f'{model_name.replace("_", " ").title()}\n'
            f'{target_type.replace("_", " ").title()} – {metric.replace("_", " ").title()}',
            fontsize=12, fontweight='bold',
        )
        ax.set_xlabel('Lookback Period (hours)', fontsize=10)
        ax.set_ylabel('Subwindow Size (hours)', fontsize=10)

    plt.tight_layout()
    return fig


def plot_best_per_lookback(df: pd.DataFrame, target_type: str) -> plt.Figure:
    """
    Line graph showing best F1 and accuracy per lookback period.

    For each lookback, selects the subwindow with the highest F1 score.

    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe
    target_type : str
        Target type for labeling

    Returns:
    --------
    plt.Figure : The created figure
    """
    best = (
        df.sort_values('f1_score', ascending=False)
          .groupby(['lookback', 'model'], as_index=False)
          .first()
    )

    models = best['model'].unique()
    colors = plt.cm.tab10.colors
    fig, (ax_f1, ax_acc) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    for i, model_name in enumerate(models):
        sub = best[best['model'] == model_name].sort_values('lookback')
        label = model_name.replace('_', ' ').title()
        c = colors[i % len(colors)]

        ax_f1.plot(sub['lookback'], sub['f1_score'], marker='o', lw=2, ms=5, label=label, color=c)
        ax_acc.plot(sub['lookback'], sub['accuracy'], marker='s', lw=2, ms=5, label=label, color=c)

        # Annotate best-subwindow size
        for _, row in sub.iterrows():
            if not np.isnan(row['f1_score']):
                ax_f1.annotate(
                    f"{int(row['subwindow'])}h",
                    xy=(row['lookback'], row['f1_score']),
                    xytext=(0, 7), textcoords='offset points',
                    fontsize=7, ha='center', color=c,
                )

    ax_f1.set_title(
        f'{target_type.replace("_", " ").title()} – Best Subwindow F1 per Lookback Period',
        fontsize=13, fontweight='bold',
    )
    for ax, ylabel in [(ax_f1, 'F1 Score'), (ax_acc, 'Accuracy')]:
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.4)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    ax_acc.set_xlabel('Lookback Period (hours)', fontsize=11)
    plt.tight_layout()
    return fig


def plot_subwindow_comparison(df: pd.DataFrame, lookbacks_to_plot: list, target_type: str) -> plt.Figure:
    """
    F1 vs subwindow size for selected lookback periods.

    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe
    lookbacks_to_plot : list
        List of lookback periods to include
    target_type : str
        Target type for labeling

    Returns:
    --------
    plt.Figure : The created figure
    """
    models = df['model'].unique()
    colors = plt.cm.tab10.colors
    n_lb = len(lookbacks_to_plot)

    fig, axes = plt.subplots(1, n_lb, figsize=(6 * n_lb, 5), sharey=True)
    if n_lb == 1:
        axes = [axes]

    for ax, lb in zip(axes, lookbacks_to_plot):
        sub_lb = df[df['lookback'] == lb]
        if sub_lb.empty:
            ax.set_visible(False)
            continue

        for i, model_name in enumerate(models):
            sub = sub_lb[sub_lb['model'] == model_name].sort_values('subwindow')
            ax.plot(
                sub['subwindow'], sub['f1_score'],
                marker='o', lw=2, ms=6,
                label=model_name.replace('_', ' ').title(),
                color=colors[i % len(colors)],
            )

        ax.set_title(f'Lookback = {lb}h', fontsize=12, fontweight='bold')
        ax.set_xlabel('Subwindow Size (hours)', fontsize=10)
        ax.set_ylabel('F1 Score', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=9)
        ax.set_xticks(sorted(sub_lb['subwindow'].unique()))

    fig.suptitle(
        f'{target_type.replace("_", " ").title()} – F1 vs Subwindow Size',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    return fig


def print_top_configurations(df: pd.DataFrame, top_n: int = 10):
    """
    Print the top N configurations by F1 score.

    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe
    top_n : int, default=10
        Number of top configurations to display
    """
    if df.empty:
        print('No results available.')
        return

    top = (
        df
        .dropna(subset=['f1_score'])
        .sort_values('f1_score', ascending=False)
        .head(top_n)[['model', 'lookback', 'subwindow', 'f1_score', 'accuracy', 'n_samples']]
        .reset_index(drop=True)
    )

    print(f'\n{"=" * 80}')
    print(f'TOP {top_n} CONFIGURATIONS BY F1 SCORE')
    print(f'{"=" * 80}\n')

    print(f'{"Rank":<6} {"Model":<20} {"Lookback":<10} {"Subwindow":<11} {"F1 Score":<10} {"Accuracy":<10} {"Samples":<8}')
    print('-' * 80)

    for idx, row in top.iterrows():
        print(
            f'{idx + 1:<6} '
            f'{row["model"]:<20} '
            f'{int(row["lookback"]):>8}h '
            f'{int(row["subwindow"]):>9}h '
            f'{row["f1_score"]:>10.4f} '
            f'{row["accuracy"]:>10.4f} '
            f'{int(row["n_samples"]):>8}'
        )

    print()


def save_results(df: pd.DataFrame, target_type: str):
    """
    Save results dataframe to CSV.

    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe
    target_type : str
        Target type for filename
    """
    output_path = DATA_DIR / f'lookback_sweep_{target_type}_results.csv'
    df.to_csv(output_path, index=False)
    print(f'Results saved to: {output_path}')


def main():
    """Main execution function."""

    # ══════════════════════════════════════════════════════════════════════════
    # CONFIGURATION - Edit these parameters
    # ══════════════════════════════════════════════════════════════════════════
    TARGET_TYPE = 'sleep'  # 'suicide_risk' | 'self_harm' | 'sleep' | 'phq9'
    MAX_LOOKBACK = 30             # sweep 1 → MAX_LOOKBACK inclusive
    PROPAGATE_LABELS = False       # propagate positive labels across a user
    BALANCED_WEIGHT = False       # use balanced class weights
    USE_LOOCV = True             # leave-one-user-out CV instead of train/test split

    # Lookback periods to highlight in subwindow comparison plot
    HIGHLIGHT_LOOKBACKS = [6, 12, 18, 24, 30]
    # ══════════════════════════════════════════════════════════════════════════

    # Run the sweep
    results_df = run_sweep(
        target_type=TARGET_TYPE,
        max_lookback=MAX_LOOKBACK,
        propagate_labels=PROPAGATE_LABELS,
        balanced_weight=BALANCED_WEIGHT,
        use_loocv=USE_LOOCV,
    )

    if results_df.empty:
        print('ERROR: No results collected. Exiting.')
        return

    # Save results to CSV
    save_results(results_df, TARGET_TYPE)

    # Print top configurations
    print_top_configurations(results_df, top_n=10)

    # Generate and save visualizations
    print(f'\n{"=" * 80}')
    print('GENERATING VISUALIZATIONS')
    print(f'{"=" * 80}\n')

    # 1. F1 Score Heatmap
    print('Creating F1 score heatmap...')
    fig_f1 = plot_heatmaps(results_df, 'f1_score', TARGET_TYPE)
    f1_path = DATA_DIR / f'lookback_sweep_{TARGET_TYPE}_f1.png'
    fig_f1.savefig(f1_path, dpi=150, bbox_inches='tight')
    plt.close(fig_f1)
    print(f'  Saved: {f1_path}')

    # 2. Accuracy Heatmap
    print('Creating accuracy heatmap...')
    fig_acc = plot_heatmaps(results_df, 'accuracy', TARGET_TYPE)
    acc_path = DATA_DIR / f'lookback_sweep_{TARGET_TYPE}_accuracy.png'
    fig_acc.savefig(acc_path, dpi=150, bbox_inches='tight')
    plt.close(fig_acc)
    print(f'  Saved: {acc_path}')

    # 3. Best per lookback line graph
    print('Creating best-per-lookback line graphs...')
    fig_line = plot_best_per_lookback(results_df, TARGET_TYPE)
    line_path = DATA_DIR / f'lookback_sweep_{TARGET_TYPE}_best_per_lookback.png'
    fig_line.savefig(line_path, dpi=150, bbox_inches='tight')
    plt.close(fig_line)
    print(f'  Saved: {line_path}')

    # 4. Subwindow comparison for selected lookbacks
    available_lookbacks = [lb for lb in HIGHLIGHT_LOOKBACKS if lb in results_df['lookback'].values]
    if available_lookbacks:
        print('Creating subwindow comparison plots...')
        fig_sw = plot_subwindow_comparison(results_df, available_lookbacks, TARGET_TYPE)
        sw_path = DATA_DIR / f'lookback_sweep_{TARGET_TYPE}_subwindow_comparison.png'
        fig_sw.savefig(sw_path, dpi=150, bbox_inches='tight')
        plt.close(fig_sw)
        print(f'  Saved: {sw_path}')

    print(f'\n{"=" * 80}')
    print('ALL DONE!')
    print(f'{"=" * 80}\n')


if __name__ == '__main__':
    main()

