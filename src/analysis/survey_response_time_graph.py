from src.database_service import DatabaseService
import pandas as pd
import logging
from pathlib import Path

import matplotlib.pyplot as plt

MORNING_START_HOUR = 5  # inclusive
MORNING_END_HOUR = 11   # inclusive


def _select_timestamp_column(df: pd.DataFrame) -> pd.Series:
    """Pick the first usable timestamp column and return it as datetime."""
    for col in ("response_timestamp", "submitted_at", "created_at", "timestamp"):
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().any():
                return parsed
    raise ValueError("No usable timestamp column found in survey_response data.")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "data" / "morning_survey_submissions_by_hour.png"

    db = DatabaseService()
    db.connect()
    survey_df = db.extract_from_database("survey_response")
    db.disconnect()

    if survey_df is None or survey_df.empty:
        logging.warning("No survey_response records found; nothing to plot.")
        return

    survey_df = survey_df.copy()
    survey_df["submission_time"] = _select_timestamp_column(survey_df)
    survey_df = survey_df.dropna(subset=["submission_time"])
    if survey_df.empty:
        logging.warning("All survey_response timestamps are null; nothing to plot.")
        return

    survey_df["hour"] = survey_df["submission_time"].dt.hour
    morning_df = survey_df[survey_df["hour"].between(MORNING_START_HOUR, MORNING_END_HOUR)]
    if morning_df.empty:
        logging.warning("No morning survey submissions found between %s and %s hours.", MORNING_START_HOUR, MORNING_END_HOUR)
        return

    counts = morning_df["hour"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", ax=ax, color="#4a90e2")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Number of submissions")
    ax.set_title(f"Morning survey submissions by hour ({MORNING_START_HOUR:02d}:00–{MORNING_END_HOUR:02d}:59)")
    ax.set_xticklabels([f"{int(h):02d}:00" for h in counts.index], rotation=0)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    logging.info("Saved morning submissions bar chart to %s", output_path)


if __name__ == "__main__":
    main()
