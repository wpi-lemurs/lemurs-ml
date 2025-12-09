"""
Utility for extracting suicide-risk labels from the database and returning a DataFrame.
This module intentionally focuses only on suicide-related questions (q2, q3, q5, q7, q12)
and computes a binary label: "at_risk" / "not_at_risk".

Usage:
from src.categorization.suicide_risk import get_suicide_risk_dataframe
df = get_suicide_risk_dataframe()
"""

import pandas as pd
from typing import Optional
from src.database_service import DatabaseService, db_config

def yesno_to_int(x):
    if x is None:
        return None
    if isinstance(x, bool):
        return 1 if x else 0
    s = str(x).strip().lower()
    if s in ("1", "yes", "y", "true", "t"):
        return 1
    if s in ("0", "no", "n", "false", "f"):
        return 0
    return None


# suicide risk label function
# Note: we use pandas.isna checks because q columns may be pandas NA/NaN after casting
def _is_valid(v) -> bool:
    return not pd.isna(v)


def suicide_risk_label(q2: Optional[int], q3: Optional[int], q5: Optional[int], q7: Optional[int],
                       q12: Optional[int]) -> str:
    if (_is_valid(q2) and int(q2) >= 2) or (_is_valid(q3) and int(q3) >= 2) or (_is_valid(q7) and int(q7) >= 2):
        return "at_risk"
    if any((v == 1) for v in [q5, q12] if _is_valid(v)):
        return "at_risk"
    return "not_at_risk"


def get_suicide_risk_dataframe() -> pd.DataFrame:
    """Connect to the DB, extract suicide-related daily answers, compute label, return DataFrame.

    Returns an empty DataFrame if the DB connection fails or no rows are found.
    Columns returned include: survey_response_id, app_user_id, timestamp, q2, q3, q5, q7, q12, and suicide_risk_label.
    """
    db = DatabaseService(**db_config)
    if not db.connect():
        # return empty df rather than raising so callers can handle gracefully
        return pd.DataFrame()
    conn = db.connection

    qids = [2, 3, 5, 7, 12]  # removed 8 and 13
    qid_list_str = ",".join(str(q) for q in qids)

    sql = f"""
    WITH daily_answers AS (
        SELECT
            sr.id AS survey_response_id,
            sr.app_user_id,
            sr.timestamp,
            a.question_id,
            a.answer
        FROM survey_response sr
        JOIN answer a
          ON sr.id = a.survey_response_id
        WHERE sr.survey_id IN (0,1)
          AND a.question_id IN ({qid_list_str})
    )
    SELECT
        survey_response_id,
        app_user_id,
        timestamp,
        MAX(CASE WHEN question_id = 2 THEN answer END)  AS q2,
        MAX(CASE WHEN question_id = 3 THEN answer END)  AS q3,
        MAX(CASE WHEN question_id = 5 THEN answer END)  AS q5,
        MAX(CASE WHEN question_id = 7 THEN answer END)  AS q7,
        MAX(CASE WHEN question_id = 12 THEN answer END) AS q12
    FROM daily_answers
    GROUP BY survey_response_id, app_user_id, timestamp
    ORDER BY app_user_id, timestamp
    ;
    """

    try:
        df = pd.read_sql(sql, conn)
    finally:
        db.disconnect()

    if df.empty:
        return df

    # Cast fields: q2, q3, q7 are numeric multiple-choice answers; convert robustly with pandas
    numeric_cols = ["q2", "q3", "q7"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    yesno_cols = ["q5", "q12"]  # removed q8 and q13
    for col in yesno_cols:
        if col in df.columns:
            df[col] = df[col].apply(yesno_to_int)

    # Use already-cast values directly; no safe_int calls
    df["suicide_risk_label"] = df.apply(
        lambda r: suicide_risk_label(
            r.get("q2"),
            r.get("q3"),
            r.get("q5"),
            r.get("q7"),
            r.get("q12"),
        ),
        axis=1,
    )

    return df


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    df = get_suicide_risk_dataframe()
    if df.empty:
        print("No rows returned or failed to connect to DB")
    else:
        print(df.head(10).to_string(index=False))
        counts = df['suicide_risk_label'].value_counts(dropna=False)
        print('\nLabel counts:')
        print(counts.to_string())
