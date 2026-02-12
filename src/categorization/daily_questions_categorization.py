import pandas as pd
from datetime import datetime, timedelta
from src.config import DATA_DIR
from src.database_service import DatabaseService, db_config

# Use centralized data directory
data_dir = DATA_DIR

# Utility converters / label functions
def safe_int(x, default=None):
    if x is None:
        return default
    try:
        return int(x)
    except (ValueError, TypeError):
        try:
            return int(float(x))
        except Exception:
            return default

def yesno_to_int(x):
    # Converts boolean/yes-no representations to 1/0/None
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

def parse_time_answer(tstr):
    """
    Accepts time strings like '11:23 pm', '23:23', '11:23pm', etc.
    Returns a datetime.time object (on today's date) or None on parse fail.
    """
    if tstr is None:
        return None
    s = str(tstr).strip()
    # Try several common formats
    fmts = ["%I:%M %p", "%I:%M%p", "%H:%M", "%H:%M:%S", "%I %p", "%I%M %p"]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            return dt
        except Exception:
            continue
    # last-ditch: try to extract digits and AM/PM
    try:
        s2 = s.replace(".", "").lower()
        if "am" in s2 or "pm" in s2:
            # insert space before am/pm for parsing
            for token in ("am","pm"):
                s2 = s2.replace(token, " " + token)
            for fmt in ["%I:%M %p", "%I %p", "%I%M %p"]:
                try:
                    dt = datetime.strptime(s2, fmt)
                    return dt
                except Exception:
                    pass
    except Exception:
        pass
    return None

# Risk label functions
def suicide_risk_label(q2, q3, q5, q7, q8, q12, q13):
    if (q2 is not None and q2 >= 2) or (q3 is not None and q3 >= 2) or (q7 is not None and q7 >= 2):
        return "at_risk"
    if any(v == 1 for v in [q5, q8, q12, q13] if v is not None):
        return "at_risk"
    return "not_at_risk"

def self_harm_risk_label(q9, q11, q15, q16, q17, q18):
    if q9 is not None and q9 >= 2:
        return "at_risk"
    if any(v == 1 for v in [q11, q15, q16, q17, q18] if v is not None):
        return "at_risk"
    return "not_at_risk"

def positive_emotion_label(total):
    if total is None:
        return None
    return "at_risk" if total <= 5 else "not_at_risk"

def negative_emotion_label(total):
    if total is None:
        return None
    return "at_risk" if total >= 14 else "not_at_risk"

def social_stress_label(total):
    if total is None:
        return None
    return "at_risk" if total >= 6 else "not_at_risk"

def social_connection_label(total):
    if total is None:
        return None
    return "at_risk" if total <= 4 else "not_at_risk"

def minority_stress_label(total):
    if total is None:
        return None
    return "at_risk" if total >= 10 else "not_at_risk"

def emotion_regulation_label(total):
    if total is None:
        return None
    return "at_risk" if total >= 9 else "not_at_risk"

def sleep_label(duration_hours, quality):
    # duration_hours may be None
    if duration_hours is None and quality is None:
        return None
    if duration_hours is not None and (duration_hours <= 5 or duration_hours >= 11):
        return "at_risk"
    if quality is not None and quality <= 2:
        return "at_risk"
    return "not_at_risk"

# Main extraction & processing

def get_daily_labels_dataframe():
    """
    Connect to DB, extract the daily questions we need, compute totals and labels,
    return a DataFrame with raw answers, computed totals, and risk labels.
    """
    db = DatabaseService(**db_config)
    if not db.connect():
        raise SystemExit("Failed to connect to database")
    conn = db.connection

    # Question IDs we need
    qids = [
        2,3,5,7,8,12,13,      # suicide
        9,11,15,16,17,18,     # self-harm & risky
        21,22,37,             # positive emotion (21,22,37)
        23,24,25,26,27,28,36, # negative emotion (23-28,36)
        31,32,33,             # social stress
        34,35,                # social connection
        40,41,42,43,44,       # minority stress
        47,48,49,50,51,52,    # emotion regulation
        54,55,56              # sleep (time in, time out, quality)
    ]

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
        WHERE sr.survey_id IN (0,1)  -- daily surveys (adjust if different)
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
        MAX(CASE WHEN question_id = 8 THEN answer END)  AS q8,
        MAX(CASE WHEN question_id = 12 THEN answer END) AS q12,
        MAX(CASE WHEN question_id = 13 THEN answer END) AS q13,

        MAX(CASE WHEN question_id = 9 THEN answer END)  AS q9,
        MAX(CASE WHEN question_id = 11 THEN answer END) AS q11,
        MAX(CASE WHEN question_id = 15 THEN answer END) AS q15,
        MAX(CASE WHEN question_id = 16 THEN answer END) AS q16,
        MAX(CASE WHEN question_id = 17 THEN answer END) AS q17,
        MAX(CASE WHEN question_id = 18 THEN answer END) AS q18,

        MAX(CASE WHEN question_id = 21 THEN answer END) AS q21,
        MAX(CASE WHEN question_id = 22 THEN answer END) AS q22,
        MAX(CASE WHEN question_id = 37 THEN answer END) AS q37,

        MAX(CASE WHEN question_id = 23 THEN answer END) AS q23,
        MAX(CASE WHEN question_id = 24 THEN answer END) AS q24,
        MAX(CASE WHEN question_id = 25 THEN answer END) AS q25,
        MAX(CASE WHEN question_id = 26 THEN answer END) AS q26,
        MAX(CASE WHEN question_id = 27 THEN answer END) AS q27,
        MAX(CASE WHEN question_id = 28 THEN answer END) AS q28,
        MAX(CASE WHEN question_id = 36 THEN answer END) AS q36,

        MAX(CASE WHEN question_id = 31 THEN answer END) AS q31,
        MAX(CASE WHEN question_id = 32 THEN answer END) AS q32,
        MAX(CASE WHEN question_id = 33 THEN answer END) AS q33,

        MAX(CASE WHEN question_id = 34 THEN answer END) AS q34,
        MAX(CASE WHEN question_id = 35 THEN answer END) AS q35,

        MAX(CASE WHEN question_id = 40 THEN answer END) AS q40,
        MAX(CASE WHEN question_id = 41 THEN answer END) AS q41,
        MAX(CASE WHEN question_id = 42 THEN answer END) AS q42,
        MAX(CASE WHEN question_id = 43 THEN answer END) AS q43,
        MAX(CASE WHEN question_id = 44 THEN answer END) AS q44,

        MAX(CASE WHEN question_id = 47 THEN answer END) AS q47,
        MAX(CASE WHEN question_id = 48 THEN answer END) AS q48,
        MAX(CASE WHEN question_id = 49 THEN answer END) AS q49,
        MAX(CASE WHEN question_id = 50 THEN answer END) AS q50,
        MAX(CASE WHEN question_id = 51 THEN answer END) AS q51,
        MAX(CASE WHEN question_id = 52 THEN answer END) AS q52,

        MAX(CASE WHEN question_id = 54 THEN answer END) AS q54,
        MAX(CASE WHEN question_id = 55 THEN answer END) AS q55,
        MAX(CASE WHEN question_id = 56 THEN answer END) AS q56

    FROM daily_answers
    GROUP BY survey_response_id, app_user_id, timestamp
    ORDER BY app_user_id, timestamp
    ;
    """

    try:
        df = pd.read_sql(sql, conn)
    finally:
        db.disconnect()

    # Clean / cast fields
    # Cast numeric 5-scale responses (0-4) to integers when present
    numeric_cols = [
        "q2","q3","q7","q9","q21","q22","q37",
        "q23","q24","q25","q26","q27","q28","q36",
        "q31","q32","q33",
        "q34","q35",
        "q40","q41","q42","q43","q44",
        "q47","q48","q49","q50","q51","q52",
        "q56"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: safe_int(x, default=None))

    # Cast yes/no fields to 0/1 ints
    yesno_cols = ["q5","q8","q11","q12","q13","q15","q16","q17","q18"]
    for col in yesno_cols:
        if col in df.columns:
            df[col] = df[col].apply(yesno_to_int)

    # Compute totals
    # Positive emotion total: q21 + q22 + q37
    df["positive_total"] = df[["q21","q22","q37"]].sum(axis=1, min_count=1).astype("Float64")

    # Negative emotion total: q23+q24+q25+q26+q27+q28 + q36
    df["negative_total"] = df[["q23","q24","q25","q26","q27","q28","q36"]].sum(axis=1, min_count=1).astype("Float64")

    # Social stress total: q31+q32+q33
    df["social_stress_total"] = df[["q31","q32","q33"]].sum(axis=1, min_count=1).astype("Float64")

    # Social connection total: q34+q35
    df["social_connection_total"] = df[["q34","q35"]].sum(axis=1, min_count=1).astype("Float64")

    # Minority stress total: q40..q44
    df["minority_total"] = df[["q40","q41","q42","q43","q44"]].sum(axis=1, min_count=1).astype("Float64")

    # Emotion regulation total: q47..q52
    df["emotion_regulation_total"] = df[["q47","q48","q49","q50","q51","q52"]].sum(axis=1, min_count=1).astype("Float64")

    # Sleep: parse q54 and q55, compute duration (hours)
    def compute_sleep_hours(row):
        in_answer = row.get("q54")
        out_answer = row.get("q55")
        if pd.isna(in_answer) or pd.isna(out_answer):
            return None
        # parse into datetimes (we only have times; attach dates)
        tin = parse_time_answer(in_answer)
        tout = parse_time_answer(out_answer)
        if tin is None or tout is None:
            return None

        # Place both on the same arbitrary date; if wake time earlier than sleep time,
        base_date = datetime(2000,1,1)
        dt_in = base_date.replace(hour=tin.hour, minute=tin.minute, second=tin.second if hasattr(tin, "second") else 0)
        dt_out = base_date.replace(hour=tout.hour, minute=tout.minute, second=tout.second if hasattr(tout, "second") else 0)
        if dt_out <= dt_in:
            dt_out += timedelta(days=1)
        duration = dt_out - dt_in
        hours = duration.total_seconds() / 3600.0
        return round(hours, 3)

    df["sleep_hours"] = df.apply(compute_sleep_hours, axis=1)

    # Sleep quality (q56)
    if "q56" in df.columns:
        df["sleep_quality"] = df["q56"].apply(lambda x: safe_int(x, default=None))

    # Compute binary labels
    df["suicide_risk_label"] = df.apply(
        lambda r: suicide_risk_label(
            safe_int(r.get("q2")), safe_int(r.get("q3")),
            yesno_to_int(r.get("q5")), safe_int(r.get("q7")),
            yesno_to_int(r.get("q8")), yesno_to_int(r.get("q12")),
            yesno_to_int(r.get("q13"))
        ), axis=1
    )

    df["self_harm_risk_label"] = df.apply(
        lambda r: self_harm_risk_label(
            safe_int(r.get("q9")),
            yesno_to_int(r.get("q11")),
            yesno_to_int(r.get("q15")),
            yesno_to_int(r.get("q16")),
            yesno_to_int(r.get("q17")),
            yesno_to_int(r.get("q18"))
        ),
        axis=1
    )

    df["positive_emotion_label"] = df["positive_total"].apply(
        lambda t: positive_emotion_label(t) if not pd.isna(t) else None
    )

    df["negative_emotion_label"] = df["negative_total"].apply(
        lambda t: negative_emotion_label(t) if not pd.isna(t) else None
    )

    df["social_stress_label"] = df["social_stress_total"].apply(
        lambda t: social_stress_label(t) if not pd.isna(t) else None
    )

    df["social_connection_label"] = df["social_connection_total"].apply(
        lambda t: social_connection_label(t) if not pd.isna(t) else None
    )

    df["minority_stress_label"] = df["minority_total"].apply(
        lambda t: minority_stress_label(t) if not pd.isna(t) else None
    )

    df["emotion_regulation_label"] = df["emotion_regulation_total"].apply(
        lambda t: emotion_regulation_label(t) if not pd.isna(t) else None
    )

    CUTOFF_HOUR = 14  # morning survey closes at 13, afternoon opens at 15; we only need a sleep label for morning, as no sleep data is recorded in the afternoon
    df["sleep_label"] = df.apply(
        lambda r: (
            'N/A' if r.get("timestamp").hour >= CUTOFF_HOUR
            else sleep_label(r.get("sleep_hours"), r.get("sleep_quality"))
        ),
        axis=1
    )

    return df

# Script entrypoint

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    df = get_daily_labels_dataframe()
    # Save to CSV for pipeline / auditing in data directory
    out_csv = data_dir / "daily_labels.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} — sample rows:")
    print(df.head(10).to_string(index=False))