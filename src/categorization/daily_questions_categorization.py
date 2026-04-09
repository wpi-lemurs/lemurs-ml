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

def averaged_yesno_to_int(x):
    """Map averaged yes/no values to 0/1. Any mean >= 0.5 is treated as yes."""
    if x is None or pd.isna(x):
        return None
    return 1 if float(x) >= 0.5 else 0

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
	"""
	Suicide Risk Score (Questions 2, 3, 5, 7, 8, 12, 13)

	Questions:
	  (2) 5-scale: I wanted to die
	  (3) 5-scale: I thought about taking my life
	  (5) yes-no: Considered a specific suicide method or made preparations
	  (7) 5-scale: How intense was your desire to kill yourself?
	  (8) yes-no: Do you intend to kill yourself right now?
	  (12) yes-no: Attempted suicide
	  (13) yes-no: Were you severely injured or required medical intervention?

	AT RISK if:
	  - Any ideation item (q2, q3, q7) ≥ 2
	  - Any yes-no question (q5, q8, q12, q13) is answered YES (1)

	Otherwise: NOT AT RISK
	"""
	if (q2 is not None and q2 >= 2) or (q3 is not None and q3 >= 2) or (q7 is not None and q7 >= 2):
		return "at_risk"
	if any(v == 1 for v in [q5, q8, q12, q13] if v is not None):
		return "at_risk"
	return "not_at_risk"

def self_harm_risk_label(q9, q11, q15, q16, q17, q18):
	"""
	Self-Harm and Risky Behavior Score (Questions 9, 11, 15, 16, 17, 18)
	Scale: 0-9

	Questions:
	  (9) 5-scale: How strong was your urge to injure yourself without intent to die?
	  (11) yes-no: Injured yourself without intent to die
	  (15) yes-no: Eaten unusually large amount of food with loss of control
	  (16) yes-no: Made yourself sick (vomit) or taken laxatives
	  (17) yes-no: Gotten drunk
	  (18) yes-no: Taken drugs or medication not as prescribed

	AT RISK if:
	  - Question 9 ≥ 2
	  - Any yes-no question (q11, q15, q16, q17, q18) is answered YES (1)

	Otherwise: NOT AT RISK
	"""
	if q9 is not None and q9 >= 2:
		return "at_risk"
	if any(v == 1 for v in [q11, q15, q16, q17, q18] if v is not None):
		return "at_risk"
	return "not_at_risk"

def positive_emotion_label(total):
	"""
	Positive Emotion Score (Questions 21, 22, 37)
	Scale: 0-12 (sum of three 5-scale items)

	Questions:
	  (21) 5-scale: Felt content
	  (22) 5-scale: Felt cheerful
	  (37) 5-scale: Performed well or succeeded at something

	AT RISK if:
	  - Sum of total score ≤ 5

	Otherwise: NOT AT RISK
	"""
	if total is None:
		return None
	return "at_risk" if total <= 5 else "not_at_risk"

def negative_emotion_label(total):
	"""
	Negative Emotion Score (Questions 23, 24, 25, 26, 27, 28, 36)
	Scale: 0-28 (sum of seven 5-scale items)

	Questions:
	  (23) 5-scale: Felt nervous
	  (24) 5-scale: Felt sad
	  (25) 5-scale: Felt useless
	  (26) 5-scale: Felt lonely
	  (27) 5-scale: Felt like giving up because nothing can be done
	  (28) 5-scale: Felt people don't understand my experiences
	  (36) 5-scale: Failed or performed poorly at something

	AT RISK if:
	  - Sum of total score ≥ 14

	Otherwise: NOT AT RISK
	"""
	if total is None:
		return None
	return "at_risk" if total >= 14 else "not_at_risk"

def social_stress_label(total):
	"""
	Social Stress Score (Questions 31, 32, 33)
	Scale: 0-12 (sum of three 5-scale items)

	Questions:
	  (31) 5-scale: Gotten into argument/disagreement with friend, significant other, or family
	  (32) 5-scale: Felt insulted or criticized
	  (33) 5-scale: Felt rejected, abandoned, excluded, or left out

	AT RISK if:
	  - Sum of total score ≥ 6

	Otherwise: NOT AT RISK
	"""
	if total is None:
		return None
	return "at_risk" if total >= 6 else "not_at_risk"

def social_connection_label(total):
	"""
	Social Connection Score (Questions 34, 35)
	Scale: 0-8 (sum of two 5-scale items)

	Questions:
	  (34) 5-scale: Felt admired or complimented
	  (35) 5-scale: Felt wanted or included

	AT RISK if:
	  - Sum of total score ≤ 4

	Otherwise: NOT AT RISK
	"""
	if total is None:
		return None
	return "at_risk" if total <= 4 else "not_at_risk"

def minority_stress_label(total):
	"""
	Minority Stress Score (Questions 40, 41, 42, 43, 44)
	Scale: 0-20 (sum of five 5-scale items)

	Questions:
	  (40) 5-scale: Avoided subjects of sex/love/attraction to conceal sexual orientation
	  (41) 5-scale: Felt alienated from self because of LGBT identity
	  (42) 5-scale: Worried people will reject me because I'm LGBT
	  (43) 5-scale: Been treated unfairly because I am LGBT
	  (44) 5-scale: Exposed to anti-LGBT media/social media content

	AT RISK if:
	  - Sum of total score ≥ 10

	Otherwise: NOT AT RISK
	"""
	if total is None:
		return None
	return "at_risk" if total >= 10 else "not_at_risk"

def emotion_regulation_label(total):
	"""
	Emotion Regulation Score (Questions 47, 48, 49, 50, 51, 52)
	Scale: 0-18 (sum of six 4-scale items; note: these are 4-scale, not 5-scale)

	When I felt negative emotions:
	  (47) 4-scale: Had difficulty focusing on other things
	  (48) 4-scale: Thought I should not feel that way
	  (49) 4-scale: Had difficulty controlling my behaviors
	  (50) 4-scale: Didn't pay attention to how I felt
	  (51) 4-scale: Thought they would last for a long time
	  (52) 4-scale: Had difficulty making sense out of them

	AT RISK if:
	  - Sum of total score ≥ 9

	Otherwise: NOT AT RISK
	"""
	if total is None:
		return None
	return "at_risk" if total >= 9 else "not_at_risk"

def sleep_label(duration_hours, quality):
	"""
	Sleep Score (Questions 54, 55, 56)
	Scale: Duration in hours + Quality (5-scale)

	Questions:
	  (54) Time-picker: What time did you fall asleep last night? (HH:MM am/pm)
	  (55) Time-picker: What time did you wake up this morning? (HH:MM am/pm)
	  (56) 5-scale: How would you rate the quality of your sleep?

	Duration is calculated as: wake_time - sleep_time

	AT RISK if:
	  - Sleep duration ≤ 5 hours
	  - Sleep duration ≥ 11 hours
	  - Quality of sleep score ≤ 2

	Otherwise: NOT AT RISK
	"""
	# duration_hours may be None
	if duration_hours is None and quality is None:
		return None
	if duration_hours is not None and (duration_hours <= 5 or duration_hours >= 11):
		return "at_risk"
	if quality is not None and quality <= 2:
		return "at_risk"
	return "not_at_risk"

# Main extraction & processing

def get_daily_labels_dataframe(average_shared_labels=False):
    """
    Connect to DB, extract the daily questions we need, compute totals and labels,
    return a DataFrame with raw answers, computed totals, and risk labels.

    Parameters:
      average_shared_labels: if True, collapse morning+afternoon surveys into one
      row per user/day by averaging non-sleep items before label computation.
      If False (default), keep per-survey rows with no averaging.
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
            sr.survey_id,
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
        MAX(survey_id) AS survey_id,
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

    # Clean / cast fields at per-survey level
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

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df["survey_date"] = df["timestamp"].dt.date

    if average_shared_labels:
        # Create one row per user/day for labels present in both surveys by averaging scores.
        shared_scale_cols = [
            "q2","q3","q7","q9","q21","q22","q37",
            "q23","q24","q25","q26","q27","q28","q36",
            "q31","q32","q33",
            "q34","q35",
            "q40","q41","q42","q43","q44",
            "q47","q48","q49","q50","q51","q52",
        ]
        shared_yesno_cols = ["q5","q8","q11","q12","q13","q15","q16","q17","q18"]

        sort_cols = ["app_user_id", "survey_date", "timestamp"]
        grouped = df.sort_values(sort_cols).groupby(["app_user_id", "survey_date"], as_index=False)

        daily_shared = grouped.agg(
            survey_response_id=("survey_response_id", "first"),
            timestamp=("timestamp", "min"),
            **{col: (col, "mean") for col in (shared_scale_cols + shared_yesno_cols)}
        )

        # Sleep comes from morning survey only (survey_id=0), then merged into the single daily row.
        morning_sleep = (
            df[df["survey_id"] == 0]
            .sort_values(sort_cols)
            .drop_duplicates(["app_user_id", "survey_date"], keep="first")
            [["app_user_id", "survey_date", "q54", "q55", "q56"]]
        )

        df = daily_shared.merge(morning_sleep, on=["app_user_id", "survey_date"], how="left")

        for col in shared_yesno_cols:
            if col in df.columns:
                df[col] = df[col].apply(averaged_yesno_to_int)

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

    # Compute binary labels from the daily-averaged score profile.
    df["suicide_risk_label"] = df.apply(
        lambda r: suicide_risk_label(
            r.get("q2"), r.get("q3"),
            r.get("q5"), r.get("q7"),
            r.get("q8"), r.get("q12"),
            r.get("q13")
        ), axis=1
    )

    df["self_harm_risk_label"] = df.apply(
        lambda r: self_harm_risk_label(
            r.get("q9"),
            r.get("q11"),
            r.get("q15"),
            r.get("q16"),
            r.get("q17"),
            r.get("q18")
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

    if average_shared_labels:
        df["sleep_label"] = df.apply(
            lambda r: sleep_label(r.get("sleep_hours"), r.get("sleep_quality")),
            axis=1
        )
    else:
        # Sleep items are only valid for morning survey rows.
        df["sleep_label"] = df.apply(
            lambda r: (
                "N/A" if r.get("survey_id") != 0
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