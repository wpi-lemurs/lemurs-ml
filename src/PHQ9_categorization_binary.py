import pandas as pd
from database_service import DatabaseService, db_config

def get_phq9_dataframe():
    """
    Connects to the DB, extracts PHQ-9 answers, computes total score and binary label,
    and returns a Pandas DataFrame.
    """
# Create DB service instance and connect
    db = DatabaseService(**db_config)
    if not db.connect():
        raise SystemExit("Failed to connect to database")
    conn = db.connection

    # SQL to extract PHQ-9 answers
    sql = """
    WITH phq9_answers AS (
    SELECT 
        sr.id AS survey_response_id,
        sr.app_user_id,
        sr.timestamp,
        a.question_id,
        CAST(a.answer AS INTEGER) AS answer
    FROM survey_response sr
    JOIN answer a
        ON sr.id = a.survey_response_id
    WHERE sr.survey_id = 2
      AND a.question_id BETWEEN 59 AND 67
)
SELECT
    survey_response_id,
    app_user_id,
    timestamp,
    MAX(CASE WHEN question_id = 59 THEN answer END) AS q1,
    MAX(CASE WHEN question_id = 60 THEN answer END) AS q2,
    MAX(CASE WHEN question_id = 61 THEN answer END) AS q3,
    MAX(CASE WHEN question_id = 62 THEN answer END) AS q4,
    MAX(CASE WHEN question_id = 63 THEN answer END) AS q5,
    MAX(CASE WHEN question_id = 64 THEN answer END) AS q6,
    MAX(CASE WHEN question_id = 65 THEN answer END) AS q7,
    MAX(CASE WHEN question_id = 66 THEN answer END) AS q8,
    MAX(CASE WHEN question_id = 67 THEN answer END) AS q9
FROM phq9_answers
GROUP BY survey_response_id, app_user_id, timestamp
ORDER BY survey_response_id;
    """

    try:
        df = pd.read_sql(sql, conn)
    finally:
        db.disconnect()

    df["phq9_total_score"] = df[["q1","q2","q3","q4","q5","q6","q7","q8","q9"]].sum(axis=1)

    def categorize_phq9(score):
        return "not depressed" if score < 10 else "depressed"

    df["severity_label"] = df["phq9_total_score"].apply(categorize_phq9)

    return df

if __name__ == "__main__":
    df = get_phq9_dataframe()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(df.head())
