import pandas as pd
from database_service import DatabaseService, db_config

# Create DB service instance and connect
db = DatabaseService(**db_config)
if not db.connect():
    raise SystemExit("Failed to connect to database")
conn = db.connection

# SQL to extract PHQ-9 answers
sql = """
WITH daily_answers AS (
    SELECT 
        sr.id AS survey_response_id,
        sr.app_user_id,        
        sr.timestamp,
        a.question_id,
        CAST(a.answer AS INTEGER) AS answer
    FROM survey_response sr
    JOIN answer a
        ON sr.id = a.survey_response_id
    WHERE sr.survey_id = 0 OR sr.survey_id = 1
      AND a.question_id BETWEEN 0 AND 56
)
SELECT
    survey_response_id,
    app_user_id,
    timestamp,
    -- TODO: UPDATE SELECTIONS FOR DESIRED QUESTIONS
    MAX(CASE WHEN question_id = 21 THEN answer END) AS content,
    MAX(CASE WHEN question_id = 22 THEN answer END) AS cheerful,
    MAX(CASE WHEN question_id = 23 THEN answer END) AS nervous,
    MAX(CASE WHEN question_id = 24 THEN answer END) AS sad,
    MAX(CASE WHEN question_id = 25 THEN answer END) AS useless,
    MAX(CASE WHEN question_id = 26 THEN answer END) AS lonely,
    MAX(CASE WHEN question_id = 27 THEN answer END) AS give_up,
    MAX(CASE WHEN question_id = 28 THEN answer END) AS misunderstood
FROM daily_answers
GROUP BY survey_response_id, app_user_id, timestamp
ORDER BY app_user_id, timestamp;
"""

# Load results into a Pandas DataFrame using the DatabaseService connection
df = pd.read_sql(sql, conn)

# Close the DB connection after extraction
db.disconnect()

# Save to CSV for ML pipeline
df.to_csv("daily_questions.csv", index=False)

print("Daily questions logged. Sample Data:")
print(df.columns)
