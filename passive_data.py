import pandas as pd
from database_service import DatabaseService

# Create db service instance
service = DatabaseService()

# Import passive df
screentime_app = service.extract_from_database("screentime_app")
screentime_app.rename(columns={'id':'screentime_app_id'}, inplace=True)

screentime_general = service.extract_from_database("screentime")
screentime_general.rename(columns={'id':'screentime_id'}, inplace=True)

# Disconnect from database when done
service.disconnect()

# Connect screentime_app and screentime_all on screentime_id
screentime_merged = screentime_app.merge(screentime_general, on='screentime_id')

# Import and append associated PHQ-9 scores
# question = pd.read_csv('lemurs-ml/data/question_202509261602.csv')
# Select only PHQ-9 questions
# phq9 = question[question['Survey' == 'PHQ9']] # Need to make Survey column first
# ACTUALLY I think we can just join it with Survey_question. Can check when DB is back up

print(screentime_merged.columns)
print(screentime_merged.head())
# print(phq9.columns)

# Feature engineering

#
