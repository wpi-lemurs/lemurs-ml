import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from google_play_scraper import app
from src.database_service import DatabaseService
import pandas as pd

'''
This script categorizes apps from screentime data using Google Play Store categories. It currently keeps the categories
specific, but for ML modeling purposes, it could be useful, and is recommended, to further analyze apps/categories
and pull from the csv to combine certain categories (e.g., all game categories into a single 'Games' category or 
photography + video players into 'Media' category).
'''

# extract screentime app data
service = DatabaseService()
screentime_app_data = service.extract_from_database("screentime_app")

def categorize_apps(screentime_app_df):
    '''
    Input a dataframe of screentime app data and create a new column for app categories
    Return the dataframe with the new column
    '''
    for app_name in screentime_app_df['app_name'].unique().tolist():
        try:
            category = app(app_name)['genreId']
        except Exception as e:
            # fallback to system app categorization
            category = categorize_system_app(app_name)
        # update category for all rows with this app_name
        screentime_app_df.loc[screentime_app_df['app_name'] == app_name, 'app_category'] = category
    
    return screentime_app_df

def categorize_system_app(app_name):
    # categorize system apps using common app_name keywords
    p = app_name.lower()

    # Lemurs app
    if 'lemurs' in p:
        return 'HEALTH_AND_FITNESS'
    
    if any(k in p for k in ['dialer', 'contacts', 'messaging', 'incallui', 'telecom', 'phone']):
        return 'COMMUNICATION'

    if any(k in p for k in ['camera', 'gallery', 'photo', 'screenshot', 'photopicker', 'markup']):
        return 'PHOTOGRAPHY'

    if any(k in p for k in ['videoplayer', 'video', 'smartmirroring']):
        return 'VIDEO_PLAYERS'
    
    if any(k in p for k in ['soundrecorder', 'dolby', 'music']):
        return 'MUSIC_AND_AUDIO'

    if "weather" in p:
        return 'WEATHER'

    if any(k in p for k in ['compass', 'navigation', 'location']):
        return 'MAPS_AND_NAVIGATION'

    if any(k in p for k in ['calculator', 'clock', 'deskclock', 'notes', 'reminder', 'calendar']):
        return 'PRODUCTIVITY'

    if any(k in p for k in ['health', 'fitness', 'wellbeing']):
        return 'HEALTH_AND_FITNESS'

    if any(k in p for k in ['wallet', 'mwallet', 'finance']):
        return 'FINANCE'

    if any(k in p for k in ['launcher', 'theme', 'wallpaper', 'aod', 'dressroom']):
        return 'PERSONALIZATION'

    if p.startswith(('com.android', 'com.google.android', 'com.samsung', 'com.miui',
                     'com.oplus', 'com.vivo', 'com.motorola', 'com.xiaomi')):
        return 'TOOLS'

    print(f"Could not categorize system app: {app_name}. Assigning 'UNKNOWN' category.")
    return 'UNKNOWN'

categorized_df = categorize_apps(screentime_app_data)

# save categorized data to data folder
output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'screentime_app_categorized.csv')
categorized_df.to_csv(output_path, index=False)