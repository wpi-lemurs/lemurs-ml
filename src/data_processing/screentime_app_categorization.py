import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
from src.database_service import DatabaseService

try:
    from google_play_scraper import app as play_store_app
except ImportError:
    play_store_app = None
    print("Warning: google_play_scraper not installed. Will use system categorization only.")

'''
This script categorizes apps from screentime data using Google Play Store categories. It currently keeps the categories
specific, but for ML modeling purposes, it could be useful, and is recommended, to further analyze apps/categories
and pull from the csv to combine certain categories (e.g., all game categories into a single 'Games' category or 
photography + video players into 'Media' category).
'''


def categorize_system_app(app_name):
    """
    Categorize system apps using common app_name keywords.

    Parameters:
    -----------
    app_name : str
        The app name/package to categorize

    Returns:
    --------
    str : App category
    """
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

    # print(f"Could not categorize system app: {app_name}. Assigning 'UNKNOWN' category.")
    return 'UNKNOWN'


def categorize_apps(screentime_app_df):
    """
    Input a dataframe of screentime app data and create a new column for app categories.

    Parameters:
    -----------
    screentime_app_df : DataFrame
        DataFrame with screentime app data containing 'app_name' column

    Returns:
    --------
    DataFrame : The input DataFrame with 'app_category' column added
    """
    # Make a copy to avoid modifying the original
    df = screentime_app_df.copy()

    for app_name in df['app_name'].unique().tolist():
        category = None

        # Try Google Play Store first if available
        if play_store_app is not None:
            try:
                category = play_store_app(app_name)['genreId']
            except Exception as e:
                pass  # Fall through to system categorization

        # Fallback to system app categorization
        if category is None:
            category = categorize_system_app(app_name)

        # update category for all rows with this app_name
        df.loc[df['app_name'] == app_name, 'app_category'] = category

    return df


def get_categorized_screentime_data():
    """
    Extract screentime app data from database and categorize it.

    This function replaces the need for a CSV file by directly extracting
    and categorizing data from the database.

    Returns:
    --------
    DataFrame : Categorized screentime app data
    """
    service = DatabaseService()
    screentime_app_data = service.extract_from_database("screentime_app")
    service.disconnect()

    categorized_df = categorize_apps(screentime_app_data)
    return categorized_df


# When run as a script, save to CSV for offline use (optional)
if __name__ == "__main__":
    categorized_df = get_categorized_screentime_data()

    # save categorized data to data folder
    output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'screentime_app_categorized.csv')
    categorized_df.to_csv(output_path, index=False)
    print(f"Saved categorized data to {output_path}")
    print(f"Total records: {len(categorized_df):,}")
