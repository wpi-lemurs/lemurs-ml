import sys
import os
import json
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

# All Google Play Store GAME_* sub-genre IDs that should be collapsed into a single GAMES category
GAME_SUBCATEGORIES = {
    'GAME_ACTION', 'GAME_ADVENTURE', 'GAME_ARCADE', 'GAME_BOARD', 'GAME_CARD',
    'GAME_CASINO', 'GAME_CASUAL', 'GAME_EDUCATIONAL', 'GAME_MUSIC', 'GAME_PUZZLE',
    'GAME_RACING', 'GAME_ROLE_PLAYING', 'GAME_SIMULATION', 'GAME_SPORTS',
    'GAME_STRATEGY', 'GAME_TRIVIA', 'GAME_WORD',
}


CACHE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'app_category_cache.json')


def normalize_category(category):
    """
    Normalize category names, collapsing all GAME_* subcategories into 'GAMES'.

    Parameters:
    -----------
    category : str
        Raw category string (e.g. from Google Play Store genreId)

    Returns:
    --------
    str : Normalized category
    """
    if category in GAME_SUBCATEGORIES or category.startswith('GAME_'):
        return 'GAMES'
    return category


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

    # MANUAL OVERRIDES for specific apps that need recategorization
    manual_categories = {
        'com.ss.android.ugc.trill': 'SOCIAL',       # TikTok
        'com.mobile.legends': 'GAMES',               # Mobile Legends
        'app.revanced.android.youtube': 'VIDEO_PLAYERS',  # YouTube ReVanced
        'com.rubenmayayo.reddit': 'SOCIAL',          # Reddit client
    }

    if app_name in manual_categories:
        return manual_categories[app_name]

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


def _load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}


def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, 'w', encoding='utf-8') as fh:
        json.dump(cache, fh)


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

    unique_apps = df['app_name'].unique().tolist()
    total_apps = len(unique_apps)

    cache = _load_cache()
    category_map = {}

    print(f"Categorizing {total_apps} unique apps (cache hit rate will speed this up)...")

    for idx, app_name in enumerate(unique_apps, 1):
        if app_name in cache:
            category_map[app_name] = cache[app_name]
            continue

        if idx % 50 == 0 or idx == total_apps:
            print(f"  Progress: {idx}/{total_apps} apps ({idx/total_apps*100:.1f}%)")

        category = None

        if play_store_app is not None:
            try:
                category = play_store_app(app_name)['genreId']
            except Exception:
                category = None

        if category is None:
            category = categorize_system_app(app_name)

        category = normalize_category(category)
        cache[app_name] = category
        category_map[app_name] = category

    _save_cache(cache)

    df['app_category'] = df['app_name'].map(category_map)
    print(f"[OK] Completed categorizing all {total_apps} apps! (cached: {len(cache)} entries)")
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
    print("Connecting to database to extract screentime app data...")
    service = DatabaseService()
    screentime_app_data = service.extract_from_database("screentime_app")
    service.disconnect()
    print(f"[OK] Extracted {len(screentime_app_data):,} screentime app records\n")

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
