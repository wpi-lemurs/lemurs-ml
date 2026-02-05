import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.database_service import DatabaseService

'''
This script performs comprehensive analysis on screentime app categories to understand:
- Category breakdown and distribution
- Usage patterns across categories
- Temporal patterns (time of day, day of week)
- Individual user behavior patterns
- Category dominance and diversity
'''

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Load the categorized screentime data
data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'screentime_app_categorized.csv')
df = pd.read_csv(data_path)

# Connect to database to get additional info
service = DatabaseService()

# Get app_user_id from screentime table
screentime_data = service.extract_from_database("screentime")
screentime_mapping = screentime_data[['id', 'app_user_id']].rename(columns={'id': 'screentime_id'})
df = df.merge(screentime_mapping, on='screentime_id', how='left')

# Use app_user_id as the user identifier
df['user_id'] = df['app_user_id']

print("="*80)
print("DATA CLEANING - Removing duplicate/cumulative records")
print("="*80)
print(f"Records before deduplication: {len(df):,}")

# DEDUPLICATION: Keep only the most recent record for each app per screentime_id
# Since total_time_ms is cumulative, we want the latest record which has the final count
df['last_time_used'] = pd.to_datetime(df['last_time_used'])
df = df.sort_values('last_time_used', ascending=True)
df = df.drop_duplicates(subset=['screentime_id', 'app_name'], keep='last')

print(f"Records after deduplication: {len(df):,}")
print(f"Records removed: {len(pd.read_csv(data_path)) - len(df):,}")
print("="*80)

# Convert time columns
df['last_time_used'] = pd.to_datetime(df['last_time_used'])
df['total_time_hours'] = df['total_time_ms'] / (1000 * 60 * 60)  # Convert to hours
df['total_time_minutes'] = df['total_time_ms'] / (1000 * 60)  # Convert to minutes

# Extract temporal features
df['hour'] = df['last_time_used'].dt.hour
df['day_of_week'] = df['last_time_used'].dt.dayofweek
df['date'] = df['last_time_used'].dt.date
df['day_name'] = df['last_time_used'].dt.day_name()

# Create output directory for analysis
output_dir = os.path.join(os.path.dirname(__file__), 'analysis_outputs', 'screentime_categories')
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("SCREENTIME APP CATEGORY ANALYSIS")
print("="*80)
print(f"\nTotal records: {len(df):,}")
print(f"Total unique users: {df['user_id'].nunique()}")
print(f"Total unique apps: {df['app_name'].nunique()}")
print(f"Total unique categories: {df['app_category'].nunique()}")
print(f"Date range: {df['last_time_used'].min()} to {df['last_time_used'].max()}")

# ============================================================================
# 1. OVERALL CATEGORY BREAKDOWN
# ============================================================================
print("\n" + "="*80)
print("1. CATEGORY BREAKDOWN - What apps are people using?")
print("="*80)

category_stats = df.groupby('app_category').agg({
    'total_time_hours': 'sum',
    'id': 'count',
    'user_id': 'nunique',
    'app_name': 'nunique'
}).round(2)
category_stats.columns = ['Total Hours', 'Record Count', 'Unique Users', 'Unique Apps']
category_stats = category_stats.sort_values('Total Hours', ascending=False)
category_stats['% of Total Time'] = (category_stats['Total Hours'] / category_stats['Total Hours'].sum() * 100).round(2)
category_stats['Avg Hours per User'] = (category_stats['Total Hours'] / category_stats['Unique Users']).round(2)

print("\nCategory Statistics:")
print(category_stats)

# Save to CSV
category_stats.to_csv(os.path.join(output_dir, 'category_statistics.csv'))

# Visualization 1: Category distribution by time spent
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Top categories by total time
top_categories = category_stats.head(15)
axes[0, 0].barh(top_categories.index, top_categories['Total Hours'], color='steelblue')
axes[0, 0].set_xlabel('Total Hours', fontsize=12)
axes[0, 0].set_title('Top 15 Categories by Total Time Spent', fontsize=14, fontweight='bold')
axes[0, 0].invert_yaxis()
for i, v in enumerate(top_categories['Total Hours']):
    axes[0, 0].text(v, i, f' {v:,.0f}h', va='center', fontsize=9)

# Pie chart of time distribution
top_10_for_pie = category_stats.head(10)
other_time = category_stats.iloc[10:]['Total Hours'].sum()
pie_data = pd.concat([top_10_for_pie['Total Hours'], pd.Series([other_time], index=['Other'])])
colors = sns.color_palette('husl', len(pie_data))
axes[0, 1].pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90, colors=colors)
axes[0, 1].set_title('Distribution of Total Time (Top 10 + Other)', fontsize=14, fontweight='bold')

# Categories by unique users
axes[1, 0].barh(top_categories.index, top_categories['Unique Users'], color='coral')
axes[1, 0].set_xlabel('Number of Unique Users', fontsize=12)
axes[1, 0].set_title('Top 15 Categories by User Adoption', fontsize=14, fontweight='bold')
axes[1, 0].invert_yaxis()
for i, v in enumerate(top_categories['Unique Users']):
    axes[1, 0].text(v, i, f' {v}', va='center', fontsize=9)

# Average hours per user per category
avg_per_user = category_stats.sort_values('Avg Hours per User', ascending=False).head(15)
axes[1, 1].barh(avg_per_user.index, avg_per_user['Avg Hours per User'], color='mediumseagreen')
axes[1, 1].set_xlabel('Average Hours per User', fontsize=12)
axes[1, 1].set_title('Top 15 Categories by Avg Hours per User', fontsize=14, fontweight='bold')
axes[1, 1].invert_yaxis()
for i, v in enumerate(avg_per_user['Avg Hours per User']):
    axes[1, 1].text(v, i, f' {v:.1f}h', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_category_breakdown.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"\n✓ Saved: 01_category_breakdown.png")

# ============================================================================
# 2. TEMPORAL PATTERNS - When are different categories used?
# ============================================================================
print("\n" + "="*80)
print("2. TEMPORAL PATTERNS - When are categories used?")
print("="*80)

# Get top 10 categories for temporal analysis
top_10_cats = category_stats.head(10).index.tolist()

# Hourly usage patterns by category
hourly_by_category = df[df['app_category'].isin(top_10_cats)].groupby(['hour', 'app_category'])['total_time_hours'].sum().reset_index()
hourly_pivot = hourly_by_category.pivot(index='hour', columns='app_category', values='total_time_hours').fillna(0)

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Heatmap of category usage by hour
sns.heatmap(hourly_pivot.T, cmap='YlOrRd', annot=False, fmt='.0f', cbar_kws={'label': 'Total Hours'}, ax=axes[0])
axes[0].set_xlabel('Hour of Day', fontsize=12)
axes[0].set_ylabel('App Category', fontsize=12)
axes[0].set_title('App Category Usage by Hour of Day (Top 10 Categories)', fontsize=14, fontweight='bold')

# Line plot showing temporal patterns
for category in top_10_cats[:7]:  # Show top 7 to avoid clutter
    cat_hourly = hourly_pivot[category]
    axes[1].plot(cat_hourly.index, cat_hourly.values, marker='o', label=category, linewidth=2)

axes[1].set_xlabel('Hour of Day', fontsize=12)
axes[1].set_ylabel('Total Hours', fontsize=12)
axes[1].set_title('Usage Patterns Throughout the Day (Top 7 Categories)', fontsize=14, fontweight='bold')
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(range(0, 24))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_temporal_patterns.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: 02_temporal_patterns.png")

# Day of week patterns
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_by_category = df[df['app_category'].isin(top_10_cats)].groupby(['day_name', 'app_category'])['total_time_hours'].sum().reset_index()
daily_pivot = daily_by_category.pivot(index='day_name', columns='app_category', values='total_time_hours').fillna(0)
daily_pivot = daily_pivot.reindex(day_order)

fig, ax = plt.subplots(figsize=(14, 8))
daily_pivot.plot(kind='bar', ax=ax, width=0.8)
ax.set_xlabel('Day of Week', fontsize=12)
ax.set_ylabel('Total Hours', fontsize=12)
ax.set_title('App Category Usage by Day of Week (Top 10 Categories)', fontsize=14, fontweight='bold')
ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '03_day_of_week_patterns.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: 03_day_of_week_patterns.png")

# ============================================================================
# 3a. TOP APPS WITHIN TOP CATEGORIES
# ============================================================================
print("\n" + "="*80)
print("3a. TOP APPS IN TOP CATEGORIES - Which specific apps dominate?")
print("="*80)

# Get top 6 categories
top_6_categories = category_stats.head(6).index.tolist()

# For each top category, get top 5 apps
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for idx, category in enumerate(top_6_categories):
    category_df = df[df['app_category'] == category]

    # Aggregate by app name
    app_stats = category_df.groupby('app_name').agg({
        'total_time_hours': 'sum',
        'user_id': 'nunique'
    }).round(2)
    app_stats.columns = ['Total Hours', 'Unique Users']
    app_stats = app_stats.sort_values('Total Hours', ascending=False).head(5)

    # Create bar chart
    y_pos = range(len(app_stats))
    axes[idx].barh(y_pos, app_stats['Total Hours'].values, color=sns.color_palette('viridis', 5))
    axes[idx].set_yticks(y_pos)
    axes[idx].set_yticklabels(app_stats.index, fontsize=9)
    axes[idx].set_xlabel('Total Hours', fontsize=11)
    axes[idx].set_title(f'{category}\n(Top 5 Apps)', fontsize=12, fontweight='bold')
    axes[idx].invert_yaxis()

    # Add value labels
    for i, v in enumerate(app_stats['Total Hours'].values):
        axes[idx].text(v, i, f' {v:,.0f}h ({app_stats["Unique Users"].iloc[i]} users)',
                      va='center', fontsize=8)

    # Save detailed stats to CSV
    app_stats.to_csv(os.path.join(output_dir, f'top_apps_in_{category.lower()}.csv'))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '03a_top_apps_in_top_categories.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: 03a_top_apps_in_top_categories.png")

# Print summary table
print("\nTop 5 Apps by Category:")
for category in top_6_categories:
    category_df = df[df['app_category'] == category]
    app_stats = category_df.groupby('app_name')['total_time_hours'].sum().sort_values(ascending=False).head(5)
    print(f"\n{category}:")
    for app, hours in app_stats.items():
        print(f"  - {app}: {hours:,.1f} hours")

# ============================================================================
# 3. USER BEHAVIOR PATTERNS - How do individuals differ?
# ============================================================================
print("\n" + "="*80)
print("3. USER BEHAVIOR PATTERNS - Individual differences")
print("="*80)

# Calculate category diversity per user (how many categories they use)
user_category_diversity = df.groupby('user_id')['app_category'].nunique().reset_index()
user_category_diversity.columns = ['user_id', 'category_count']

user_total_time = df.groupby('user_id')['total_time_hours'].sum().reset_index()
user_total_time.columns = ['user_id', 'total_hours']

user_behavior = user_category_diversity.merge(user_total_time, on='user_id')
user_behavior = user_behavior.sort_values('total_hours', ascending=False)

print(f"\nUser Category Diversity:")
print(f"Average categories per user: {user_behavior['category_count'].mean():.1f}")
print(f"Median categories per user: {user_behavior['category_count'].median():.1f}")
print(f"Min/Max categories: {user_behavior['category_count'].min()} / {user_behavior['category_count'].max()}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Category diversity distribution
axes[0, 0].hist(user_behavior['category_count'], bins=20, color='skyblue', edgecolor='black')
axes[0, 0].axvline(user_behavior['category_count'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {user_behavior["category_count"].mean():.1f}')
axes[0, 0].axvline(user_behavior['category_count'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {user_behavior["category_count"].median():.1f}')
axes[0, 0].set_xlabel('Number of Different Categories Used', fontsize=12)
axes[0, 0].set_ylabel('Number of Users', fontsize=12)
axes[0, 0].set_title('Category Diversity Distribution Across Users', fontsize=14, fontweight='bold')
axes[0, 0].legend()

# Scatter: diversity vs total time
axes[0, 1].scatter(user_behavior['category_count'], user_behavior['total_hours'], alpha=0.6, s=80, color='purple')
axes[0, 1].set_xlabel('Number of Different Categories Used', fontsize=12)
axes[0, 1].set_ylabel('Total Screen Time (Hours)', fontsize=12)
axes[0, 1].set_title('Category Diversity vs Total Screen Time', fontsize=14, fontweight='bold')
# Add correlation
corr = user_behavior['category_count'].corr(user_behavior['total_hours'])
axes[0, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[0, 1].transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), verticalalignment='top')

# User dominant category (what category do they spend most time on?)
user_dominant_category = df.groupby(['user_id', 'app_category'])['total_time_hours'].sum().reset_index()
user_dominant_category = user_dominant_category.sort_values(['user_id', 'total_time_hours'], ascending=[True, False])
user_dominant_category = user_dominant_category.groupby('user_id').first().reset_index()

dominant_category_counts = user_dominant_category['app_category'].value_counts().head(15)
axes[1, 0].barh(range(len(dominant_category_counts)), dominant_category_counts.values, color='teal')
axes[1, 0].set_yticks(range(len(dominant_category_counts)))
axes[1, 0].set_yticklabels(dominant_category_counts.index)
axes[1, 0].set_xlabel('Number of Users', fontsize=12)
axes[1, 0].set_title('Most Common Dominant Category per User', fontsize=14, fontweight='bold')
axes[1, 0].invert_yaxis()
for i, v in enumerate(dominant_category_counts.values):
    axes[1, 0].text(v, i, f' {v}', va='center', fontsize=9)

# Category concentration (Gini coefficient style - are users focused on few categories or spread out?)
def calculate_category_concentration(user_id):
    user_data = df[df['user_id'] == user_id].groupby('app_category')['total_time_hours'].sum().sort_values(ascending=False)
    if len(user_data) == 0:
        return 0
    total = user_data.sum()
    if total == 0:
        return 0
    # Calculate what % of time is spent on top category
    return (user_data.iloc[0] / total) * 100

user_behavior['top_category_pct'] = user_behavior['user_id'].apply(calculate_category_concentration)

axes[1, 1].hist(user_behavior['top_category_pct'], bins=20, color='orange', edgecolor='black')
axes[1, 1].axvline(user_behavior['top_category_pct'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {user_behavior["top_category_pct"].mean():.1f}%')
axes[1, 1].set_xlabel('% of Time Spent on Top Category', fontsize=12)
axes[1, 1].set_ylabel('Number of Users', fontsize=12)
axes[1, 1].set_title('Usage Concentration - Focus vs Diversity', fontsize=14, fontweight='bold')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '04_user_behavior_patterns.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: 04_user_behavior_patterns.png")

# Save user behavior data
user_behavior.to_csv(os.path.join(output_dir, 'user_behavior_metrics.csv'), index=False)

# ============================================================================
# 4. CATEGORY COMBINATIONS - What categories are used together?
# ============================================================================
print("\n" + "="*80)
print("4. CATEGORY COMBINATIONS - What's used together?")
print("="*80)

# For each user, see what categories they use on the same day
daily_user_categories = df.groupby(['user_id', 'date', 'app_category'])['total_time_hours'].sum().reset_index()

# Create co-occurrence matrix
from itertools import combinations

category_pairs = []
for (user, date), group in daily_user_categories.groupby(['user_id', 'date']):
    categories = group['app_category'].unique()
    if len(categories) > 1:
        for cat1, cat2 in combinations(sorted(categories), 2):
            category_pairs.append((cat1, cat2))

if len(category_pairs) > 0:
    co_occurrence = pd.DataFrame(category_pairs, columns=['Category1', 'Category2'])
    co_occurrence_counts = co_occurrence.groupby(['Category1', 'Category2']).size().reset_index(name='Count')
    co_occurrence_counts = co_occurrence_counts.sort_values('Count', ascending=False)

    print(f"\nTop 20 Category Pairs Used Together:")
    print(co_occurrence_counts.head(20))

    co_occurrence_counts.to_csv(os.path.join(output_dir, 'category_co_occurrence.csv'), index=False)

    # Visualize top co-occurrences
    top_pairs = co_occurrence_counts.head(20)
    top_pairs['Pair'] = top_pairs['Category1'] + ' + ' + top_pairs['Category2']

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(range(len(top_pairs)), top_pairs['Count'].values, color='mediumpurple')
    ax.set_yticks(range(len(top_pairs)))
    ax.set_yticklabels(top_pairs['Pair'].values, fontsize=9)
    ax.set_xlabel('Co-occurrence Count (Same User, Same Day)', fontsize=12)
    ax.set_title('Top 20 Category Pairs Used Together', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    for i, v in enumerate(top_pairs['Count'].values):
        ax.text(v, i, f' {v}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_category_combinations.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: 05_category_combinations.png")

# ============================================================================
# 5. CATEGORY TRENDS OVER TIME
# ============================================================================
print("\n" + "="*80)
print("5. TRENDS OVER TIME - Are patterns changing?")
print("="*80)

# Daily trends for top categories
daily_category_trends = df[df['app_category'].isin(top_10_cats[:5])].groupby(['date', 'app_category'])['total_time_hours'].sum().reset_index()
daily_category_trends['date'] = pd.to_datetime(daily_category_trends['date'])
daily_category_trends = daily_category_trends.sort_values('date')

fig, ax = plt.subplots(figsize=(16, 8))
for category in top_10_cats[:5]:
    cat_data = daily_category_trends[daily_category_trends['app_category'] == category]
    ax.plot(cat_data['date'], cat_data['total_time_hours'], marker='o', label=category, linewidth=2, markersize=4)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Total Hours', fontsize=12)
ax.set_title('Daily Usage Trends for Top 5 Categories', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '06_trends_over_time.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: 06_trends_over_time.png")

# ============================================================================
# 5a. SCREENTIME DATA COMPLETENESS BY USER
# ============================================================================
print("\n" + "="*80)
print("5a. SCREENTIME DATA COMPLETENESS - How complete is data per user?")
print("="*80)

# Daily screentime by user
daily_user_screentime = df.groupby(['user_id', 'date']).agg({
    'total_time_hours': 'sum',
    'id': 'count'
}).reset_index()
daily_user_screentime.columns = ['user_id', 'date', 'total_hours', 'record_count']
daily_user_screentime['date'] = pd.to_datetime(daily_user_screentime['date'])

# Get all dates in range
all_dates = pd.date_range(daily_user_screentime['date'].min(), daily_user_screentime['date'].max(), freq='D')
all_users = sorted(df['user_id'].unique())

# Create completeness metrics
user_completeness = []
for user_id in all_users:
    user_data = daily_user_screentime[daily_user_screentime['user_id'] == user_id]
    days_with_data = len(user_data)
    total_days = len(all_dates)
    completeness_pct = (days_with_data / total_days) * 100
    total_hours = user_data['total_hours'].sum()
    avg_daily_hours = user_data['total_hours'].mean()

    user_completeness.append({
        'user_id': user_id,
        'days_with_data': days_with_data,
        'total_days': total_days,
        'completeness_pct': completeness_pct,
        'total_hours': total_hours,
        'avg_daily_hours': avg_daily_hours
    })

completeness_df = pd.DataFrame(user_completeness).sort_values('completeness_pct', ascending=False)
print("\nUser Screentime Completeness:")
print(completeness_df.to_string(index=False))
completeness_df.to_csv(os.path.join(output_dir, 'user_screentime_completeness.csv'), index=False)

# Visualization: Daily screentime trends by user
fig, axes = plt.subplots(2, 1, figsize=(18, 12))

# Plot 1: Daily total hours per user
colors_user = sns.color_palette('tab20', len(all_users))
for idx, user_id in enumerate(all_users):
    user_data = daily_user_screentime[daily_user_screentime['user_id'] == user_id].sort_values('date')

    # Fill in missing dates with 0
    user_series = user_data.set_index('date')['total_hours'].reindex(all_dates, fill_value=0)

    axes[0].plot(user_series.index, user_series.values,
                marker='o', label=f'User {user_id}', linewidth=2,
                markersize=3, alpha=0.8, color=colors_user[idx])

axes[0].set_xlabel('Date', fontsize=12)
axes[0].set_ylabel('Total Hours per Day', fontsize=12)
axes[0].set_title('Daily Screentime by User - Data Completeness Over Time', fontsize=14, fontweight='bold')
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1)
axes[0].grid(True, alpha=0.3)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 2: Daily record count per user (shows data collection frequency)
for idx, user_id in enumerate(all_users):
    user_data = daily_user_screentime[daily_user_screentime['user_id'] == user_id].sort_values('date')

    # Fill in missing dates with 0
    user_series = user_data.set_index('date')['record_count'].reindex(all_dates, fill_value=0)

    axes[1].plot(user_series.index, user_series.values,
                marker='o', label=f'User {user_id}', linewidth=2,
                markersize=3, alpha=0.8, color=colors_user[idx])

axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel('Number of Records per Day', fontsize=12)
axes[1].set_title('Daily Record Count by User - Data Collection Frequency', fontsize=14, fontweight='bold')
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1)
axes[1].grid(True, alpha=0.3)
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '06a_user_screentime_completeness.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: 06a_user_screentime_completeness.png")

# Additional visualization: Heatmap of data presence
print("\nGenerating data presence heatmap...")
presence_matrix = pd.DataFrame(index=all_dates, columns=all_users)
for user_id in all_users:
    user_data = daily_user_screentime[daily_user_screentime['user_id'] == user_id]
    for _, row in user_data.iterrows():
        presence_matrix.loc[row['date'], user_id] = row['total_hours']

presence_matrix = presence_matrix.fillna(0).astype(float)

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(presence_matrix.T, cmap='YlGnBu', cbar_kws={'label': 'Hours'}, ax=ax)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('User ID', fontsize=12)
ax.set_title('Screentime Data Presence Heatmap - Hours per User per Day', fontsize=14, fontweight='bold')

# Simplify x-axis labels to show fewer dates
n_dates = len(all_dates)
tick_spacing = max(1, n_dates // 15)  # Show ~15 date labels
ax.set_xticks(range(0, n_dates, tick_spacing))
ax.set_xticklabels([all_dates[i].strftime('%Y-%m-%d') for i in range(0, n_dates, tick_spacing)], rotation=45, ha='right')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '06b_data_presence_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: 06b_data_presence_heatmap.png")

# ============================================================================
# 6. INSIGHTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("KEY INSIGHTS SUMMARY")
print("="*80)

# Calculate key metrics
total_time = df['total_time_hours'].sum()
avg_session = df['total_time_minutes'].mean()
most_used_category = category_stats.index[0]
most_used_time = category_stats.iloc[0]['Total Hours']
most_adopted_category = category_stats.sort_values('Unique Users', ascending=False).index[0]
peak_hour = df.groupby('hour')['total_time_hours'].sum().idxmax()
peak_day = daily_pivot.sum(axis=1).idxmax()

insights = f"""
WHAT DOES THE CATEGORY BREAKDOWN TELL US?

1. DOMINANT CATEGORIES:
   - The most time-consuming category is '{most_used_category}' with {most_used_time:,.1f} hours total
   - Top 3 categories account for {category_stats.head(3)['% of Total Time'].sum():.1f}% of all screen time
   - {len(category_stats)} different app categories are being used

2. USER ADOPTION:
   - '{most_adopted_category}' is used by the most users ({category_stats.loc[most_adopted_category, 'Unique Users']} users)
   - Average user engages with {user_behavior['category_count'].mean():.1f} different categories
   - {(user_behavior['top_category_pct'] > 50).sum()} users spend >50% of time on a single category

3. TEMPORAL PATTERNS:
   - Peak usage hour: {peak_hour}:00
   - Peak usage day: {peak_day}
   - Usage patterns vary significantly by category (see heatmap)

4. USAGE DIVERSITY:
   - Correlation between category diversity and total time: {corr:.3f}
   - Average time concentration on top category: {user_behavior['top_category_pct'].mean():.1f}%

WHY DOES IT MATTER?

- Health & Well-being: Understanding which categories dominate can reveal potential concerns
  (e.g., excessive social media vs productive apps)
  
- Personalization: Different usage patterns suggest need for personalized interventions

- Data Quality: Categories with very few users or records may indicate data collection issues

- Research Design: Knowing peak usage times can inform when to deliver surveys/interventions

PATTERNS & INSIGHTS:

- Category co-occurrence reveals typical user workflows and habits
- Temporal patterns show when different types of apps are most relevant
- User diversity metrics identify different user archetypes (focused vs explorers)
- Trends over time can reveal study participation effects or external factors

All visualizations and detailed statistics have been saved to:
{output_dir}
"""

print(insights)

# Save insights to file
with open(os.path.join(output_dir, 'insights_summary.txt'), 'w') as f:
    f.write(insights)

print(f"\n✓ Analysis complete! All outputs saved to: {output_dir}")
print("="*80)
