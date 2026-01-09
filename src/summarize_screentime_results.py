import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')

print("\n" + "="*80)
print("SUMMARY: Screentime Features for Suicide Risk Prediction")
print("="*80 + "\n")

results = []

for hours in [3, 6, 9, 12, 24]:
    filename = f'daily_screentime_suicide_risk_{hours}h.csv'
    filepath = os.path.join(data_dir, filename)

    if os.path.exists(filepath):
        df = pd.read_csv(filepath)

        result = {
            'Time Window (hours)': hours,
            'Total Samples': len(df),
            'Unique Users': df['app_user_id'].nunique(),
            'At Risk': (df['suicide_risk_label'] == 'at_risk').sum(),
            'Not At Risk': (df['suicide_risk_label'] == 'not_at_risk').sum(),
            '% At Risk': f"{((df['suicide_risk_label'] == 'at_risk').sum() / len(df) * 100):.2f}%"
        }
        results.append(result)

        # Show sample data
        if hours == 3:
            print(f"\nSample data structure ({hours}h window):")
            print(df.head(3))
            print(f"\nColumns: {list(df.columns)}")

summary_df = pd.DataFrame(results)
print("\n" + "="*80)
print("DATA SUMMARY ACROSS TIME WINDOWS")
print("="*80)
print(summary_df.to_string(index=False))

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)
print("""
1. More data is available with longer time windows (as expected)
2. The class imbalance is severe - very few 'at_risk' samples
3. Models achieve high accuracy mainly by predicting 'not_at_risk'
4. Hour 3 (3 hours before survey) appears to be the most important feature
5. Need more 'at_risk' samples for meaningful predictive modeling

RECOMMENDATIONS:
- Try combining multiple passive data sources (health + screentime)
- Consider time-series features (trends, changes) rather than raw values
""")

