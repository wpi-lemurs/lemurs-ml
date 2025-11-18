#!/usr/bin/env python3
"""
Generated with GPT-5 Mini
generate_health_connect_steps.py

Generates a synthetic CSV dataset that mimics Android Health Connect "steps" sessions.

Default behavior:
- 40 app users
- 4 weeks (28 days) of data
- For each user, each day the user may produce between min_sessions_per_day and max_sessions_per_day sessions
  (default min=15, max=120). You can adjust these to produce smaller or larger datasets.
- Ensures that sessions for the same user do not overlap.
- Steps per session: 1-500
- Session duration: 1-300 seconds
- Timestamps use millisecond precision: YYYY-MM-DD HH:MM:SS.sss

Usage:
    python generate_health_connect_steps.py output.csv
    python generate_health_connect_steps.py output.csv --seed 42 --users 40 --weeks 4 --min-per-day 15 --max-per-day 120

"""
import csv
import sys
import argparse
import random
from datetime import datetime, timedelta

ISO_FMT = "%Y-%m-%d %H:%M:%S.%f"

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def format_ts(dt):
    # Format with millisecond precision: YYYY-MM-DD HH:MM:SS.sss
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def generate_sessions_for_user(user_id, start_date, days, min_per_day, max_per_day, rng):
    """
    Generate non-overlapping sessions for a single user across the given date range.
    Returns list of dicts with keys: app_user_id, steps, start_ts (datetime), end_ts (datetime)
    """
    sessions = []
    # Track last end per user to avoid overlaps (we will generate per-day sets and maintain order)
    for day_offset in range(days):
        # Optionally user may have 0 sessions that day; we randomize presence
        num = rng.randint(min_per_day, max_per_day)
        # For each session for this day, create a non-overlapping time by scheduling random start times,
        # then sort and shift to avoid overlaps.
        day_base = start_date + timedelta(days=day_offset)
        candidates = []
        for i in range(num):
            # Choose a random window between 05:00 and 23:00 local time
            hour = rng.randint(5, 22)
            minute = rng.randint(0, 59)
            second = rng.randint(0, 59)
            ms = rng.randint(0, 999)
            start = datetime(day_base.year, day_base.month, day_base.day, hour, minute, second, ms*1000)
            duration = rng.randint(1, 300)  # seconds
            end = start + timedelta(seconds=duration, milliseconds=rng.randint(0, 999))
            steps = rng.randint(1, 500)
            candidates.append((start, end, steps))
        # Sort by start and shift forward if overlapping
        candidates.sort(key=lambda x: x[0])
        for start, end, steps in candidates:
            if sessions and start <= sessions[-1]['end_ts']:
                # shift start to be after last end by 1-3 seconds to avoid overlap
                gap = rng.randint(1, 3)
                start = sessions[-1]['end_ts'] + timedelta(seconds=gap)
                # maintain duration
                dur = end - start
                # Recompute end based on original duration length:
                end = start + (end - start if dur.total_seconds() > 0 else timedelta(seconds=1))
            sessions.append({
                'app_user_id': user_id,
                'steps': steps,
                'start_ts': start,
                'end_ts': end
            })
    return sessions

def generate_dataset(output_path, users=40, weeks=4, start_date_str="2025-10-21",
                     min_per_day=15, max_per_day=120, seed=None):
    rng = random.Random(seed)
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    days = weeks * 7
    all_sessions = []
    for user_id in range(1, users + 1):
        user_sessions = generate_sessions_for_user(user_id, start_date, days, min_per_day, max_per_day, rng)
        all_sessions.extend(user_sessions)
    # Sort globally by start timestamp (optional)
    all_sessions.sort(key=lambda s: (s['start_ts'], s['app_user_id']))
    # Write CSV
    with open(output_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'app_user_id', 'steps', 'start_timestamp', 'end_timestamp'])
        for idx, s in enumerate(all_sessions, start=1):
            writer.writerow([
                idx,
                s['app_user_id'],
                s['steps'],
                format_ts(s['start_ts']),
                format_ts(s['end_ts'])
            ])
    print(f"Wrote {len(all_sessions)} rows to {output_path}")

def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("output", help="Output CSV path")
    ap.add_argument("--users", type=int, default=40, help="Number of app users (default 40)")
    ap.add_argument("--weeks", type=int, default=4, help="Number of weeks of data (default 4)")
    ap.add_argument("--start-date", default="2025-10-21", help="Start date YYYY-MM-DD (default 2025-10-21)")
    ap.add_argument("--min-per-day", type=int, default=15, help="Min sessions per user per day (default 15)")
    ap.add_argument("--max-per-day", type=int, default=120, help="Max sessions per user per day (default 120)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    args = ap.parse_args(argv[1:])
    if args.min_per_day < 0 or args.max_per_day < args.min_per_day:
        ap.error("Invalid per-day session bounds")
    generate_dataset(args.output, users=args.users, weeks=args.weeks,
                     start_date_str=args.start_date,
                     min_per_day=args.min_per_day, max_per_day=args.max_per_day,
                     seed=args.seed)

if __name__ == "__main__":
    main(sys.argv)