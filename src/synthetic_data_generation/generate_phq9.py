#!/usr/bin/env python3
"""
generate_phq9_responses.py

Generates a synthetic CSV dataset that mimics weekly PHQ-9 survey responses.

Behavior:
- Default: 40 app users, 4 weeks of data (start date default 2025-10-21).
- Most users have 1 response per week; a small fraction of users are "occasional"
  responders and skip some weeks. Per-week response probability is high for most users.
- Questions q1..q9 range 0-3.
- phq9_total_score is the sum of q1..q9.
- severity_label mapping:
    0-4 -> minimal
    5-9 -> mild
    10-14 -> moderate
    15-19 -> moderately_severe
    20+ -> severe

Usage:
    python3 generate_phq9_responses.py output.csv
    python3 generate_phq9_responses.py output.csv --users 40 --weeks 4 --start-date 2025-10-21 --seed 42

Options:
    --users N            Number of users (default 40)
    --weeks W            Number of weeks (default 4)
    --start-date YYYY-MM-DD  Start date (default 2025-10-21)
    --seed S             Random seed for reproducible output
    --response-prob P    Base per-week response probability (default 0.9)
    --occasional-frac F Fraction of users designated "occasional" (default 0.12)
    --occasional-prob P  Per-week response prob for occasional users (default 0.45)
"""
import csv
import sys
import argparse
import random
from datetime import datetime, timedelta

ISO_FMT = "%Y-%m-%d %H:%M:%S.%f"

def format_ts(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def severity_label_from_score(score):
    if score <= 4:
        return "minimal"
    if score <= 9:
        return "mild"
    if score <= 14:
        return "moderate"
    if score <= 19:
        return "moderately_severe"
    return "severe"

def weighted_response(rng):
    # Realistic bias toward lower answers: weights for 0,1,2,3
    weights = [0.40, 0.30, 0.20, 0.10]
    r = rng.random()
    cum = 0.0
    for i, w in enumerate(weights):
        cum += w
        if r <= cum:
            return i
    return 3

def generate_responses(users, weeks, start_date, base_prob, occasional_frac,
                       occasional_prob, rng):
    all_rows = []
    # Pick a subset of users to be occasional responders
    occasional_users = set(rng.sample(range(1, users+1), max(1, int(users * occasional_frac))))
    survey_id = 1
    for user in range(1, users + 1):
        for week in range(weeks):
            # Determine whether the user responds this week
            prob = occasional_prob if user in occasional_users else base_prob
            if rng.random() > prob:
                continue  # skipped this week
            # choose a day within the week and time of day
            week_start = start_date + timedelta(days=week*7)
            day_offset = rng.randint(0, 6)
            hour = rng.randint(6, 22)
            minute = rng.randint(0, 59)
            second = rng.randint(0, 59)
            ms = rng.randint(0, 999)
            ts = datetime(week_start.year, week_start.month, week_start.day,
                          0, 0, 0) + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second, milliseconds=ms)
            # generate q1..q9
            qs = [weighted_response(rng) for _ in range(9)]
            total = sum(qs)
            label = severity_label_from_score(total)
            row = {
                "survey_response_id": survey_id,
                "app_user_id": user,
                "q1": qs[0],
                "q2": qs[1],
                "q3": qs[2],
                "q4": qs[3],
                "q5": qs[4],
                "q6": qs[5],
                "q7": qs[6],
                "q8": qs[7],
                "q9": qs[8],
                "phq9_total_score": total,
                "severity_label": label,
                "timestamp": ts  # will not be saved as column, only used for sorting if desired
            }
            all_rows.append(row)
            survey_id += 1
    # sort by timestamp then user (optional)
    all_rows.sort(key=lambda r: (r["timestamp"], r["app_user_id"]))
    # reassign sequential survey_response_id after sorting for chronological ordering
    for i, r in enumerate(all_rows, start=1):
        r["survey_response_id"] = i
    return all_rows

def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        headers = ["survey_response_id", "app_user_id",
                   "q1","q2","q3","q4","q5","q6","q7","q8","q9",
                   "phq9_total_score","severity_label",
                   "response_timestamp"]
        writer.writerow(headers)
        for r in rows:
            writer.writerow([
                r["survey_response_id"],
                r["app_user_id"],
                r["q1"], r["q2"], r["q3"], r["q4"], r["q5"], r["q6"], r["q7"], r["q8"], r["q9"],
                r["phq9_total_score"],
                r["severity_label"],
                format_ts(r["timestamp"])
            ])
    print(f"Wrote {len(rows)} rows to {path}")

def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("output", help="Output CSV path")
    ap.add_argument("--users", type=int, default=40, help="Number of app users (default 40)")
    ap.add_argument("--weeks", type=int, default=4, help="Number of weeks (default 4)")
    ap.add_argument("--start-date", default="2025-10-21", help="Start date YYYY-MM-DD (default 2025-10-21)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    ap.add_argument("--response-prob", type=float, default=0.90, help="Base per-week response probability (default 0.90)")
    ap.add_argument("--occasional-frac", type=float, default=0.12, help="Fraction of occasional users who skip more often (default 0.12)")
    ap.add_argument("--occasional-prob", type=float, default=0.45, help="Per-week response prob for occasional users (default 0.45)")
    args = ap.parse_args(argv[1:])
    rng = random.Random(args.seed)
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    rows = generate_responses(users=args.users, weeks=args.weeks, start_date=start_date,
                              base_prob=args.response_prob,
                              occasional_frac=args.occasional_frac,
                              occasional_prob=args.occasional_prob,
                              rng=rng)
    write_csv(args.output, rows)

if __name__ == "__main__":
    main(sys.argv)