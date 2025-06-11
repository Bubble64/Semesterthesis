#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import os
from scipy.stats import bootstrap

def load_cluster_data(path):
    df = pd.read_csv(path, parse_dates=['week_start_date'])
    df['cluster'] = df['cluster'].astype(int)
    return df


def load_workout_data(path):
    df = pd.read_csv(path, parse_dates=['ACTIVITY_START_TIME_LOCAL', 'ACTIVITY_END_TIME_LOCAL'])
    df['activity_date'] = df['ACTIVITY_START_TIME_LOCAL'].dt.date
    df['week_start_date'] = (
        pd.to_datetime(df['activity_date']) - \
        pd.to_timedelta(pd.to_datetime(df['activity_date']).dt.weekday, unit='d')
    )
    return df


def aggregate_workouts(df):
    agg = df.groupby(['USER_ID', 'week_start_date']).agg(
        total_duration_min=pd.NamedAgg(column='ACTIVITY_DURATION', aggfunc='sum'),
        total_calories=pd.NamedAgg(column='CALORIES_BURNED', aggfunc='sum'),
        avg_intensity_score=pd.NamedAgg(column='SCALED_RAW_INTENSITY_SCORE', aggfunc='mean'),
        workouts_count=pd.NamedAgg(column='ID', aggfunc='count')
    ).reset_index()
    cat_dur = (
        df.groupby(['USER_ID', 'week_start_date', 'CATEGORY'])['ACTIVITY_DURATION']
          .sum()
          .unstack(fill_value=0)
          .add_prefix('duration_')
          .reset_index()
    )
    merged = agg.merge(cat_dur, on=['USER_ID', 'week_start_date'], how='left')
    cat_cols = [c for c in merged.columns if c.startswith('duration_')]
    merged[cat_cols] = merged[cat_cols].fillna(0)
    return merged


def merge_active(cluster_df, workout_agg):
    active = cluster_df.merge(workout_agg, on=['USER_ID','week_start_date'], how='inner')
    numeric = active.select_dtypes(include=[np.number]).columns.tolist()
    active[numeric] = active[numeric].fillna(0)
    return active


def merge_all(cluster_df, workout_agg):
    # only include users who have at least one workout
    valid_users = workout_agg['USER_ID'].unique()
    cluster_filtered = cluster_df[cluster_df['USER_ID'].isin(valid_users)]
    all_w = cluster_filtered.merge(workout_agg, on=['USER_ID','week_start_date'], how='left')
    numeric = all_w.select_dtypes(include=[np.number]).columns.tolist()
    all_w[numeric] = all_w[numeric].fillna(0)
    return all_w


def analyze_and_save(df, static_metrics, output_prefix, output_dir, median_include_zeros=False):
    category_metrics = [c for c in df.columns if c.startswith('duration_')]
    metrics = static_metrics + category_metrics
    stats = []
    for metric in metrics:
        for cid, grp in df.groupby('cluster'):
            vals = grp[metric].dropna().values
            mean = vals.mean() if len(vals)>0 else np.nan
            if median_include_zeros:
                med_vals = vals
            else:
                med_vals = vals[vals>0]
            med = np.median(med_vals) if len(med_vals)>0 else np.nan
            if len(vals)>1:
                mean_res = bootstrap((vals,), np.mean, confidence_level=0.95, n_resamples=1000, method='percentile')
                mean_low, mean_high = mean_res.confidence_interval.low, mean_res.confidence_interval.high
            else:
                mean_low, mean_high = np.nan, np.nan
            if len(med_vals)>1:
                med_res = bootstrap((med_vals,), np.median, confidence_level=0.95, n_resamples=1000, method='percentile')
                med_low, med_high = med_res.confidence_interval.low, med_res.confidence_interval.high
            else:
                med_low, med_high = np.nan, np.nan
            stats.append({
                'metric': metric,
                'cluster': int(cid),
                'mean': mean,
                'mean_ci_lower': mean_low,
                'mean_ci_upper': mean_high,
                'median': med,
                'median_ci_lower': med_low,
                'median_ci_upper': med_high
            })
    df_stats = pd.DataFrame(stats)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{output_prefix}_cluster_stats.csv")
    df_stats.to_csv(path, index=False)
    print(f"Saved {output_prefix} stats to {path}")
    return df_stats


def main(cluster_csv, workout_csv, output_dir):
    clusters = load_cluster_data(cluster_csv)
    workouts = load_workout_data(workout_csv)
    agg = aggregate_workouts(workouts)
    static = ['total_duration_min','total_calories','avg_intensity_score','workouts_count']
    active_df = merge_active(clusters, agg)
    analyze_and_save(active_df, static, 'active_weeks', output_dir, median_include_zeros=False)
    all_df = merge_all(clusters, agg)
    analyze_and_save(all_df, static, 'all_weeks', output_dir, median_include_zeros=True)

if __name__=='__main__':
    p = argparse.ArgumentParser(description='Cluster vs metrics for active/all weeks')
    p.add_argument('cluster_csv', help='cluster assignments')
    p.add_argument('workout_csv', help='workout data')
    p.add_argument('-o','--output-dir', default='.', help='output dir')
    args = p.parse_args()
    main(args.cluster_csv, args.workout_csv, args.output_dir)
