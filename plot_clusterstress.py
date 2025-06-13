import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for remote cluster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
DATA_CLUSTERS = 'cluster_survey_selected.csv'
DATA_SURVEY   = 'mhs_survey_stress.csv'
BINS          = [0, 14, 27, 41]
BIN_LABELS    = [0, 1, 2]
BOOTSTRAPS    = 1000

# Load and merge data
clusters = pd.read_csv(DATA_CLUSTERS, parse_dates=['week_start_date'])
survey   = pd.read_csv(DATA_SURVEY)
# Parse dates
survey['survey_date'] = pd.to_datetime(survey['SUBMITDATE'], format='%m/%d/%y')
# Merge within 7-day window
data = pd.merge(clusters, survey, on='USER_ID')
mask = (
    (data['survey_date'] >= data['week_start_date']) &
    (data['survey_date'] <= data['week_start_date'] + pd.Timedelta(days=7))
)
data = data.loc[mask].copy()

# Compute total score and stress level
question_cols = [c for c in survey.columns if c not in ('USER_ID', 'SUBMITDATE', 'survey_date')]
data['sum_score'] = data[question_cols].sum(axis=1)
data['stress_level'] = pd.cut(
    data['sum_score'], bins=BINS, labels=BIN_LABELS, include_lowest=True
).astype(int)

# 1) Generate boxplots of each metric by cluster
metrics = question_cols + ['sum_score', 'stress_level']
for metric in metrics:
    fig, ax = plt.subplots(figsize=(8, 5))
    data.boxplot(column=metric, by='cluster', ax=ax)
    ax.set_title(f'Boxplot of {metric} by Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel(metric)
    plt.suptitle('')  # remove automatic title
    plt.tight_layout()
    fig.savefig(f'boxplot_{metric}_by_cluster.png')
    plt.close(fig)

# Helper: bootstrap CIs
def bootstrap_ci(values, func=np.mean, b=BOOTSTRAPS, alpha=0.05):
    n = len(values)
    stats = np.array([func(np.random.choice(values, size=n, replace=True)) for _ in range(b)])
    lower = np.percentile(stats, 100 * (alpha / 2))
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return lower, upper

# 2) Compute mean, median, and 95% CIs per cluster
records = []
for cluster, df in data.groupby('cluster'):
    rec = {'cluster': cluster}
    for col in metrics:
        vals = df[col].dropna().values
        rec[f'mean_{col}'] = vals.mean()
        rec[f'median_{col}'] = np.median(vals)
        lo_m, hi_m = bootstrap_ci(vals, func=np.mean)
        lo_md, hi_md = bootstrap_ci(vals, func=np.median)
        rec[f'mean_ci_lower_{col}'] = lo_m
        rec[f'mean_ci_upper_{col}'] = hi_m
        rec[f'median_ci_lower_{col}'] = lo_md
        rec[f'median_ci_upper_{col}'] = hi_md
    records.append(rec)

stats_df = pd.DataFrame(records)
stats_df.to_csv('stats_by_cluster_mean_median_CI95.csv', index=False)
print('Saved stats with 95% CIs to stats_by_cluster_mean_median_CI95.csv')
