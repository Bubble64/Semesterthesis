import pandas as pd
import matplotlib
matplotlib.use('Agg')       # Non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


# 1) Load the data
clusters = pd.read_csv(
    'cluster_survey_selected.csv',
    parse_dates=['week_start_date']
)
survey = pd.read_csv('mhs_survey_stress.csv')

# 2) Parse survey dates and merge within 7-day window
survey['survey_date'] = pd.to_datetime(
    survey['SUBMITDATE'], format='%m/%d/%y'
)
merged = pd.merge(clusters, survey, on='USER_ID')
mask = (
    (merged['survey_date'] >= merged['week_start_date']) &
    (merged['survey_date'] <= merged['week_start_date'] + pd.Timedelta(days=7))
)
merged = merged.loc[mask].copy()

# 3) Sum across all survey questions and bin into stress levels
survey_cols = [
    c for c in survey.columns
    if c not in ('USER_ID','SUBMITDATE','survey_date')
]
merged['sum_score'] = merged[survey_cols].sum(axis=1)

bins   = [0, 14, 27, 41]   # 0–13, 14–26, 27–40
labels = [0,     1,     2]
merged['stress_level'] = pd.cut(
    merged['sum_score'],
    bins=bins,
    labels=labels,
    include_lowest=True
).astype(int)

# 4) Save the full merged table
merged.to_csv('merged_results.csv', index=False)
print("Saved merged results to merged_results.csv")

# 5) Compute mean & std by cluster, including stress_level
stat_cols = survey_cols + ['stress_level']
grouped   = merged.groupby('cluster')[stat_cols]
mean_df   = grouped.mean()
std_df    = grouped.std()

stats_df = pd.concat({'mean': mean_df, 'std': std_df}, axis=1)
stats_df.columns = [f"{stat}_{col}" for stat, col in stats_df.columns]
stats_df.to_csv('stats_by_cluster_stats.csv')
print("Saved combined mean/std to stats_by_cluster_stats.csv")

# 6) Plot mean ± std for each cluster separately
for cluster in mean_df.index:
    means = mean_df.loc[cluster]
    stds  = std_df.loc[cluster]

    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(means.index, means.values, yerr=stds.values, capsize=5)
    ax.set_xlabel('Survey Question or Stress Level')
    ax.set_ylabel('Mean Value')
    ax.set_title(f'Cluster {cluster}: Mean ± STD of Survey Metrics')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    filename = f"Plot_stress_cluster_{cluster}.png"
    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved plot for cluster {cluster} as {filename}")

# 7a) Overall correlation with the numeric cluster label

to_agg = ['cluster'] + survey_cols + ['sum_score', 'stress_level']

# 2) Group by USER_ID and take the mean of each column
agg_df = (
    merged[to_agg + ['USER_ID']]
    .groupby('USER_ID', as_index=False)
    .mean()
)

# 3) Compute the correlation matrix on the user-level means
corr_agg = agg_df[to_agg].corr()

# 4) Extract the correlations with the (mean) cluster label
corr_with_cluster_agg = corr_agg['cluster'].loc[survey_cols + ['sum_score', 'stress_level']]

# 5) Save to CSV
corr_with_cluster_agg.to_csv('correlations_with_cluster_user_agg.csv', header=True)

corr = merged[survey_cols + ['sum_score','stress_level','cluster']].corr()
corr_with_cluster = corr['cluster'].loc[survey_cols + ['sum_score','stress_level']]
corr_with_cluster.to_csv('correlations_with_cluster.csv', header=True)
print("Saved correlations to correlations_with_cluster.csv")

fig, ax = plt.subplots(figsize=(8,5))
corr_with_cluster.plot.bar(ax=ax)
ax.axhline(0)
ax.set_ylabel('Pearson r with Cluster')
ax.set_title('Correlation of Variables with Cluster Assignment')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
fig.savefig("Plot_corr_overall.png")
plt.close(fig)
print("Saved overall correlation plot as Plot_corr_overall.png")

# 1) Aggregate per USER_ID & cluster: mean of metrics within each user-cluster pair
agg_df = (
    merged.groupby(['USER_ID', 'cluster'], as_index=False)
          .agg({**{col: 'mean' for col in survey_cols + ['sum_score', 'stress_level']}})
)

# 2) One-hot encode the cluster label on this grouped data
df_corr_agg = pd.get_dummies(agg_df, columns=['cluster'], prefix='cluster')

# 3) Define variables to correlate
cluster_cols = [c for c in df_corr_agg.columns if c.startswith('cluster_')]
vars_to_corr = survey_cols + ['sum_score', 'stress_level']

# 4) Compute and save the correlation matrix for reference
full_corr = df_corr_agg[cluster_cols + vars_to_corr].corr()
corr_cluster_vs_questions = full_corr.loc[cluster_cols, vars_to_corr]

# 5) Compute r and p-value for each cluster–variable pair
results = []
for cluster_col in cluster_cols:
    for var in vars_to_corr:
        r, p = pearsonr(df_corr_agg[cluster_col], df_corr_agg[var])
        results.append({
            'cluster':    cluster_col,
            'variable':   var,
            'r_value':    r,
            'p_value':    p
        })

signif_df = pd.DataFrame(results)

# 6) Save outputs
corr_cluster_vs_questions.to_csv('corr_cluster_vs_questions_user_cluster_agg.csv')
signif_df.to_csv('signif_cluster_vs_questions_user_cluster_agg.csv', index=False)

print("Saved per-cluster vs. question correlations to corr_cluster_vs_questions_user_cluster_agg.csv")
print("Saved significance (r & p-values) to signif_cluster_vs_questions_user_cluster_agg.csv")

# 1) Load the data
clusters = pd.read_csv(
    'cluster_survey_selected.csv',
    parse_dates=['week_start_date']
)
survey = pd.read_csv('mhs_survey_stress.csv')

# 2) Parse survey dates and merge within 7-day window
survey['survey_date'] = pd.to_datetime(
    survey['SUBMITDATE'], format='%m/%d/%y'
)
merged = pd.merge(clusters, survey, on='USER_ID')
mask = (
    (merged['survey_date'] >= merged['week_start_date']) &
    (merged['survey_date'] <= merged['week_start_date'] + pd.Timedelta(days=7))
)
merged = merged.loc[mask].copy()

# 3) Sum across all survey questions and bin into stress levels
survey_cols = [
    c for c in survey.columns
    if c not in ('USER_ID','SUBMITDATE','survey_date')
]
merged['sum_score'] = merged[survey_cols].sum(axis=1)

bins   = [0, 14, 27, 41]   # 0–13, 14–26, 27–40
labels = [0,     1,     2]
merged['stress_level'] = pd.cut(
    merged['sum_score'],
    bins=bins,
    labels=labels,
    include_lowest=True
).astype(int)


# 5) Compute bootstrap CIs for mean & median by cluster
stat_cols = survey_cols + ['sum_score', 'stress_level']
clusters_list = merged['cluster'].unique()
B = 1000  # number of bootstrap resamples

# prepare a dict to collect results
records = []

for cluster in clusters_list:
    sub = merged[merged['cluster'] == cluster]
    n = len(sub)
    rec = {'cluster': cluster}
    
    for col in stat_cols:
        vals = sub[col].dropna().values
        # original point estimates
        rec[f'mean_{col}']   = vals.mean()
        rec[f'median_{col}'] = np.median(vals)
        
        # bootstrap
        boot_means   = np.empty(B)
        boot_medians = np.empty(B)
        for i in range(B):
            samp = np.random.choice(vals, size=n, replace=True)
            boot_means[i]   = samp.mean()
            boot_medians[i] = np.median(samp)
        
        lo, hi = np.percentile(boot_means,   [2.5, 97.5])
        rec[f'mean_ci_lower_{col}'] = lo
        rec[f'mean_ci_upper_{col}'] = hi
        
        lo, hi = np.percentile(boot_medians, [2.5, 97.5])
        rec[f'median_ci_lower_{col}'] = lo
        rec[f'median_ci_upper_{col}'] = hi
    
    records.append(rec)

# 6) Build DataFrame and save
stats_df = pd.DataFrame.from_records(records)
stats_df.to_csv('stats_by_cluster_mean_median_CI95.csv', index=False)
print("Saved combined mean/median and 95% CIs to stats_by_cluster_mean_median_CI95.csv")