import pandas as pd
import numpy as np
from umap import UMAP
import matplotlib
matplotlib.use("Agg")
import pingouin as pg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.fft import rfft
from scipy.stats import linregress
from statsmodels.tsa.stattools import acf
import math
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr
import itertools
import re
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# Helpers
# =============================================================================

def remove_outliers(X, threshold=3):
    mask = (np.abs(X) <= threshold).all(axis=1)
    return X[mask], mask, (~mask).sum()

def cross_validated_db(X, k, folds=5, random_state=42):
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    scores = []
    for tr, _ in kf.split(X):
        Xtr = X[tr]
        if len(Xtr) < k:
            continue
        labs = KMeans(n_clusters=k, random_state=random_state).fit_predict(Xtr)
        try:
            scores.append(davies_bouldin_score(Xtr, labs))
        except:
            scores.append(np.inf)
    return np.mean(scores) if scores else np.inf

def stability_ari(X, k, runs=5):
    labs = [KMeans(n_clusters=k, random_state=seed).fit_predict(X)
            for seed in range(runs)]
    pairs = itertools.combinations(labs, 2)
    aris = [adjusted_rand_score(a, b) for a, b in pairs]
    return np.mean(aris) if aris else 1.0

# =============================================================================
# 1) Load & Prepare Sleep Data
# =============================================================================
df = pd.read_csv('slpwk_survey.csv')
df['LOWER_DAYS'] = pd.to_datetime(df['LOWER_DAYS'])
df = df.sort_values(['USER_ID','LOWER_DAYS'])

all_sensors = ['SLEEP_PERFORMANCE_SCORE','SLEEP_LATENCY','SKIN_TEMP_CELSIUS','BLOOD_OXYGEN','RESPIRATORY_RATE','RESTING_HEART_RATE','HRV','SLEEP_CONSISTENCY','SLEEP_EFFICIENCY','TIME_IN_BED_MINUTES','MINUTES_OF_SLEEP','LIGHT_SLEEP_DURATION_MINUTES','LIGHT_SLEEP_PERCENT','REM_SLEEP_DURATION_MINUTES','REM_SLEEP_PERCENT','SLOW_WAVE_SLEEP_DURATION_MINUTES','SLOW_WAVE_SLEEP_PERCENT','WAKE_DURATION_MINUTES','WAKE_DURATION_PERCENT','RESTORATIVE_SLEEP_MINUTES','RESTORATIVE_SLEEP_PERCENT','DISTURBANCES','AROUSAL_TIME_MINUTES','SLEEP_DEBT_MINUTES','SLEEP_NEED_MINUTES','RECOVERY_SCORE','SLEEP_START_LOCAL','SLEEP_END_LOCAL','CALORIES_BURNED','DAY_AVG_HEART_RATE','DAY_MAX_HEART_RATE','SCALED_DAY_STRAIN','DAY_STRAIN'
]
df[all_sensors] = df[all_sensors].apply(pd.to_numeric, errors='coerce')
df['week'] = df.groupby('USER_ID').cumcount() // 7

# =============================================================================
# 2) Compute Weekly Features (Descriptive + Raw) and Detect Complete Sensors
# =============================================================================
def sample_entropy(series, m=2, r=None):
    """
    Compute Sample Entropy of a 1D array `series`.
    Returns NaN if there are insufficient matches.
    """
    x = np.array(series, dtype=float)
    N = len(x)
    if r is None:
        r = 0.2 * np.std(x, ddof=0)
    def _phi(m):
        # Build overlapping m‐length vectors
        X = np.array([x[i:i+m] for i in range(N - m + 1)])
        C = 0
        for i in range(len(X)):
            # Chebyshev distance
            d = np.max(np.abs(X - X[i]), axis=1)
            C += np.sum(d <= r) - 1
        # Normalize: total comparisons per template = (N-m+1)-1
        M = len(X)
        return C / (M*(M-1)) if M > 1 else 0

    phi_m   = _phi(m)
    phi_m1  = _phi(m+1)
    # avoid log(0) or negative
    if phi_m <= 0 or phi_m1 <= 0:
        return np.nan
    return -math.log(phi_m1 / phi_m)

def compute_weekly_features(df, sensors):
    """
    Returns:
      desc_df: DataFrame of descriptive + trend/ACF/entropy features per (USER_ID,week)
      complete_sensors: list of sensors that appeared at least once with no NaNs
    """
    sensor_valid = {s: False for s in sensors}
    desc_rows = []
    raw_rows  = []

    # Build rows
    for (uid, wk), g in df.groupby(['USER_ID','week']):
        if len(g) != 7:
            continue
        desc = {'USER_ID': uid,
                'week_start_date': g['LOWER_DAYS'].iloc[0]}
        raw  = {'USER_ID': uid,
                'week_start_date': g['LOWER_DAYS'].iloc[0]}

        any_ok = False
        for s in sensors:
            arr = pd.to_numeric(g[s], errors='coerce').values
            if np.isnan(arr).any():
                continue
            # mark sensor valid
            sensor_valid[s] = True
            any_ok = True

            # Descriptive
            desc[f"{s}_mean"]   = arr.mean()
            desc[f"{s}_std"]    = arr.std()
            desc[f"{s}_min"]    = arr.min()
            desc[f"{s}_max"]    = arr.max()
            desc[f"{s}_median"] = np.median(arr)

            # Trend (slope)
            slope = linregress(np.arange(7), arr).slope
            desc[f"{s}_slope"] = slope

            # Lag-1 autocorrelation
            acfs = acf(arr, nlags=1, fft=False)
            desc[f"{s}_acf1"] = acfs[1] if len(acfs)>1 else np.nan

            # Sample entropy
            desc[f"{s}_sampen"] = sample_entropy(arr)

            # Raw series placeholders
            for i, v in enumerate(arr):
                raw[f"{s}_{i}"] = v

        if any_ok:
            desc_rows.append(desc)
            raw_rows.append(raw)

    # Create DataFrames
    desc_df = pd.DataFrame(desc_rows)
    raw_df  = pd.DataFrame(raw_rows)

    # Determine which sensors ever had a complete week
    complete_sensors = [s for s, ok in sensor_valid.items() if ok]
    if not complete_sensors:
        raise RuntimeError("No sensor has a complete 7-day block.")

    # Compute PCA(1) on each sensor's raw series
    for s in complete_sensors:
        cols = [f"{s}_{i}" for i in range(7)]
        # select only rows where all 7 raw columns are present
        sub = raw_df[cols].dropna()
        if sub.shape[0] >= 2:
            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(sub.values).ravel()
            desc_df.loc[sub.index, f"{s}_pc1"] = pc1
        else:
            desc_df[f"{s}_pc1"] = np.nan

    # Reorder columns: USER_ID, week_start_date, then features for each complete sensor
    ordered_cols = ['USER_ID','week_start_date']
    for s in complete_sensors:
        ordered_cols += [
            f"{s}_mean", f"{s}_std", f"{s}_min", f"{s}_max", f"{s}_median",
            f"{s}_slope", f"{s}_acf1", f"{s}_sampen", f"{s}_pc1"
        ]
    desc_df = desc_df[ordered_cols]

    return desc_df, complete_sensors  
feat_df, complete_sensors = compute_weekly_features(df, all_sensors)

# =============================================================================
# 3) Load & Merge Survey Data
# =============================================================================
survey = pd.read_csv('mhs_survey_stress.csv', parse_dates=['SUBMITDATE'])
survey.columns = survey.columns.str.strip()
survey.rename(columns={'SUBMITDATE':'survey_date'}, inplace=True)
survey['survey_date'] = pd.to_datetime(survey['survey_date'])
survey_cols = [c for c in survey.columns if re.fullmatch(r"s\d+", c)]

merged = feat_df.merge(survey, on='USER_ID', how='left')
merged['start'] = merged['week_start_date']
merged['end']   = merged['start'] + pd.Timedelta(days=7)
mask = (merged['survey_date']>=merged['start']) & (merged['survey_date']<merged['end'])
merged = merged[mask]

avg_survey = (
    merged.groupby(['USER_ID','week_start_date'])[survey_cols]
          .mean()
          .reset_index()
)
feat_df = feat_df.merge(avg_survey, on=['USER_ID','week_start_date'], how='inner')
if feat_df.empty:
    raise RuntimeError("No surveys matched the weekly windows.")

# =============================================================================
# 4) Feature Selection via Pearson, Spearman, MI (with fallback)
# =============================================================================
# physical feature columns
phys_cols = [c for c in feat_df.columns 
             if any(c.startswith(f"{s}_") for s in complete_sensors)]

# standardize for correlation
Xphys = pd.DataFrame(
    StandardScaler().fit_transform(feat_df[phys_cols]),
    columns=phys_cols
)
# Drop any feature columns that still contain NaNs
Xphys = Xphys.dropna(axis=1)
phys_cols = Xphys.columns.tolist()

#repeated measures
# 1) Split multi-week vs single-week users
user_counts = feat_df['USER_ID'].value_counts()
multi_users  = user_counts[user_counts >= 2].index
single_users = user_counts[user_counts == 1].index

df_multi  = feat_df[feat_df['USER_ID'].isin(multi_users)].reset_index(drop=True)
df_single = feat_df[feat_df['USER_ID'].isin(single_users)].reset_index(drop=True)

# 2) Residualize phys features ONLY
Xw = df_multi[phys_cols].copy()
for col in phys_cols:
    subj_mean = df_multi.groupby('USER_ID')[col].transform('mean')
    Xw[col] -= subj_mean

# 3) Residualize survey items ONLY
Yw = df_multi[survey_cols].copy()
for col in survey_cols:
    subj_mean = df_multi.groupby('USER_ID')[col].transform('mean')
    Yw[col] -= subj_mean


# feat_df has USER_ID, feature columns, and survey columns
user_feat = feat_df.groupby('USER_ID')[phys_cols].mean()
user_survey = feat_df.groupby('USER_ID')[survey_cols].mean()

agg_corr = pd.concat([user_feat, user_survey], axis=1) \
             .corr().loc[phys_cols, survey_cols]
agg_corr.to_csv("pearson_between_subjects.csv")

pearson_mat = pd.concat([Xphys, feat_df[survey_cols]], axis=1) \
               .corr().loc[phys_cols, survey_cols]
pearson_mat.to_csv("pearson_feature_survey_corr.csv")

pearson_score = pearson_mat.abs().mean(axis=1)

# Threshold
threshold = 0.10
selected = pearson_score[pearson_score >= threshold].index.tolist()

# Failsafe: if none pass, pick top-10 by absolute Pearson
if not selected:
    selected = pearson_score.sort_values(ascending=False).head(10).index.tolist()
    print(f"No features passed |r|≥{threshold}; falling back to top 10 by |r|:")
else:
    print(f"Selected features with |r|≥{threshold}:")

print(selected)
# Fallback: top-10 by MI
if not selected:
    selected = mi.sort_values(ascending=False).head(10).index.tolist()
    print(f"No feature passed thresholds; fallback top-10 by MI:\n{selected}")
else:
    print("Selected features by Pearson/Spearman/MI thresholds:")
    for f in selected:
        print(f" {f}: pearson={pearson[f]:.2f}, spearman={spearman[f]:.2f}, mi={mi[f]:.3f}")

# =============================================================================
# C) PRUNE HIGHLY CORRELATED FEATURES
# =============================================================================
# Build the DataFrame of just the selected features
Xsel = Xphys[selected]

# Compute feature–feature Pearson correlation
corr_ff = Xsel.corr().abs()

# To avoid dropping both in a pair, look at upper triangle only:
upper = corr_ff.where(np.triu(np.ones(corr_ff.shape), k=1).astype(bool))
upper.to_csv("feature-feature_matrix.csv")
sub_fs = pearson_mat.loc[selected,survey_cols]
sub_ff = corr_ff
#plot heatmap of feature feature and feature survey
def plot_heatmap(mat, fname):
    plt.figure(figsize=(max(8, mat.shape[1]*0.3),
                       max(6, mat.shape[0]*0.3)))
    im = plt.imshow(mat.values, aspect='auto', cmap='viridis',
                    vmin=-0.15, vmax=0.15)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(np.arange(mat.shape[1]), mat.columns,
               rotation=90, fontsize=8)
    plt.yticks(np.arange(mat.shape[0]), mat.index,
               fontsize=8)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()

#plot_heatmap(sub_ff, 'top10_feature_feature_heatmap.png')
plot_heatmap(sub_fs, 'top10_feature_survey_heatmap.png')


# Any column with a correlation > 0.9 to another gets dropped
to_drop = [col for col in upper.columns if any(upper[col] > 0.90)]
print(f"Dropping {len(to_drop)} highly‐collinear features:", to_drop)

# Final feature set
selected = [f for f in selected if f not in to_drop]
Xfinal = Xsel[selected].values

# =============================================================================
# 5) Remove Outliers
# =============================================================================
Xsel = Xphys[selected].values
Xf, mask_out, n_out = remove_outliers(Xsel, threshold=3)
print(f"Removed {n_out} outliers (of {len(Xsel)} samples).")
feat_df = feat_df[mask_out].reset_index(drop=True)

# =============================================================================
# 6) UMAP → 2D
# =============================================================================
emb = UMAP(n_components=2, random_state=42).fit_transform(Xf)

# =============================================================================
# 7) Choose k via CV Davies–Bouldin
# =============================================================================
best_db, best_k, best_labels = np.inf, None, None
for k in range(2, min(9, len(emb)+1)):
    db = cross_validated_db(emb, k)
    print(f"k={k}, CV Davies–Bouldin={db:.3f}")
    if db < best_db:
        best_db, best_k = db, k
        best_labels = KMeans(n_clusters=k, random_state=42).fit_predict(emb)
print(f"Chosen k={best_k} (CV DB={best_db:.3f})")

# =============================================================================
# 8) Stability ARI
# =============================================================================
ari = stability_ari(emb, best_k)
print(f"Cluster stability (ARI): {ari:.3f}")

# =============================================================================
# 9) Save Results
# =============================================================================
out = feat_df[['USER_ID','week_start_date'] + selected].copy()
out['umap1'], out['umap2'] = emb[:,0], emb[:,1]
out['cluster'] = best_labels
out.to_csv("cluster_survey_selected.csv", index=False)
print("Saved → cluster_survey_selected.csv")

# 10) UMAP Plot
plt.figure(figsize=(6,5))
plt.scatter(emb[:,0], emb[:,1], c=best_labels, cmap='viridis', s=30, alpha=0.8)
plt.title(f"Clusters (k={best_k})")
plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
plt.tight_layout(); plt.savefig("umap_clusters_survey.png"); plt.close()
print("Saved → umap_clusters_survey.png")
