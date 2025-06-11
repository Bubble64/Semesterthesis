import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def cramers_v(chi2, n, r, k):
    """Compute Cramer's V."""
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

# 1) Load data
demo = pd.read_csv('demographic_small.csv', parse_dates=['FIRST_SUBMISSION_DATE','LAST_SUBMISSION_DATE','BIRTHDAY'])
clust = pd.read_csv('cluster_survey_selected.csv', parse_dates=['week_start_date'])

# 2) Merge on USER_ID
df = pd.merge(demo, clust, on='USER_ID', how='inner')

# 3) Compute age at FIRST_SUBMISSION_DATE
df['age'] = ((df['FIRST_SUBMISSION_DATE'] - df['BIRTHDAY']).dt.days / 365.25).astype(int)

# 4) Bin age into quartiles
df['age_group'] = pd.qcut(df['age'], q=4, labels=['Q1','Q2','Q3','Q4'])

# 5) Gender vs Cluster
ct_gender = pd.crosstab(df['GENDER'], df['cluster'])
chi2_g, p_g, dof_g, exp_g = chi2_contingency(ct_gender)
n_g = ct_gender.values.sum()
r_g, k_g = ct_gender.shape

v_g = cramers_v(chi2_g, n_g, r_g, k_g)

print("=== Gender vs Cluster ===")
print(ct_gender)
print(f"Chi2 = {chi2_g:.2f}, dof = {dof_g}, p = {p_g:.3f}")
print(f"Cramer's V = {v_g:.3f}\n\"")

# 6) Age Group vs Cluster
ct_age = pd.crosstab(df['age_group'], df['cluster'])
chi2_a, p_a, dof_a, exp_a = chi2_contingency(ct_age)
n_a = ct_age.values.sum()
r_a, k_a = ct_age.shape

v_a = cramers_v(chi2_a, n_a, r_a, k_a)

print("=== Age Group vs Cluster ===")
print(ct_age)
print(f"Chi2 = {chi2_a:.2f}, dof = {dof_a}, p = {p_a:.3f}")
print(f"Cramer's V = {v_a:.3f}\n\"")

# 7) Save observed and expected tables
ct_gender.to_csv('ct_gender_cluster.csv')
ct_age.to_csv('ct_agegroup_cluster.csv')

exp_gender = pd.DataFrame(exp_g, index=ct_gender.index, columns=ct_gender.columns)
exp_age    = pd.DataFrame(exp_a, index=ct_age.index,    columns=ct_age.columns)
exp_gender.to_csv('expected_gender_cluster.csv')
exp_age.to_csv('expected_agegroup_cluster.csv')

print("Contingency tables, expected counts, and Cramer's V saved/calculated.")
