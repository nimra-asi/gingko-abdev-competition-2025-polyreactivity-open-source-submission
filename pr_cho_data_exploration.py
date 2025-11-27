# Databricks notebook source
!pip install -r requirements_regression.txt

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

import pickle
import json
from pathlib import Path
from typing import Dict, List, Literal, Any, Optional
from tqdm import tqdm

# COMMAND ----------

path_to_add = "/Workspace/Users/nimra.asi.ext@boehringer-ingelheim.com/gingko-abdev-competition"

if path_to_add not in sys.path:
    sys.path.append(path_to_add)

# COMMAND ----------

from utils.utils import create_feature_space, create_feature_space_full_seqs

# COMMAND ----------

# MAGIC %md
# MAGIC # read in data

# COMMAND ----------

df = pd.read_csv("/Workspace/Users/nimra.asi.ext@boehringer-ingelheim.com/gingko-abdev-competition/data/GDPa1_v1.2_20250814.csv")
print(df.shape)
df.head(2)

df = df.rename(columns=({"SEC %Monomer": "SEC_perc_Monomer",
                         "AC-SINS_pH6.0": "AC_SINS_pH6.0",
                         "AC-SINS_pH7.4": "AC_SINS_pH7.4",
                         "vh_protein_sequence": "vh",
                         "vl_protein_sequence": "vl"}))

# df.isna().sum()
lower_cols = [x.lower() for x in df.columns]
# print(lower_cols)
df.columns = lower_cols

df.head(2)

# COMMAND ----------

df.isna().sum()

# COMMAND ----------

# reading in earlier ver of data for additional datapoints

dfe = pd.read_excel("/Workspace/Users/nimra.asi.ext@boehringer-ingelheim.com/gingko-abdev-competition/data/gdpa1_v1.0_20250502.xlsx", sheet_name="Prior literature Data")
print(dfe.shape)
dfe.head(2)

# COMMAND ----------

dfe.isna().sum()

# COMMAND ----------

pr_cols = ['pr_cho','pr_ova','makowski2021_pspscore_smp', 'shehata2019_psrscore_smp', 'jain2017_psrscore_smp', 'makowski2021_pspscore_ova']

# COMMAND ----------

# MAGIC %md
# MAGIC # analyze property of interest

# COMMAND ----------

# MAGIC %md
# MAGIC #### identify missing values

# COMMAND ----------

check = df.loc[df["pr_cho"].isna()]
print(check.shape)
check

missing_abs = check["antibody_name"].unique().tolist()
len(missing_abs)

# COMMAND ----------

check = dfe.loc[dfe["antibody_name"].isin(missing_abs)]
print(check.shape)
check.shape

# COMMAND ----------

display(df["pr_cho"].describe())
print("\nlimited range in values")

# COMMAND ----------

# MAGIC %md
# MAGIC #### analyze how close the external values are to internally measured ones 

# COMMAND ----------

dfe.columns

# COMMAND ----------

comb = df.merge(dfe, on=["antibody_name", "antibody_id"], how="inner")
print(df.shape)
print(comb.shape)
comb.head(2)

# COMMAND ----------

comb[pr_cols].isna().sum()

# COMMAND ----------

comb = df.merge(dfe, on=["antibody_name", "antibody_id"], how="inner")
print(df.shape)
print(comb.shape)
comb.head(2)

target_col = 'jain2017_psrscore_smp'
# target_col = 'makowski2021_pspscore_smp'
# target_col = 'shehata2019_psrscore_smp'
# target_col = 'makowski2021_pspscore_ova'
# target_cols = 'pr_ova'

merged_df = comb.dropna(subset=['pr_cho', target_col]).reset_index(drop=True).copy()
print(merged_df.shape)

merged_df["abs_diff"] = np.abs(merged_df["hic"] - merged_df[target_col])
merged_df["rel_diff_perc"] = merged_df["abs_diff"] / merged_df[["hic", target_col]].mean(axis=1)

# 3. Print Summary
print("--- Alignment Summary ---")
print(f"Number of antibodies common to both DFs: {len(merged_df)}")
print("\nDescriptive Statistics for Differences:")
print(merged_df[['abs_diff', 'rel_diff_perc']].describe())


# COMMAND ----------

max_val = max(merged_df["pr_cho"].max(), merged_df[target_col].max()) * 1.05

# 2. Create Scatter Plot
plt.figure(figsize=(7, 6))
plt.scatter(merged_df["pr_cho"], merged_df[target_col], alpha=0.6, edgecolors='w', linewidth=0.5)

# Add the line of perfect agreement (y=x line)
plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Agreement ($y=x$)')

orig_col = 'pr_cho'
plt.title(f'Comparison of {orig_col} Values Across Sources')
plt.xlabel(f"{orig_col} from Gingko")
plt.ylabel(f"{orig_col} from {target_col}")
# plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# If points cluster tightly around the red dashed line, substitution is likely okay.

# COMMAND ----------

from scipy import stats

correlation, p_corr = stats.pearsonr(merged_df["pr_cho"], merged_df[target_col])
correlation, p_corr = stats.spearmanr(merged_df["pr_cho"], merged_df[target_col])

print(f"Correlation between HIC values: {correlation:.2f}")
print(f"P-value for Correlation: {p_corr:.3e}")

t_stat, p_ttest = stats.ttest_rel(merged_df["pr_cho"], merged_df[target_col])

print(f"\n**Paired T-Test P-value:** {p_ttest:.3e}")
print(f"T-statistic: {t_stat:.3f}")

# Interpretation:
alpha = 0.05
if p_ttest < alpha:
    print(f"**Conclusion:** P-value is < {alpha}. **REJECT** the null hypothesis.")
    print("There is a **statistically significant difference** in the mean values. Substitution is RISKY.")
else:
    print(f"**Conclusion:** P-value is > {alpha}. **FAIL TO REJECT** the null hypothesis.")
    print("There is **NO significant difference** in the mean values. Substitution is likely OK.")

# COMMAND ----------

target_col

# COMMAND ----------

target_col

test = comb.loc[comb["antibody_name"].isin(missing_abs)].reset_index(drop=True)
print(test.shape)
test.head(2)

test[pr_cols].isna().sum()

test['pr_cho'].fillna(test[target_col], inplace=True)
test[pr_cols].isna().sum()

# COMMAND ----------

print("can impute missing pr_cho values with jain et al 2017 prpscore_smp values directly")

# COMMAND ----------

# MAGIC %md
# MAGIC #### look for duplicates in the dataset

# COMMAND ----------

# look for duplicated seqs: iscalimab and lucatumumab

d = df.loc[df.duplicated(subset=["hc_protein_sequence", "lc_protein_sequence"], keep="first")]
# print(d.shape)
display(d)
d = df.loc[df.duplicated(subset=["hc_protein_sequence"])]
display(d)
# print(d.shape)
d = df.loc[df.duplicated(subset=["lc_protein_sequence"])]
# print(d.shape)
display(d)

seq1 = df.loc[df["antibody_name"]=="iscalimab"]["hc_protein_sequence"].values[0]
seq2 = df.loc[df["antibody_name"]=="lucatumumab"]["hc_protein_sequence"].values[0]
if seq1 == seq2: print("iscalimab and lucatumumab have the same hc seq")

seq1 = df.loc[df["antibody_name"]=="iscalimab"]["lc_protein_sequence"].values[0]
seq2 = df.loc[df["antibody_name"]=="lucatumumab"]["lc_protein_sequence"].values[0]
if seq1 == seq2: print("iscalimab and lucatumumab have the same lc seq")

check = df.loc[((df["antibody_name"]=="iscalimab") | (df["antibody_name"]=="lucatumumab"))]
display(check)


# COMMAND ----------

print("iscalimab and luctumab have exactly duplicated seqs, remove one of them?")
print("lucatumab also has hc and lc similarities with other antibodies in the dataset, and they are in diff hierarchical cluster folds from lucatumab")
print("CONCLUSION: remove lucatumab from the dataset moving forward as there will be no loss of training data and removal of data leakage that may be artifically inflating model performance")

# COMMAND ----------

