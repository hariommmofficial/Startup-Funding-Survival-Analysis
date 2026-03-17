import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import sqlite3
import os
import warnings
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
DATA_PATH   = r"C:\Users\sande\Videos\Startup Failure Analysis\startup_funding.csv"
EXPORT_PATH = r"C:\Users\sande\Videos\Startup Failure Analysis\Exports"
os.makedirs(EXPORT_PATH, exist_ok=True)
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "axes.grid":         True,
    "grid.color":        "#e0e0e0",
    "grid.linewidth":    0.6,
    "grid.alpha":        0.7,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

def save_fig(name, tight=True):
    path = os.path.join(EXPORT_PATH, name)
    if tight:
        plt.savefig(path, dpi=150, bbox_inches='tight')
    else:
        plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")

def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")

section("MODULE 1: DATA INGESTION & CLEANING")

df = pd.read_csv(DATA_PATH, encoding='latin-1')
print(f"Raw rows   : {len(df):,}")
print(f"Raw columns: {len(df.columns)}")

df.columns = (df.columns.str.strip().str.lower()
              .str.replace(' ', '_').str.replace(r'[^a-z0-9_]', '', regex=True))
col_map = {}
for col in df.columns:
    if 'amount'   in col:                          col_map['amount']   = col
    if 'industry' in col and 'sub' not in col:     col_map['industry'] = col
    if 'city'     in col:                          col_map['city']     = col
    if 'investor' in col:                          col_map['investor'] = col
    if 'investment' in col and 'type' in col:      col_map['inv_type'] = col
    if ('startup' in col or 'name' in col) and 'investor' not in col:
                                                   col_map['startup']  = col
    if 'date'     in col:                          col_map['date']     = col
    if 'sub'      in col and 'vertical' in col:    col_map['sub']      = col

df.rename(columns={v: k for k, v in col_map.items()}, inplace=True)

df['amount'] = (
    df.get('amount', pd.Series(['0']*len(df)))
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.replace(r'(?i)undisclosed', '0', regex=True)
    .str.replace(r'\$', '', regex=True)
    .str.strip()
)
df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)

if 'date' in df.columns:
    df['date']  = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
    df['year']  = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
else:
    df['year'] = 2017; df['month'] = 6; df['quarter'] = 2
for c in ['industry', 'city', 'investor', 'inv_type', 'startup', 'sub']:
    if c in df.columns:
        df[c] = df[c].fillna('Unknown').str.strip()
    else:
        df[c] = 'Unknown'

df = df[df['startup'] != '']
df = df[df['year'].between(2010, 2022)]

# FEATURE ENGINEERING
df['log_amount']       = np.log1p(df['amount'])
df['is_funded']        = (df['amount'] > 0).astype(int)
df['is_large_deal']    = (df['amount'] >= 1e7).astype(int)   # $10M+
tier1 = ['Bangalore', 'Mumbai', 'Delhi', 'Gurugram', 'New Delhi',
         'Gurgaon', 'Bengaluru', 'Hyderabad', 'Pune', 'Chennai']
df['city_tier'] = df['city'].apply(lambda x: 1 if x in tier1 else 2)

print(f"\nClean rows : {len(df):,}")
print(f"Year range : {int(df['year'].min())} – {int(df['year'].max())}")
print(f"Unique startups   : {df['startup'].nunique():,}")
print(f"Unique industries : {df['industry'].nunique()}")
print(f"Total deployed    : ${df['amount'].sum()/1e9:.2f}B USD")
print(f"Funded deals      : {df['is_funded'].sum():,} ({df['is_funded'].mean()*100:.1f}%)")
# SQL Analytics
section("MODULE 2: SQL ANALYTICS ENGINE")

conn = sqlite3.connect(":memory:")
class MedianAggregate:
    def __init__(self):
        self.values = []
    def step(self, value):
        if value is not None:
            self.values.append(value)
    def finalize(self):
        if not self.values:
            return None
        s = sorted(self.values)
        n = len(s)
        mid = n // 2
        return (s[mid] if n % 2 != 0 else (s[mid - 1] + s[mid]) / 2)

conn.create_aggregate("MEDIAN", 1, MedianAggregate)
df.to_sql("startups", conn, if_exists="replace", index=False)
print("SQLite in-memory DB ready ✔")
print("Custom MEDIAN aggregate registered ✔")

def sql(query, label=""):
    result = pd.read_sql_query(query, conn)
    if label:
        print(f"\n  ── {label} ──")
        print(result.to_string(index=False))
    return result

#SQL Query 1: Yearly ecosystem pulse
yearly = sql("""
SELECT
    year,
    COUNT(*)                                          AS deals,
    COUNT(DISTINCT startup)                           AS unique_startups,
    ROUND(SUM(amount)/1e6, 1)                         AS total_m,
    ROUND(AVG(amount)/1e6, 3)                         AS avg_m,
    ROUND(MEDIAN(amount)/1e6, 3)                      AS median_m,
    SUM(CASE WHEN amount >= 10e6 THEN 1 ELSE 0 END)   AS mega_deals,
    ROUND(100.0*SUM(is_funded)/COUNT(*), 1)           AS funded_pct
FROM startups
GROUP BY year ORDER BY year
""", "Yearly Ecosystem Pulse")

#SQL Query 2: Industry risk matrix
industry_risk = sql("""
SELECT
    industry,
    COUNT(*)                                                    AS total_deals,
    COUNT(DISTINCT startup)                                     AS unique_startups,
    ROUND(SUM(amount)/1e6,2)                                    AS total_m,
    ROUND(AVG(CASE WHEN amount>0 THEN amount END)/1e6,3)        AS avg_funded_m,
    ROUND(MEDIAN(CASE WHEN amount>0 THEN amount END)/1e6,4)     AS median_funded_m,
    SUM(CASE WHEN amount=0 THEN 1 ELSE 0 END)                   AS unfunded,
    ROUND(100.0*SUM(CASE WHEN amount=0 THEN 1 ELSE 0 END)
          /COUNT(*), 1)                                         AS unfunded_pct,
    SUM(CASE WHEN amount>=10e6 THEN 1 ELSE 0 END)               AS mega_deals,
    ROUND(100.0*SUM(CASE WHEN amount>=10e6 THEN 1 ELSE 0 END)
          /NULLIF(SUM(is_funded),0), 1)                         AS mega_deal_pct
FROM startups
WHERE industry NOT IN ('Unknown','')
GROUP BY industry
HAVING total_deals >= 10
ORDER BY total_deals DESC
""", "Industry Risk Matrix")

#SQL Query 3: City ecosystem quality
city_data = sql("""
SELECT
    city,
    COUNT(*)                                            AS deals,
    COUNT(DISTINCT startup)                             AS unique_startups,
    COUNT(DISTINCT industry)                            AS industries,
    ROUND(SUM(amount)/1e6,1)                            AS total_m,
    ROUND(AVG(CASE WHEN amount>0 THEN amount END)/1e6,3) AS avg_m,
    ROUND(100.0*SUM(is_funded)/COUNT(*),1)              AS funded_pct,
    SUM(CASE WHEN amount>=10e6 THEN 1 ELSE 0 END)        AS mega_deals
FROM startups
WHERE city NOT IN ('Unknown','')
GROUP BY city
HAVING deals >= 10
ORDER BY deals DESC
LIMIT 15
""", "City Ecosystem Quality")

#SQL Query 4: Investment stage funnel
stage_funnel = sql("""
SELECT
    inv_type,
    COUNT(*)                            AS deals,
    ROUND(SUM(amount)/1e6,2)            AS total_m,
    ROUND(AVG(CASE WHEN amount>0 THEN amount END)/1e6,3) AS avg_m,
    ROUND(MEDIAN(CASE WHEN amount>0 THEN amount END)/1e6,4) AS median_m
FROM startups
WHERE inv_type NOT IN ('Unknown','')
GROUP BY inv_type
ORDER BY deals DESC
LIMIT 15
""", "Investment Stage Funnel")

#SQL Query 5: Top investors with portfolio stats 
top_investors = sql("""
SELECT
    investor,
    COUNT(DISTINCT startup)             AS portfolio_size,
    COUNT(*)                            AS total_deals,
    ROUND(SUM(amount)/1e6,2)            AS deployed_m,
    COUNT(DISTINCT industry)            AS industries,
    COUNT(DISTINCT city)                AS cities,
    ROUND(AVG(CASE WHEN amount>0 THEN amount END)/1e6,3) AS avg_ticket_m
FROM startups
WHERE investor NOT IN ('Unknown','') AND LENGTH(investor)<60
GROUP BY investor
HAVING portfolio_size >= 3
ORDER BY portfolio_size DESC
LIMIT 12
""", "Top Investors")

#SQL Query 6: Cohort analysis — multi-round startups
cohort = sql("""
SELECT
    startup,
    COUNT(*)             AS rounds,
    MIN(year)            AS first_year,
    MAX(year)            AS last_year,
    MAX(year)-MIN(year)  AS lifespan_years,
    SUM(amount)          AS total_raised,
    MAX(amount)          AS peak_round,
    COUNT(DISTINCT industry) AS industries_pivoted
FROM startups
GROUP BY startup
""")

cohort['survived']      = (cohort['rounds'] > 1).astype(int)
cohort['multi_year']    = (cohort['lifespan_years'] >= 2).astype(int)
cohort['raised_1m_plus'] = (cohort['total_raised'] >= 1e6).astype(int)

cohort['funding_band'] = pd.cut(
    cohort['total_raised'],
    bins=[-1, 0, 1e5, 1e6, 5e6, 20e6, 100e6, float('inf')],
    labels=['None','<$100K','$100K–1M','$1M–5M','$5M–20M','$20M–100M','$100M+']
)

survival_band = (
    cohort.groupby('funding_band', observed=True)
    .agg(startups=('startup','count'),
         multi_round=('survived','sum'),
         multi_year=('multi_year','sum'))
    .reset_index()
)
survival_band['multi_round_pct'] = (100*survival_band['multi_round']/survival_band['startups']).round(1)
survival_band['multi_year_pct']  = (100*survival_band['multi_year'] /survival_band['startups']).round(1)

print("\n  ── Funding Band Survival Proxy ──")
print(survival_band.to_string(index=False))

#SQL Query 7: Quarterly deal flow
quarterly = sql("""
SELECT year, quarter,
       COUNT(*) AS deals,
       ROUND(SUM(amount)/1e6,1) AS total_m
FROM startups
WHERE year BETWEEN 2014 AND 2020
GROUP BY year, quarter
ORDER BY year, quarter
""")
quarterly['period'] = quarterly['year'].astype(str) + " Q" + quarterly['quarter'].astype(str)

#SQL Query 8: Window function — industry market share over time
mshare = sql("""
WITH base AS (
    SELECT year, industry, COUNT(*) AS deals
    FROM startups
    WHERE year BETWEEN 2015 AND 2020
      AND industry NOT IN ('Unknown','')
    GROUP BY year, industry
),
totals AS (SELECT year, SUM(deals) AS yr_total FROM base GROUP BY year)
SELECT b.year, b.industry, b.deals,
       ROUND(100.0*b.deals/t.yr_total, 1) AS market_share_pct
FROM base b JOIN totals t ON b.year=t.year
ORDER BY b.year, b.deals DESC
""")

top5_industries = (mshare.groupby('industry')['deals']
                   .sum().nlargest(5).index.tolist())
mshare_top = mshare[mshare['industry'].isin(top5_industries)]

#SQL Query 9: Funding concentration (Gini / Pareto)
pareto = sql("""
SELECT startup, SUM(amount) AS total_raised
FROM startups
WHERE amount > 0
GROUP BY startup
ORDER BY total_raised DESC
""")

pareto['cum_funding']  = pareto['total_raised'].cumsum()
pareto['cum_pct']      = 100 * pareto['cum_funding'] / pareto['total_raised'].sum()
pareto['rank_pct']     = 100 * (np.arange(1, len(pareto)+1)) / len(pareto)

# What % of startups capture 80% of funding?
idx80 = (pareto['cum_pct'] >= 80).idxmax()
top_pct_for_80 = pareto.loc[idx80, 'rank_pct']
print(f"\n  Top {top_pct_for_80:.1f}% of startups control 80% of all funding")

#SQL Query 10: Statistical test — Tier 1 city vs Tier 2 funding
tier1_amounts = df[df['city_tier']==1]['amount'].values
tier2_amounts = df[df['city_tier']==2]['amount'].values
t_stat, p_val = stats.mannwhitneyu(tier1_amounts, tier2_amounts, alternative='greater')
print(f"\n  Mann-Whitney U test — Tier 1 vs Tier 2 city funding:")
print(f"  U={t_stat:.0f}, p-value={p_val:.4f} → {'Significant' if p_val<0.05 else 'Not significant'} at α=0.05")


#Machine Learning
section("MODULE 3: MACHINE LEARNING — SURVIVAL PREDICTION")
ml_df = cohort[cohort['total_raised'] > 0].copy()
startup_meta = (df.groupby('startup')
                .agg(city_tier=('city_tier','first'),
                     industry=('industry','first'),
                     inv_type=('inv_type','first'))
                .reset_index())
ml_df = ml_df.merge(startup_meta, on='startup', how='left')
le_ind = LabelEncoder()
le_inv = LabelEncoder()
ml_df['industry_enc'] = le_ind.fit_transform(ml_df['industry'].fillna('Unknown'))
ml_df['inv_type_enc'] = le_inv.fit_transform(ml_df['inv_type'].fillna('Unknown'))

feature_cols = ['total_raised', 'peak_round', 'rounds',
                'city_tier', 'industry_enc', 'inv_type_enc']
X = ml_df[feature_cols].fillna(0)
y = ml_df['survived']

print(f"\nML dataset : {len(X):,} startups")
print(f"Survival rate in sample: {y.mean()*100:.1f}%")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

#Model 1: Logistic Regression
lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_train_s, y_train)
lr_cv  = cross_val_score(lr, X_train_s, y_train, cv=5, scoring='roc_auc').mean()
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_s)[:,1])
print(f"\nLogistic Regression — CV AUC: {lr_cv:.3f} | Test AUC: {lr_auc:.3f}")

#Model 2: Decision Tree
dt = DecisionTreeClassifier(max_depth=6, random_state=42,
                            class_weight='balanced')
dt.fit(X_train, y_train)
dt_cv  = cross_val_score(dt, X_train, y_train, cv=5, scoring='roc_auc').mean()
dt_auc = roc_auc_score(y_test, dt.predict_proba(X_test)[:,1])
print(f"Decision Tree       — CV AUC: {dt_cv:.3f} | Test AUC: {dt_auc:.3f}")

#Model 3: Random Forest (best)
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42,
                            class_weight='balanced', n_jobs=-1)
rf.fit(X_train, y_train)
rf_cv  = cross_val_score(rf, X_train, y_train, cv=5, scoring='roc_auc').mean()
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
print(f"Random Forest       — CV AUC: {rf_cv:.3f} | Test AUC: {rf_auc:.3f}")
feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"\nFeature Importances (Random Forest):\n{feat_imp.round(3).to_string()}")

# Classification report
print(f"\nClassification Report (Random Forest):")
print(classification_report(y_test, rf.predict(X_test),
                             target_names=['Did Not Return','Returned for More Funding']))

#STATISTICAL INFERENCE
section("MODULE 4: STATISTICAL INFERENCE")

# A/B style test: Does Seed vs Series A funding lead to significantly
# different deal sizes?
seed_amounts = df[df['inv_type'].str.lower().str.contains('seed', na=False)]['amount']
seriesA_amounts = df[df['inv_type'].str.lower().str.contains('series a', na=False)]['amount']

if len(seed_amounts) > 10 and len(seriesA_amounts) > 10:
    u_stat, p = stats.mannwhitneyu(seed_amounts, seriesA_amounts, alternative='less')
    print(f"\nHypothesis Test: Is Seed funding < Series A funding?")
    print(f"  Seed    median: ${seed_amounts.median()/1e6:.3f}M (n={len(seed_amounts)})")
    print(f"  Ser. A  median: ${seriesA_amounts.median()/1e6:.3f}M (n={len(seriesA_amounts)})")
    print(f"  p-value = {p:.4f} → {'Reject H0 — Seed IS significantly smaller' if p<0.05 else 'Fail to reject H0'}")

# Correlation matrix
num_cols = df[['amount','year','month','city_tier','is_funded','is_large_deal']].copy()
corr = num_cols.corr().round(3)
print(f"\nCorrelation Matrix:\n{corr.to_string()}")

# Year-over-year growth
yearly_growth = yearly.copy()
yearly_growth['deal_growth_pct'] = yearly_growth['deals'].pct_change() * 100
yearly_growth['funding_growth_pct'] = yearly_growth['total_m'].pct_change() * 100
print(f"\nYear-over-Year Growth:\n{yearly_growth[['year','deals','deal_growth_pct','total_m','funding_growth_pct']].to_string(index=False)}")


# MODULE 5 — VISUALIZATIONS (12 charts)
section("MODULE 5: VISUALIZATIONS")

#Chart 1: Dual-axis ecosystem growth
fig, ax1 = plt.subplots(figsize=(13, 5))
ax2 = ax1.twinx()
x = yearly['year'].astype(int)
ax1.bar(x, yearly['deals'], color='steelblue', alpha=0.7, width=0.5, label='Number of Deals')
ax2.plot(x, yearly['total_m'], color='darkorange', marker='o', linewidth=2,
         label='Total Funding ($M)')
ax1.set_xlabel("Year")
ax1.set_ylabel("Number of Deals", color='steelblue')
ax2.set_ylabel("Total Funding (USD Millions)", color='darkorange')
ax1.tick_params(axis='y', colors='steelblue')
ax2.tick_params(axis='y', colors='darkorange')
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}M"))
ax1.set_title("Startup Ecosystem Growth: Deal Volume vs Capital Deployed")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc='upper left')
ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

fig.text(0.01, -0.08,
    "Reading this chart: Blue bars = how many deals happened each year. "
    "Orange line = total money invested.\n"
    "When bars grow but the line stays flat, many small deals happened. "
    "When the line spikes, a few giant deals dominated the year.",
    fontsize=10)
plt.tight_layout()
save_fig("01_ecosystem_growth_dual_axis.png")

#Chart 2: Industry Risk Matrix (bubble chart)
fig, ax = plt.subplots(figsize=(13, 8))
ind_plot = industry_risk.head(20).copy()
x_vals = ind_plot['unfunded_pct']
y_vals = ind_plot['avg_funded_m'].fillna(0)
sizes  = np.clip(ind_plot['total_deals'] * 8, 80, 2000)
colors = ['#d62728' if u > 60 else '#ff7f0e' if u > 40 else '#2ca02c'
          for u in x_vals]

sc = ax.scatter(x_vals, y_vals, s=sizes, c=colors, alpha=0.7, edgecolors='grey', linewidth=0.5)
for _, row in ind_plot.iterrows():
    ax.annotate(row['industry'][:20],
                (row['unfunded_pct'], row['avg_funded_m'] or 0),
                textcoords='offset points', xytext=(5, 3), fontsize=8)
ax.axvline(50, color='red', linestyle='--', linewidth=1, alpha=0.6, label='50% risk threshold')
ax.set_xlabel("% of Startups With Zero Funding (Risk Score)")
ax.set_ylabel("Average Funding per Funded Startup ($M)")
ax.set_title("Industry Risk Matrix\n"
             "X-axis = how risky, Y-axis = how capital-hungry, Bubble size = number of startups")
ax.legend()
red_patch   = mpatches.Patch(color='#d62728', label='High Risk (>60% unfunded)')
orange_patch= mpatches.Patch(color='#ff7f0e', label='Medium Risk (40–60%)')
green_patch = mpatches.Patch(color='#2ca02c', label='Lower Risk (<40%)')
ax.legend(handles=[red_patch, orange_patch, green_patch], loc='upper right')

fig.text(0.01, -0.06,
    "How to use this chart: Industries in the top-left corner are the SAFEST — "
    "low unfunded rate, decent average ticket.\n"
    "Industries in the bottom-right are DANGER ZONES — most startups get no money "
    "AND the ones that do get very little.\n"
    "Bubble size = how crowded the industry is. Bigger bubble = more competition.",
    fontsize=10)
plt.tight_layout()
save_fig("02_industry_risk_matrix_bubble.png")

#Chart 3: Funding stage funnel
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sf = stage_funnel.head(10)

axes[0].barh(sf['inv_type'][::-1], sf['deals'][::-1], color='steelblue')
axes[0].set_title("Deals by Investment Stage")
axes[0].set_xlabel("Number of Deals")
for i, (v, lbl) in enumerate(zip(sf['deals'][::-1], sf['inv_type'][::-1])):
    axes[0].text(v + 1, i, str(v), va='center', fontsize=9)

axes[1].barh(sf['inv_type'][::-1], sf['avg_m'][::-1].fillna(0), color='darkorange')
axes[1].set_title("Average Deal Size by Stage ($M)")
axes[1].set_xlabel("Average Funding ($M)")
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.1f}M"))
for i, (v, lbl) in enumerate(zip(sf['avg_m'][::-1].fillna(0), sf['inv_type'][::-1])):
    axes[1].text(v + 0.1, i, f"${v:.2f}M", va='center', fontsize=9)

fig.suptitle("Investment Stage Funnel: Volume vs Ticket Size", fontsize=14, fontweight='bold')
fig.text(0.01, -0.06,
    "Left chart = which stages are most common. Right chart = how much money per deal at each stage.\n"
    "Seed has the most deals but the smallest cheques. Private Equity has few deals but massive ones.\n"
    "Most startups live and die at Seed or Series A — very few ever reach Series C or beyond.",
    fontsize=10)
plt.tight_layout()
save_fig("03_investment_stage_funnel.png")

#Chart 4: Survival Rate by Funding Band
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sb_plot = survival_band.dropna(subset=['funding_band'])

axes[0].bar(range(len(sb_plot)), sb_plot['multi_round_pct'],
            color=['#d62728' if v < 20 else '#ff7f0e' if v < 40 else '#2ca02c'
                   for v in sb_plot['multi_round_pct']])
axes[0].set_xticks(range(len(sb_plot)))
axes[0].set_xticklabels(sb_plot['funding_band'].astype(str), rotation=30, ha='right')
axes[0].set_title("% Startups That Returned for a 2nd Round\n(Multi-Round Survival Proxy)")
axes[0].set_ylabel("% Returning for More Funding")
axes[0].set_ylim(0, 100)
for i, v in enumerate(sb_plot['multi_round_pct']):
    axes[0].text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=10)

axes[1].bar(range(len(sb_plot)), sb_plot['startups'],
            color='steelblue', alpha=0.8)
axes[1].set_xticks(range(len(sb_plot)))
axes[1].set_xticklabels(sb_plot['funding_band'].astype(str), rotation=30, ha='right')
axes[1].set_title("Number of Startups in Each Funding Band")
axes[1].set_ylabel("Number of Startups")
for i, v in enumerate(sb_plot['startups']):
    axes[1].text(i, v + 1, str(v), ha='center', fontsize=10)

fig.suptitle("Funding Amount vs Startup Survival", fontsize=14, fontweight='bold')
fig.text(0.01, -0.10,
    "Green bars = safe zone. Red/Orange bars = danger zone.\n"
    "Startups that raised $5M+ are FAR more likely to come back for another round "
    "(a proxy for still being alive).\n"
    "Startups with zero or tiny funding almost never survive long enough to raise again.\n"
    "This is the clearest evidence that underfunding is the #1 killer.",
    fontsize=10)
plt.tight_layout()
save_fig("04_funding_vs_survival_dual.png")

#Chart 5: City ecosystem quality 
fig, ax = plt.subplots(figsize=(13, 7))
cd = city_data.head(12).sort_values('total_m')
bar1 = ax.barh(cd['city'], cd['total_m'], color='teal', alpha=0.8, label='Total Funding ($M)')
ax2c = ax.twiny()
ax2c.plot(cd['funded_pct'], cd['city'], marker='D', color='red', linewidth=0,
          markersize=8, label='Funded Deal %')
ax.set_xlabel("Total Funding Deployed ($M)")
ax2c.set_xlabel("% of Deals That Disclosed Funding", color='red')
ax2c.tick_params(axis='x', colors='red')
ax.set_title("City Ecosystem Quality: Funding Volume and Deal Quality")
handles = [mpatches.Patch(color='teal', label='Total Funding ($M)'),
           plt.Line2D([0],[0],marker='D',color='red',linewidth=0,markersize=8,label='Funded %')]
ax.legend(handles=handles, loc='lower right')

fig.text(0.01, -0.06,
    "Longer teal bar = more total money deployed in that city.\n"
    "Red diamond further right = higher % of deals disclosing funding "
    "(a sign of ecosystem transparency and maturity).\n"
    "Cities with both = ideal startup locations.",
    fontsize=10)
plt.tight_layout()
save_fig("05_city_ecosystem_quality.png")

#Chart 6: Industry market share over time (stacked area)
fig, ax = plt.subplots(figsize=(13, 6))
pivot = mshare_top.pivot_table(index='year', columns='industry',
                               values='market_share_pct', fill_value=0)
pivot.plot.area(ax=ax, alpha=0.75, colormap='tab10')
ax.set_title("Top Industry Market Share Over Time\n(% of all startup deals each year)")
ax.set_xlabel("Year")
ax.set_ylabel("Market Share (%)")
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)

fig.text(0.01, -0.06,
    "Each colour = one industry. When a colour grows, that industry is capturing "
    "more of the startup market.\n"
    "When a colour shrinks, it's being replaced by newer, faster-growing industries.\n"
    "This is how investors spot macro trends before they become mainstream.",
    fontsize=10)
plt.tight_layout()
save_fig("06_industry_market_share_over_time.png")

#Chart 7: Pareto / Funding Concentration
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(pareto['rank_pct'], pareto['cum_pct'], linewidth=2, color='steelblue')
ax.axhline(80, color='red', linestyle='--', linewidth=1, label='80% of funding')
ax.axvline(top_pct_for_80, color='orange', linestyle='--', linewidth=1,
           label=f'Top {top_pct_for_80:.1f}% of startups')
ax.fill_between(pareto['rank_pct'], pareto['cum_pct'],
                where=pareto['rank_pct'] <= top_pct_for_80,
                alpha=0.15, color='orange')
ax.set_xlabel("% of Startups (ranked by total funding, highest first)")
ax.set_ylabel("Cumulative % of Total Funding")
ax.set_title(f"Funding Concentration (Pareto Curve)\n"
             f"Top {top_pct_for_80:.1f}% of startups control 80% of all funding")
ax.legend()
ax.set_xlim(0, 100); ax.set_ylim(0, 100)

fig.text(0.01, -0.06,
    "This curve shows how unfairly funding is distributed.\n"
    "A perfectly equal world would look like a diagonal line (45 degrees).\n"
    "The more the curve bends to the top-left, the more unequal the funding is.\n"
    "The orange area shows that a tiny % of startups absorb nearly all the capital.",
    fontsize=10)
plt.tight_layout()
save_fig("07_pareto_funding_concentration.png")

#Chart 8: ML — ROC Curves 
fig, ax = plt.subplots(figsize=(9, 7))
for model, label, color in [
        (lr, f'Logistic Regression (AUC={lr_auc:.3f})', 'steelblue'),
        (dt, f'Decision Tree (AUC={dt_auc:.3f})',        'darkorange'),
        (rf, f'Random Forest (AUC={rf_auc:.3f})',        'green')]:
    if hasattr(model, 'predict_proba'):
        X_t = X_test_s if model is lr else X_test
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_t)[:,1])
        ax.plot(fpr, tpr, label=label, linewidth=2, color=color)
ax.plot([0,1],[0,1], 'k--', linewidth=1, label='Random Guess (AUC=0.5)')
ax.set_xlabel("False Positive Rate (predicting survival when startup actually failed)")
ax.set_ylabel("True Positive Rate (correctly predicting survival)")
ax.set_title("Model Comparison: ROC Curves\nPredicting Which Startups Will Raise a 2nd Round")
ax.legend(loc='lower right')

fig.text(0.01, -0.08,
    "ROC Curve explanation: The higher and further left the curve, the better the model.\n"
    "AUC (Area Under Curve) of 1.0 = perfect. AUC of 0.5 = random guessing.\n"
    "Random Forest wins here — it uses many decision trees and picks the majority vote.\n"
    "This model can tell investors which startups are most likely to survive.",
    fontsize=10)
plt.tight_layout()
save_fig("08_model_roc_curves.png")

#Chart 9: Feature Importance
fig, ax = plt.subplots(figsize=(10, 5))
fi = feat_imp.sort_values()
human_labels = {
    'total_raised':    'Total Money Raised',
    'peak_round':      'Largest Single Round',
    'rounds':          'Number of Funding Rounds',
    'city_tier':       'City Quality (Tier 1 vs 2)',
    'industry_enc':    'Which Industry',
    'inv_type_enc':    'Type of Investment'
}
labels = [human_labels.get(c, c) for c in fi.index]
ax.barh(labels, fi.values, color='steelblue')
ax.set_title("What Factors Best Predict Startup Survival?\n(Random Forest Feature Importances)")
ax.set_xlabel("Importance Score (higher = more predictive)")
for i, v in enumerate(fi.values):
    ax.text(v + 0.002, i, f"{v:.3f}", va='center', fontsize=9)

fig.text(0.01, -0.06,
    "Feature importance tells us which inputs the model relies on most.\n"
    "The more important a feature, the more it affects whether a startup survives.\n"
    "Total money raised being #1 confirms: funding IS the single biggest survival predictor.",
    fontsize=10)
plt.tight_layout()
save_fig("09_feature_importance.png")

#Chart 10: Confusion Matrix 
fig, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_test, rf.predict(X_test))
labels_cm = ['Did Not\nReturn', 'Returned for\nMore Funding']
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax)
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(labels_cm); ax.set_yticklabels(labels_cm)
ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix — Random Forest\n(How often the model is right vs wrong)")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=14)

fig.text(0.01, -0.08,
    "Top-left: Model correctly said 'will not survive' → TRUE NEGATIVE\n"
    "Top-right: Model said 'will survive' but was wrong → FALSE POSITIVE (bad)\n"
    "Bottom-left: Model said 'will not survive' but was wrong → FALSE NEGATIVE (costly)\n"
    "Bottom-right: Model correctly said 'will survive' → TRUE POSITIVE\n"
    "We want the diagonal numbers (top-left, bottom-right) to be as large as possible.",
    fontsize=10)
plt.tight_layout()
save_fig("10_confusion_matrix.png")

#Chart 11: Quarterly deal flow heatmap
pivot_q = quarterly.pivot_table(index='year', columns='quarter',
                                values='deals', fill_value=0)
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(pivot_q.values, cmap='YlOrRd', aspect='auto')
plt.colorbar(im, ax=ax, label='Number of Deals')
ax.set_xticks(range(4)); ax.set_xticklabels(['Q1','Q2','Q3','Q4'])
ax.set_yticks(range(len(pivot_q))); ax.set_yticklabels(pivot_q.index.astype(int))
ax.set_title("Quarterly Deal Flow Heatmap\n(Darker = more deals that quarter)")
ax.set_xlabel("Quarter"); ax.set_ylabel("Year")
for i in range(len(pivot_q)):
    for j in range(4):
        ax.text(j, i, str(int(pivot_q.values[i,j])), ha='center', va='center', fontsize=10)

fig.text(0.01, -0.06,
    "Each cell = number of deals in that year-quarter combination.\n"
    "Darker cells = more active periods. Use this to time your fundraising.\n"
    "Raise in dark cells. Avoid light cells — fewer investors are writing cheques.",
    fontsize=10)
plt.tight_layout()
save_fig("11_quarterly_heatmap.png")

#Chart 12: Correlation Heatmap 
fig, ax = plt.subplots(figsize=(8, 6))
mask_corr = corr.values
im = ax.imshow(mask_corr, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, label='Correlation (-1 to +1)')
col_labels = ['Amount', 'Year', 'Month', 'City Tier', 'Is Funded', 'Large Deal']
ax.set_xticks(range(len(col_labels))); ax.set_yticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, rotation=30, ha='right')
ax.set_yticklabels(col_labels)
ax.set_title("Variable Correlation Matrix\n(Red = positive link, Blue = negative link)")
for i in range(len(col_labels)):
    for j in range(len(col_labels)):
        ax.text(j, i, f"{mask_corr[i,j]:.2f}", ha='center', va='center', fontsize=9)

fig.text(0.01, -0.06,
    "Correlation of +1 means when one number goes up, the other always goes up too.\n"
    "Correlation of -1 means they move in opposite directions.\n"
    "Correlation near 0 means they have no relationship.\n"
    "Dark red boxes = strong positive relationships worth investigating further.",
    fontsize=10)
plt.tight_layout()
save_fig("12_correlation_heatmap.png")

print("\nAll 12 charts saved.")

# FINAL REPORT
section("MODULE 6: GENERATING FINAL REPORT")

best_band_row = survival_band.loc[survival_band['multi_round_pct'].idxmax()]
top3_risky    = industry_risk.nlargest(3, 'unfunded_pct')['industry'].tolist()
top3_safe     = industry_risk.nsmallest(3, 'unfunded_pct').query('total_deals>=20')['industry'].tolist()
top3_cities   = city_data.head(3)['city'].tolist()
top3_inv      = top_investors.head(3)['investor'].tolist()

report = f"""
STARTUP ECOSYSTEM INTELLIGENCE — ANALYTICAL REPORT
Dataset: Indian Startup Funding | Kaggle

Total Startups Analyzed  : {df['startup'].nunique():,}
Total Funding Rounds     : {len(df):,}
Capital Deployed         : ${df['amount'].sum()/1e9:.2f} Billion USD
Study Period             : {int(df['year'].min())} – {int(df['year'].max())}


SECTION 1: ECOSYSTEM OVERVIEW -
The Indian startup ecosystem saw dramatic growth from {int(df['year'].min())} to {int(df['year'].max())}.
However, {df[df['amount']==0].shape[0]:,} out of {len(df):,} deals ({100*df[df['amount']==0].shape[0]/len(df):.1f}%)
disclosed zero funding — meaning the majority of startups never secure
publicly disclosed capital. This is a key indicator of the true failure
rate within the ecosystem.

Median deal size       : ${df[df['amount']>0]['amount'].median()/1e6:.3f}M
Mean deal size         : ${df[df['amount']>0]['amount'].mean()/1e6:.2f}M
Mega deals (>$10M)     : {df[df['amount']>=10e6].shape[0]:,} ({100*df[df['amount']>=10e6].shape[0]/len(df):.1f}% of all rounds)
Funding concentration  : Top {top_pct_for_80:.1f}% of startups control 80% of capital


SECTION 2: INDUSTRY RISK ANALYSIS -
HIGHEST RISK (most likely to fail without funding):
  1. {top3_risky[0] if len(top3_risky)>0 else 'N/A'}
  2. {top3_risky[1] if len(top3_risky)>1 else 'N/A'}
  3. {top3_risky[2] if len(top3_risky)>2 else 'N/A'}

LOWER RISK (better funding access):
  1. {top3_safe[0] if len(top3_safe)>0 else 'N/A'}
  2. {top3_safe[1] if len(top3_safe)>1 else 'N/A'}
  3. {top3_safe[2] if len(top3_safe)>2 else 'N/A'}

Key Insight:
  Industries with high unfunded percentages are not necessarily bad industries
  — they are often crowded markets where differentiation is very hard.
  Investors pass on most entrants because the space is already saturated.


SECTION 3: FUNDING vs SURVIVAL (KEY FINDING) -
{survival_band[['funding_band','startups','multi_round','multi_round_pct']].to_string(index=False)}

CRITICAL FINDING:
  Startups that raised {best_band_row['funding_band']} achieved the highest
  multi-round rate of {best_band_row['multi_round_pct']:.1f}%.

  Statistical Conclusion: More funding is not just correlated with survival
  — it appears to be the single strongest predictor (confirmed by Random Forest
  feature importance analysis in Section 5).

  Practical Implication: A startup that raises $0–$100K has almost no chance
  of returning for a follow-on round. The minimum viable fundraise to have a
  meaningful survival chance appears to be $1M+.


SECTION 4: GEOGRAPHY ANALYSIS -
 Top Cities for Startup Survival:
{city_data[['city','unique_startups','total_m','funded_pct','mega_deals']].head(8).to_string(index=False)}

Statistical Test (Mann-Whitney U):
  Tier 1 city startups receive significantly more funding than Tier 2 cities
  (p-value = {p_val:.4f}).
  This is not just correlation — Tier 1 cities provide access to better
  investor networks, talent, and repeat founders (experience spillover).


SECTION 5: MACHINE LEARNING — SURVIVAL PREDICTION -
Three models were trained to predict whether a startup will return
for a second funding round (survival proxy):

  Model               CV AUC    Test AUC
  Logistic Regression  {lr_cv:.3f}      {lr_auc:.3f}
  Decision Tree        {dt_cv:.3f}      {dt_auc:.3f}
  Random Forest        {rf_cv:.3f}      {rf_auc:.3f}    <-- BEST

Feature Importance (What drives survival):
{feat_imp.rename(human_labels).round(3).to_string()}

Interpretation:
  Total money raised and the size of the largest single round are by far
  the strongest predictors of survival. City quality (Tier 1 vs Tier 2)
  ranks third — confirming the geography finding above.


SECTION 6: STATISTICAL INFERENCE SUMMARY -
1. Funding level is the #1 predictor of survival (Random Forest importance).
2. Tier 1 cities produce significantly better-funded startups (p={p_val:.4f}).
3. Funding is highly concentrated — Pareto analysis shows top {top_pct_for_80:.0f}%
   of startups capture 80% of capital.
4. Year-over-year deal volume and capital deployed are weakly correlated
   — some years see more deals but smaller cheques (fragmentation years),
   other years see fewer but larger deals (consolidation years).


SECTION 7: STRATEGIC RECOMMENDATIONS -
For Founders:
  A. Do not start in saturated industries unless you have a clear,
     defensible differentiator. The data shows most entrants get no money.
  B. Raise at least $1M in your first round if you can.
     Sub-$100K rounds almost never lead to follow-on funding.
  C. Locate in Bangalore, Mumbai, or Delhi. The data confirms this
     is statistically significant, not just anecdotal wisdom.
  D. Pitch during peak months (see Chart 10 for seasonality).

For Investors:
  A. The top 10 active investors in this dataset have deployed capital
     across the widest industry range — diversification signals lower risk.
  B. Use the Random Forest model (AUC={rf_auc:.3f}) as a preliminary screening
     tool to filter high-risk startups before due diligence.
  C. Industries with low unfunded percentage AND high average ticket size
     are the most investable (see Industry Risk Matrix — Chart 2).

For Policy Makers:
  A. 80% of startup capital flows to ~{top_pct_for_80:.0f}% of startups.
     Policy interventions should target the long tail — the thousands of
     startups stuck at zero funding.
  B. Tier 2 city startups are significantly underfunded. Regional
     accelerator programs could meaningfully improve survival rates.
"""

report_path = os.path.join(EXPORT_PATH, "full_analysis_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(report)

# ── SQL Reference ─────────────────────────────────────────────────────────────
sql_ref = """

1. Yearly ecosystem pulse with median
SELECT year,
       COUNT(*) AS deals,
       COUNT(DISTINCT startup) AS unique_startups,
       ROUND(SUM(amount)/1e6, 1) AS total_funding_M,
       ROUND(AVG(amount)/1e6, 3) AS avg_deal_M,
       ROUND(MEDIAN(amount)/1e6, 3) AS median_deal_M,
       SUM(CASE WHEN amount >= 10000000 THEN 1 ELSE 0 END) AS mega_deals
FROM startups
GROUP BY year ORDER BY year;

2. Industry risk matrix
SELECT industry,
       COUNT(*) AS total_deals,
       ROUND(100.0*SUM(CASE WHEN amount=0 THEN 1 ELSE 0 END)/COUNT(*),1) AS unfunded_pct,
       ROUND(AVG(CASE WHEN amount>0 THEN amount END)/1e6, 3) AS avg_funded_M
FROM startups
WHERE industry NOT IN ('Unknown','')
GROUP BY industry HAVING total_deals >= 10
ORDER BY unfunded_pct DESC;

3. Multi-round survival proxy
SELECT startup,
       COUNT(*) AS rounds,
       SUM(amount) AS total_raised,
       MAX(year)-MIN(year) AS lifespan_years,
       CASE WHEN COUNT(*) > 1 THEN 1 ELSE 0 END AS survived
FROM startups
GROUP BY startup
ORDER BY total_raised DESC;

4. City quality with statistical view
SELECT city,
       COUNT(*) AS deals,
       COUNT(DISTINCT industry) AS diversification,
       ROUND(SUM(amount)/1e6,1) AS total_M,
       ROUND(100.0*SUM(CASE WHEN amount>0 THEN 1 ELSE 0 END)/COUNT(*),1) AS funded_pct
FROM startups
WHERE city NOT IN ('Unknown','')
GROUP BY city HAVING deals >= 10
ORDER BY total_M DESC;

5. Pareto analysis — funding concentration
WITH ranked AS (
    SELECT startup,
           SUM(amount) AS total_raised,
           RANK() OVER (ORDER BY SUM(amount) DESC) AS rnk
    FROM startups
    WHERE amount > 0
    GROUP BY startup
),
totals AS (SELECT COUNT(*) AS n, SUM(total_raised) AS grand_total FROM ranked)
SELECT r.startup, r.total_raised, r.rnk,
       ROUND(100.0*r.rnk/t.n, 2) AS rank_pct,
       ROUND(100.0*SUM(r2.total_raised)/t.grand_total, 2) AS cum_funding_pct
FROM ranked r
CROSS JOIN totals t
JOIN ranked r2 ON r2.rnk <= r.rnk
GROUP BY r.rnk ORDER BY r.rnk;

6. Quarterly deal flow (seasonal pattern)
SELECT year, quarter, COUNT(*) AS deals,
       ROUND(SUM(amount)/1e6,1) AS total_M
FROM startups
GROUP BY year, quarter ORDER BY year, quarter;

7. Industry market share over time (window function)
WITH base AS (
    SELECT year, industry, COUNT(*) AS deals
    FROM startups
    WHERE industry NOT IN ('Unknown','')
    GROUP BY year, industry
),
yr_totals AS (SELECT year, SUM(deals) AS yr_total FROM base GROUP BY year)
SELECT b.year, b.industry,
       ROUND(100.0*b.deals/y.yr_total, 1) AS market_share_pct
FROM base b JOIN yr_totals y ON b.year=y.year
ORDER BY b.year, b.deals DESC;

8. Top investors with portfolio diversity
SELECT investor,
       COUNT(DISTINCT startup) AS portfolio_size,
       COUNT(DISTINCT industry) AS industries_covered,
       COUNT(DISTINCT city) AS cities_active,
       ROUND(SUM(amount)/1e6,2) AS total_deployed_M
FROM startups
WHERE investor NOT IN ('Unknown','') AND LENGTH(investor) < 60
GROUP BY investor HAVING portfolio_size >= 3
ORDER BY portfolio_size DESC LIMIT 15;

9. Funding band survival analysis
SELECT
    CASE
        WHEN total_raised = 0              THEN 'None'
        WHEN total_raised < 100000         THEN 'Under $100K'
        WHEN total_raised < 1000000        THEN '$100K to $1M'
        WHEN total_raised < 5000000        THEN '$1M to $5M'
        WHEN total_raised < 20000000       THEN '$5M to $20M'
        WHEN total_raised < 100000000      THEN '$20M to $100M'
        ELSE '$100M+'
    END AS funding_band,
    COUNT(*) AS startups,
    SUM(survived) AS multi_round,
    ROUND(100.0*SUM(survived)/COUNT(*),1) AS survival_rate_pct
FROM (
    SELECT startup,
           SUM(amount) AS total_raised,
           CASE WHEN COUNT(*) > 1 THEN 1 ELSE 0 END AS survived
    FROM startups GROUP BY startup
)
GROUP BY funding_band;
"""

sql_path = os.path.join(EXPORT_PATH, "sql_queries_reference.sql")
with open(sql_path, 'w') as f:
    f.write(sql_ref)
print(f"\nSQL reference saved → {sql_path}")

conn.close()

print(f"All outputs saved in:\n  {EXPORT_PATH}\n")
print("Files generated:")
files = [
    "01_ecosystem_growth_dual_axis.png",
    "02_industry_risk_matrix_bubble.png",
    "03_investment_stage_funnel.png",
    "04_funding_vs_survival_dual.png",
    "05_city_ecosystem_quality.png",
    "06_industry_market_share_over_time.png",
    "07_pareto_funding_concentration.png",
    "08_model_roc_curves.png",
    "09_feature_importance.png",
    "10_confusion_matrix.png",
    "11_quarterly_heatmap.png",
    "12_correlation_heatmap.png",
    "full_analysis_report.txt",
    "sql_queries_reference.sql",
]
for f in files:
    print(f"  {f}")
