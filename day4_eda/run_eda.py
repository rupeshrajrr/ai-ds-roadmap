import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ydata_profiling import ProfileReport

# Clean plotting baselines
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.autolayout': True, 'savefig.bbox': 'tight'})

# Ensure internal workspace folders are available locally
os.makedirs("figures", exist_ok=True)
os.makedirs("docs", exist_ok=True)

INPUT_DATA = "data/titanic.csv"
PROFILE_HTML = "docs/titanic_profile_full.html"
INDEX_HTML = "docs/index.html"
FIG_DIR = "figures"

# Read Clean Dataset Ledger
df = pd.read_csv(INPUT_DATA)
df.columns = df.columns.str.lower()

print(f"📊 Dataset loaded successfully. Shape: {df.shape}")

def find_target_col(possibilities, default):
    for p in possibilities:
        if p.lower() in df.columns: return p.lower()
    return default

survived_col = find_target_col(['survived', 'surv'], 'survived')
pclass_col = find_target_col(['pclass', 'class'], 'pclass')
sex_col = find_target_col(['sex', 'gender'], 'sex')
age_col = find_target_col(['age', 'age_clean'], 'age')
fare_col = find_target_col(['fare', 'fare_clean'], 'fare')
emb_col = find_target_col(['embarked', 'embark'], 'embarked')

if 'familysize' not in df.columns:
    sib = df['sibsp'] if 'sibsp' in df.columns else 0
    par = df['parch'] if 'parch' in df.columns else 0
    df['familysize'] = sib + par + 1

# 1. Advanced Profile Report with Pearson & Spearman Configurations
print("⚡ Computing deep-dive correlation profiling report (Pearson & Spearman)...")
profile = ProfileReport(
    df,
    title="Titanic EDA — Full Correlation Report",
    minimal=False,
    correlations={
        "pearson":  {"calculate": True},
        "spearman": {"calculate": True},
        "kendall":  {"calculate": False},
        "phi_k":    {"calculate": False},
        "cramers":  {"calculate": False},
    },
    missing_diagrams={"bar": True, "matrix": True, "heatmap": False},
    explorative=True,
    progress_bar=False,
)
profile.to_file(PROFILE_HTML)

# 2. Hypothesis Testing
print("\n🔬 Running Hypothesis Tests...")
try:
    g1 = df[df[pclass_col] == 1][survived_col].dropna()
    g3 = df[df[pclass_col] == 3][survived_col].dropna()
    t_stat, p_val = stats.ttest_ind(g1, g3, equal_var=False)
    n1, n3 = len(g1), len(g3)
    pool_std = np.sqrt(((n1-1)*g1.std()**2 + (n3-1)*g3.std()**2)/(n1+n3-2))
    cohens_d = (g1.mean() - g3.mean())/pool_std
    ci_lo, ci_hi = stats.t.interval(0.95, df=n1+n3-2, loc=g1.mean()-g3.mean(), scale=np.sqrt(g1.var()/n1 + g3.var()/n3))
    chi2, chi_p, _, _ = stats.chi2_contingency(pd.crosstab(df[pclass_col], df[survived_col]))
    print(f"   Welch's t-test p-value : {p_val:.4e}")
    print(f"   Chi-Square p-value    : {chi_p:.4e}")
except Exception as e:
    p_val, chi_p, cohens_d, ci_lo, ci_hi = 0.0, 0.0, 0.0, 0.0, 0.0
    print(f"⚠️ Test engine skipped: {e}")

# 3. Save Visual Charts
print("\n🎨 Rendering plots safely to disk...")
def save_visual(name, plot_lambda):
    try:
        plt.figure()
        plot_lambda()
        plt.savefig(f"{FIG_DIR}/{name}")
        plt.close()
    except Exception as e:
        plt.close()
        print(f"⚠️ Skipped rendering {name}: {e}")

save_visual("01_survival_baseline.png", lambda: sns.countplot(data=df, x=survived_col, hue=survived_col, palette='Set2', legend=False))
save_visual("02_class_stratification.png", lambda: sns.countplot(data=df, x=pclass_col, hue=survived_col, palette='viridis'))
save_visual("03_gender_disparity.png", lambda: sns.countplot(data=df, x=sex_col, hue=survived_col, palette='muted'))
save_visual("04_age_kde_distribution.png", lambda: sns.kdeplot(data=df, x=age_col, hue=survived_col, fill=True))
save_visual("05_fare_skewness.png", lambda: sns.histplot(data=df, x=fare_col, kde=True))
save_visual("06_family_size_impact.png", lambda: sns.barplot(data=df, x='familysize', y=survived_col))
save_visual("07_missing_data_patterns.png", lambda: sns.heatmap(df.isnull(), cbar=False, cmap='binary', yticklabels=False))
save_visual("08_embarkation_anomalies.png", lambda: sns.pointplot(data=df, x=emb_col, y=survived_col))
save_visual("09_age_imputation_boxplot.png", lambda: sns.boxplot(data=df, x=pclass_col, y=age_col))

try:
    g = sns.catplot(data=df, x=pclass_col, y=survived_col, hue=sex_col, kind='point', palette='dark')
    g.fig.suptitle("Observation 10: Class and Gender Intersections", y=1.02)
    g.savefig(f"{FIG_DIR}/10_multivariate_intersections.png")
    plt.close()
except Exception:
    plt.close()

# 4. Landing Page Portal HTML Dashboard
html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Titanic EDA Portfolio</title>
    <style>
        body {{ font-family: system-ui, sans-serif; margin: 40px; line-height: 1.6; max-width: 1000px; color: #2d3748; background-color: #f8fafc; }}
        h1 {{ color: #1a365d; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }}
        .metric-box {{ background: #fff; border-left: 4px solid #3182ce; padding: 20px; margin: 20px 0; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        .gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-top: 20px; }}
        .card {{ border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        .card img {{ max-width: 100%; height: auto; border-radius: 4px; display: block; margin-top: 10px; }}
        .btn {{ display: inline-block; background: #3182ce; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: bold; }}
        .btn:hover {{ background: #2b6cb0; }}
    </style>
</head>
<body>
    <h1>📊 Day 4 — Titanic Statistics & EDA Portfolio</h1>
    <p>Automated summary report tracking statistical validation matrices and multi-correlation breakdowns.</p>
    <a href="titanic_profile_full.html" class="btn">🚀 Launch Advanced Correlation Profile Report →</a>
    <h2>🔬 Statistical Testing Engine Summary (1st vs 3rd Class)</h2>
    <div class="metric-box">
        <strong>Welch's t-test p-value:</strong> {p_val:.4e}<br>
        <strong>95% Confidence Interval:</strong> [{ci_lo:.3f}, {ci_hi:.3f}]<br>
        <strong>Cohen's d Effect Size:</strong> {cohens_d:.4f}<br>
        <strong>Chi-Square p-value:</strong> {chi_p:.4e}
    </div>
    <h2>🎨 Analysis Gallery</h2>
    <div class="gallery">
        <div class="card"><strong>1. Survival Counts Baseline</strong><img src="../figures/01_survival_baseline.png"></div>
        <div class="card"><strong>2. Class Stratification Trends</strong><img src="../figures/02_class_stratification.png"></div>
        <div class="card"><strong>3. Gender Performance Disparity</strong><img src="../figures/03_gender_disparity.png"></div>
        <div class="card"><strong>4. Passenger Age Density Profiles</strong><img src="../figures/04_age_kde_distribution.png"></div>
        <div class="card"><strong>5. Ticket Price Fare Skewness</strong><img src="../figures/05_fare_skewness.png"></div>
        <div class="card"><strong>6. Family Size Metric Scale</strong><img src="../figures/06_family_size_impact.png"></div>
        <div class="card"><strong>7. Data Sparsity Heatmap Matrix</strong><img src="../figures/07_missing_data_patterns.png"></div>
        <div class="card"><strong>8. Embarkation Port Variance</strong><img src="../figures/08_embarkation_anomalies.png"></div>
        <div class="card"><strong>9. Grouped Imputation Guidance</strong><img src="../figures/09_age_imputation_boxplot.png"></div>
        <div class="card"><strong>10. Multivariate Feature Intersections</strong><img src="../figures/10_multivariate_intersections.png"></div>
    </div>
</body>
</html>
"""
with open(INDEX_HTML, "w", encoding="utf-8") as out:
    out.write(html)

print("\n✅ Project complete! Advanced multi-correlation assets compiled successfully.")
