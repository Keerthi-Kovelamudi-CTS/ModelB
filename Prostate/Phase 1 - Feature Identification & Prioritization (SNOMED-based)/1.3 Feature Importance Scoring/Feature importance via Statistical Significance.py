"""
FEATURE IMPORTANCE SCORING (PATH A)
STATISTICAL ANALYSIS - 12 MONTH WINDOW

Requirements:
- Chi-Squared Test
- Odds Ratio (OR) with 95% CI
- Information Gain
- Effect sizes (Cohen's d)
- Prevalence difference
- FDR correction (Benjamini-Hochberg)

Output: Ranked list of top 250 discriminative features
"""

import pandas as pd
import numpy as np
from scipy.special import erf
from math import sqrt
from scipy.stats import t as t_dist, f as f_dist, chi2_contingency
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import warnings
import os

warnings.filterwarnings('ignore')

# ========================================================================
# DIAGNOSTIC EXCLUSIONS (Post-diagnostic codes to exclude)
# ========================================================================

DIAGNOSTIC_EXCLUSIONS = [
    # Referrals
    'fast track referral', '2 week rule', 'urgent referral', 
    'cancer referral', 'referred to urologist', 'urology referral',
    'referral to urologist',
    
    # Suspected/Raised (part of workup)
    'suspected cancer', 'suspected urological', 'suspected prostate',
    'raised psa', 'raised prostate', 'psa abnormal', 
    'prostate specific antigen abnormal',
    
    # Clinic visits (diagnostic pathway)
    'seen in urology', 'seen in oncology', 'seen by urologist',
    'urology clinic', 'oncology clinic', 'prostate clinic',
    'seen in fast track', 'seen in prostate',
    
    # Biopsy/Diagnostic procedures
    'biopsy', 'transrectal', 'transperineal', 
    'echography of prostate', 'mri of pelvis', 'pet scan',
    'positron emission tomography', 'cystoscopy',
    
    # Cancer diagnosis codes
    'carcinoma', 'adenocarcinoma', 'malignant', 'neoplasm',
    '[rfc]', 'reason for care', 'cancer of the prostate',
    'carcinoma in situ', 'tumour marker', 'histology abnormal',
    
    # Cancer management/treatment
    'cancer safety', 'cancer care review', 'cancer diagnosis',
    'cancer information', 'cancer support', 'cancer multidisciplinary',
    'chemotherapy', 'radiotherapy', 'hormone therapy',
    'gonadorelin', 'hormone implant', 'hormone antagonist',
    'who performance status', 'multidisciplinary team',
    
    # PSA monitoring (post-diagnosis)
    'psa monitoring', 'psa monitored',
    
    # Surgical/Treatment procedures
    'prostatectomy', 'turbt', 'transurethral resection',
    'urethral catheter', 'catheterisation', 'indwelling catheter',
]

def is_diagnostic(term):
    """Check if a term matches diagnostic exclusion patterns."""
    if pd.isna(term):
        return False
    term_lower = str(term).lower()
    for pattern in DIAGNOSTIC_EXCLUSIONS: 
        if pattern in term_lower:
            return True
    return False

# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

def two_proportion_ztest(x1, n1, x2, n2) -> Tuple[float, float]:
    """
    Two-proportion Z-test for comparing prevalence between cohorts.
    Returns (z_statistic, p_value) for H0: p1 == p2.
    """
    if any(v is None for v in [x1, n1, x2, n2]) or min(n1, n2) <= 0:
        return (np.nan, np.nan)
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2) if (n1 + n2) > 0 else np.nan
    denom = sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2)) if p_pool not in [0, 1] else np.nan
    if denom == 0 or np.isnan(denom):
        return (np.nan, np.nan)
    z = (p1 - p2) / denom
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return (z, p)

def welch_t_from_summary(m1, s1, n1, m2, s2, n2) -> Tuple[float, float, float]:
    """
    Welch's t-test using summary statistics.
    Returns (t_statistic, degrees_of_freedom, p_value).
    """
    if any(v is None for v in [m1, s1, n1, m2, s2, n2]) or min(n1, n2) < 2:
        return (np.nan, np.nan, np.nan)
    s1sq = s1**2
    s2sq = s2**2
    se = sqrt(s1sq/n1 + s2sq/n2)
    if se == 0:
        return (np.nan, np.nan, np.nan)
    t = (m1 - m2) / se
    df_num = (s1sq/n1 + s2sq/n2)**2
    df_den = (s1sq**2) / (n1**2 * (n1 - 1)) + (s2sq**2) / (n2**2 * (n2 - 1))
    df = df_num / df_den if df_den != 0 else np.nan
    p = 2 * (1 - t_dist.cdf(abs(t), df)) if not np.isnan(df) else np.nan
    return (t, df, p)

def f_test_variances(s1, n1, s2, n2) -> Tuple[float, float]:
    """
    Two-sided F-test for comparing variances.
    Returns (F_statistic, p_value).
    """
    if any(v is None for v in [s1, n1, s2, n2]) or min(n1, n2) < 2:
        return (np.nan, np.nan)
    s1sq, s2sq = s1**2, s2**2
    if s1sq == 0 or s2sq == 0:
        return (np.nan, np.nan)
    if s1sq >= s2sq:
        F = s1sq / s2sq
        dfn, dfd = n1 - 1, n2 - 1
    else:
        F = s2sq / s1sq
        dfn, dfd = n2 - 1, n1 - 1
    p_one = 1 - f_dist.cdf(F, dfn, dfd)
    p = 2 * min(p_one, 1 - p_one)
    return (F, p)

def cohens_d_from_summary(m1, s1, n1, m2, s2, n2) -> float:
    """
    Cohen's d effect size using pooled standard deviation.
    """
    if any(v is None for v in [m1, s1, n1, m2, s2, n2]) or min(n1, n2) < 2:
        return np.nan
    sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
    sp = sqrt(sp2) if sp2 > 0 else np.nan
    if sp == 0 or np.isnan(sp):
        return np.nan
    return (m1 - m2) / sp

def fdr_bh(pvals: pd.Series, alpha=0.05) -> pd.Series:
    """
    Benjamini-Hochberg FDR correction for multiple testing.
    Returns q-values (adjusted p-values).
    """
    p = pvals.values.astype(float)
    n = np.sum(~np.isnan(p))
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(p) + 1)
    q = p * n / ranks
    q_sorted = np.minimum.accumulate(np.flip(np.sort(np.where(np.isnan(p), np.inf, q))))
    q_adj = np.empty_like(q)
    q_adj[order] = np.flip(q_sorted)
    q_adj = np.minimum(q_adj, 1.0)
    return pd.Series(q_adj, index=pvals.index)

def entropy(p):
    """Calculate binary entropy for information gain."""
    if p == 0 or p == 1 or np.isnan(p):
        return 0
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

def information_gain(p_pos, p_neg, n_pos, n_neg):
    """
    Calculate information gain for binary classification.
    Measures how much information a feature provides about cancer status.
    """
    if any(np.isnan([p_pos, p_neg, n_pos, n_neg])):
        return 0
    
    p_feature = (p_pos * n_pos + p_neg * n_neg) / (n_pos + n_neg)
    
    if p_feature == 0 or p_feature == 1:
        return 0
    
    # Prior entropy (before knowing feature)
    p_class = n_pos / (n_pos + n_neg)
    H_prior = entropy(p_class)
    
    # Posterior entropy (after knowing feature)
    if p_feature > 0:
        p_cancer_given_present = (p_pos * n_pos) / (p_pos * n_pos + p_neg * n_neg)
        H_present = entropy(p_cancer_given_present)
    else:
        H_present = 0
    
    if p_feature < 1:
        p_cancer_given_absent = ((1-p_pos) * n_pos) / ((1-p_pos) * n_pos + (1-p_neg) * n_neg)
        H_absent = entropy(p_cancer_given_absent)
    else:
        H_absent = 0
    
    H_posterior = p_feature * H_present + (1 - p_feature) * H_absent
    return max(0, H_prior - H_posterior)

# ========================================================================
# MAIN COMPARISON FUNCTION
# ========================================================================

def compare_populations_comprehensive(csv_a_dataframe, csv_b_dataframe,
                                     alpha: float = 0.05,
                                     median_diff_threshold: float = 1.0) -> pd.DataFrame:
    """
    Comprehensive feature importance analysis with all required metrics.
    
    Returns:
    - DataFrame with all metrics, filtered to significant features
    - Ranked by composite importance score
    """
    
    a = csv_a_dataframe.copy()
    b = csv_b_dataframe.copy()

    def coerce(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize and clean input dataframes."""
        out = df.copy()
        out.columns = out.columns.str.upper()
        
        # Handle SNOMED_ID column variations
        if 'SNOMED_C_T_CONCEPT_ID' in out.columns:
            out['SNOMED_ID'] = out['SNOMED_C_T_CONCEPT_ID']
        elif 'SNOMED_ID' not in out.columns:
            return None
        
        out["TERM"] = out["TERM"].astype(str).str.strip().str.strip('"')
        out["SNOMED_ID"] = out["SNOMED_ID"].astype(str).str.strip()
        
        for c in ["N_PATIENT_COUNT", "N_PATIENT_COUNT_TOTAL"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        
        for c in ["AVG_VALUE", "MEDIAN_VALUE", "STD_VALUE", "FREQ_VALUE", "MEDIAN_SNOMED_AGE"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        
        # Calculate percentage if not present
        if "PCT_IN_POPULATION" not in out.columns:
            out["PCT_IN_POPULATION"] = (out["N_PATIENT_COUNT"] / out["N_PATIENT_COUNT_TOTAL"]) * 100
        
        return out

    a = coerce(a)
    b = coerce(b)
    
    if a is None or b is None:
        return None
    
    # Rename columns to distinguish cohorts
    a = a.rename(columns=lambda c: f"A_{c}" if c not in ["TERM", "SNOMED_ID"] else c)
    b = b.rename(columns=lambda c: f"B_{c}" if c not in ["TERM", "SNOMED_ID"] else c)

    # Merge on TERM + SNOMED_ID
    df = a.merge(b, on=["TERM", "SNOMED_ID"], how="outer")
    
    # Fill missing counts with 0
    df["A_N_PATIENT_COUNT"] = df["A_N_PATIENT_COUNT"].fillna(0)
    df["B_N_PATIENT_COUNT"] = df["B_N_PATIENT_COUNT"].fillna(0)
    
    # ========================================================================
    # TEST 1: TWO-PROPORTION Z-TEST
    # ========================================================================
    
    x1 = df["A_N_PATIENT_COUNT"]
    n1 = df["A_N_PATIENT_COUNT_TOTAL"].replace(0, np.nan)
    x2 = df["B_N_PATIENT_COUNT"]
    n2 = df["B_N_PATIENT_COUNT_TOTAL"].replace(0, np.nan)

    zstats, p_prop = [], []
    for i in range(len(df)):
        z, p = two_proportion_ztest(x1.iloc[i], n1.iloc[i], x2.iloc[i], n2.iloc[i])
        zstats.append(z)
        p_prop.append(p)
    
    df["prop_z"] = zstats
    df["prop_p"] = p_prop
    df["prop_diff"] = (x1/n1 - x2/n2) * 100
    
    # ========================================================================
    # TEST 2: CHI-SQUARED TEST
    # ========================================================================
    
    chi2_stats, chi2_pvals = [], []
    for i in range(len(df)):
        a_count = df["A_N_PATIENT_COUNT"].iloc[i]
        b_count = df["A_N_PATIENT_COUNT_TOTAL"].iloc[i] - a_count
        c_count = df["B_N_PATIENT_COUNT"].iloc[i]
        d_count = df["B_N_PATIENT_COUNT_TOTAL"].iloc[i] - c_count
        
        contingency = np.array([[a_count, b_count], [c_count, d_count]])
        
        try:
            chi2, p, dof, expected = chi2_contingency(contingency)
            chi2_stats.append(chi2)
            chi2_pvals.append(p)
        except:
            chi2_stats.append(np.nan)
            chi2_pvals.append(np.nan)
    
    df["chi2_statistic"] = chi2_stats
    df["chi2_pvalue"] = chi2_pvals
    
    # ========================================================================
    # TEST 3: ODDS RATIO WITH 95% CI
    # ========================================================================
    
    or_values, or_ci_lower, or_ci_upper = [], [], []
    for i in range(len(df)):
        a_count = df["A_N_PATIENT_COUNT"].iloc[i]
        b_count = df["A_N_PATIENT_COUNT_TOTAL"].iloc[i] - a_count
        c_count = df["B_N_PATIENT_COUNT"].iloc[i]
        d_count = df["B_N_PATIENT_COUNT_TOTAL"].iloc[i] - c_count
        
        if b_count > 0 and c_count > 0:
            or_val = (a_count * d_count) / (b_count * c_count) if (b_count * c_count) > 0 else np.inf
            
            # 95% Confidence Interval
            if a_count > 0 and c_count > 0 and d_count > 0:
                log_or = np.log(or_val)
                se_log_or = sqrt(1/a_count + 1/b_count + 1/c_count + 1/d_count)
                ci_low = np.exp(log_or - 1.96 * se_log_or)
                ci_high = np.exp(log_or + 1.96 * se_log_or)
            else:
                ci_low, ci_high = np.nan, np.nan
        else:
            or_val, ci_low, ci_high = np.inf, np.nan, np.nan
        
        or_values.append(or_val)
        or_ci_lower.append(ci_low)
        or_ci_upper.append(ci_high)

    df["odds_ratio"] = or_values
    df["or_ci_lower"] = or_ci_lower
    df["or_ci_upper"] = or_ci_upper
    
    # ========================================================================
    # TEST 4: INFORMATION GAIN
    # ========================================================================
    
    df["information_gain"] = df.apply(
        lambda row: information_gain(
            row['A_PCT_IN_POPULATION'] / 100 if not pd.isna(row['A_PCT_IN_POPULATION']) else 0,
            row['B_PCT_IN_POPULATION'] / 100 if not pd.isna(row['B_PCT_IN_POPULATION']) else 0,
            row['A_N_PATIENT_COUNT_TOTAL'] if not pd.isna(row['A_N_PATIENT_COUNT_TOTAL']) else 1,
            row['B_N_PATIENT_COUNT_TOTAL'] if not pd.isna(row['B_N_PATIENT_COUNT_TOTAL']) else 1
        ),
        axis=1
    )
    
    # ========================================================================
    # TEST 5: WELCH'S T-TEST
    # ========================================================================
    
    tvals, dfs, p_means, d_eff = [], [], [], []
    for i in range(len(df)):
        t, df_w, p = welch_t_from_summary(
            df["A_AVG_VALUE"].iloc[i], df["A_STD_VALUE"].iloc[i], df["A_N_PATIENT_COUNT"].iloc[i],
            df["B_AVG_VALUE"].iloc[i], df["B_STD_VALUE"].iloc[i], df["B_N_PATIENT_COUNT"].iloc[i]
        )
        tvals.append(t)
        dfs.append(df_w)
        p_means.append(p)
        
        d = cohens_d_from_summary(
            df["A_AVG_VALUE"].iloc[i], df["A_STD_VALUE"].iloc[i], df["A_N_PATIENT_COUNT"].iloc[i],
            df["B_AVG_VALUE"].iloc[i], df["B_STD_VALUE"].iloc[i], df["B_N_PATIENT_COUNT"].iloc[i]
        )
        d_eff.append(d)
    
    df["mean_t"] = tvals
    df["mean_df"] = dfs
    df["mean_p"] = p_means
    df["cohens_d"] = d_eff
    df["mean_diff"] = df["A_AVG_VALUE"] - df["B_AVG_VALUE"]
    
    # ========================================================================
    # TEST 6: F-TEST FOR VARIANCES
    # ========================================================================
    
    Fvals, p_vars = [], []
    for i in range(len(df)):
        F, p = f_test_variances(
            df["A_STD_VALUE"].iloc[i], df["A_N_PATIENT_COUNT"].iloc[i],
            df["B_STD_VALUE"].iloc[i], df["B_N_PATIENT_COUNT"].iloc[i]
        )
        Fvals.append(F)
        p_vars.append(p)
    
    df["var_F"] = Fvals
    df["var_p"] = p_vars
    df["std_diff"] = df["A_STD_VALUE"] - df["B_STD_VALUE"]
    
    # ========================================================================
    # TEST 7: MEDIAN & FREQUENCY COMPARISONS
    # ========================================================================
    
    df["median_diff"] = df["A_MEDIAN_VALUE"] - df["B_MEDIAN_VALUE"]
    df["median_flag"] = df["median_diff"].abs() >= median_diff_threshold
    df["freq_diff"] = df["A_FREQ_VALUE"] - df["B_FREQ_VALUE"]
    df["freq_flag"] = df["A_FREQ_VALUE"].ne(df["B_FREQ_VALUE"]) & df["A_FREQ_VALUE"].notna() & df["B_FREQ_VALUE"].notna()
    
    # ========================================================================
    # MULTIPLE TESTING CORRECTION
    # ========================================================================
    
    for col in ["prop_p", "mean_p", "var_p", "chi2_pvalue"]:
        if col in df.columns:
            df[f"{col}_q"] = fdr_bh(df[col], alpha=alpha)

    # Significance flags
    df["sig_prop"] = (df["prop_p_q"] <= alpha)
    df["sig_chi2"] = (df["chi2_pvalue_q"] <= alpha) if "chi2_pvalue_q" in df.columns else False
    df["sig_mean"] = (df["mean_p_q"] <= alpha)
    df["sig_var"] = (df["var_p_q"] <= alpha)
    
    # ========================================================================
    # COMPOSITE RANKING SCORE
    # ========================================================================
    
    scaler = MinMaxScaler()
    
    # 1. Odds Ratio score (handle inf)
    or_finite = df['odds_ratio'].replace([np.inf, -np.inf], np.nan)
    max_or = or_finite.max() if not or_finite.isna().all() else 10
    df['or_for_scoring'] = df['odds_ratio'].replace([np.inf, -np.inf], max_or * 1.5)
    df['or_score'] = scaler.fit_transform(df['or_for_scoring'].fillna(0).values.reshape(-1, 1))
    
    # 2. Prevalence difference score
    df['prev_diff_abs'] = df['prop_diff'].abs()
    df['prev_score'] = scaler.fit_transform(df['prev_diff_abs'].fillna(0).values.reshape(-1, 1))
    
    # 3. Chi-squared score
    chi2_finite = df['chi2_statistic'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df['chi2_score'] = scaler.fit_transform(chi2_finite.values.reshape(-1, 1))
    
    # 4. Information Gain score
    df['ig_score'] = scaler.fit_transform(df['information_gain'].fillna(0).values.reshape(-1, 1))
    
    # 5. Statistical significance score
    df['neg_log_p'] = -np.log10(df['prop_p_q'].replace(0, 1e-300))
    df['sig_score'] = scaler.fit_transform(df['neg_log_p'].fillna(0).values.reshape(-1, 1))
    
    # Weighted composite score
    df['composite_score'] = (
        0.35 * df['or_score'] +
        0.25 * df['prev_score'] +
        0.15 * df['chi2_score'] +
        0.10 * df['ig_score'] +
        0.15 * df['sig_score']
    )
    
    # ========================================================================
    # KEEP RELEVANT COLUMNS
    # ========================================================================
    
    keep = [
        "TERM", "SNOMED_ID",
        "A_N_PATIENT_COUNT", "A_N_PATIENT_COUNT_TOTAL", "A_PCT_IN_POPULATION",
        "A_AVG_VALUE", "A_MEDIAN_VALUE", "A_STD_VALUE", "A_FREQ_VALUE",
        "B_N_PATIENT_COUNT", "B_N_PATIENT_COUNT_TOTAL", "B_PCT_IN_POPULATION",
        "B_AVG_VALUE", "B_MEDIAN_VALUE", "B_STD_VALUE", "B_FREQ_VALUE",
        "prop_diff", "prop_z", "prop_p", "prop_p_q", "sig_prop",
        "chi2_statistic", "chi2_pvalue", "chi2_pvalue_q", "sig_chi2",
        "odds_ratio", "or_ci_lower", "or_ci_upper",
        "information_gain",
        "mean_diff", "cohens_d", "mean_t", "mean_df", "mean_p", "mean_p_q", "sig_mean",
        "std_diff", "var_F", "var_p", "var_p_q", "sig_var",
        "median_diff", "median_flag", "freq_diff", "freq_flag",
        "composite_score"
    ]
    
    keep = [c for c in keep if c in df.columns]
    df_out = df[keep].copy()

    # Filter to significant features
    mask = df_out["sig_prop"] | df_out["sig_mean"] | df_out["sig_var"] | df_out["median_flag"] | df_out["freq_flag"]
    df_out = df_out[mask].sort_values("composite_score", ascending=False)
    
    return df_out

# ========================================================================
# MAIN EXECUTION
# ========================================================================

if __name__ == "__main__":
    
    # FILE PATHS
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(base_dir, 'Data')
    
    POSITIVE_FILE = os.path.join(data_dir, 'top-snomed_prostateCancer_12mo_Window.csv')
    NEGATIVE_FILE = os.path.join(data_dir, 'top-snomed_no-prostate-cancer_12mo_Window.csv')
    
    # LOAD DATA
    try:
        positive = pd.read_csv(POSITIVE_FILE)
        negative = pd.read_csv(NEGATIVE_FILE)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print(f"Please ensure CSV files are in the '{data_dir}' directory")
        exit(1)
    
    # RUN COMPREHENSIVE ANALYSIS
    result = compare_populations_comprehensive(positive, negative, alpha=0.05, median_diff_threshold=1.0)
    
    if result is None:
        print("ERROR: Comparison failed")
        exit(1)
    
    # ========================================================================
    # APPLY DIAGNOSTIC EXCLUSION FILTER
    # ========================================================================
    
    total_compared = len(positive) + len(negative)
    result['is_diagnostic'] = result['TERM'].apply(is_diagnostic)
    result['is_artifact'] = result['is_diagnostic']
    result['exclusion_reason'] = ''
    result.loc[result['is_diagnostic'], 'exclusion_reason'] = 'Diagnostic/post-diagnostic code'
    
    # Create filtered dataset (exclude diagnostic codes)
    result_clinical = result[~result['is_diagnostic']].copy()
    
    # ========================================================================
    # EXPORT RESULTS
    # ========================================================================
    
    results_dir = os.path.join(base_dir, 'Results2_Analysis1.3')
    os.makedirs(results_dir, exist_ok=True)
    
    # Export ALL features (with artifact flag)
    output_file_all = os.path.join(results_dir, 'comprehensive_analysis_12mo_ALL.csv')
    result.to_csv(output_file_all, index=False)
    
    # Export CLINICAL features only
    output_file = os.path.join(results_dir, 'comprehensive_analysis_12mo.csv')
    result_clinical.to_csv(output_file, index=False)
    
    # Export TOP 250 features
    top_250 = result_clinical.head(250)
    top_250_file = os.path.join(results_dir, 'TOP_250_FEATURES_12mo.csv')
    top_250.to_csv(top_250_file, index=False)
    
    # Export excluded artifacts
    artifacts = result[result['is_artifact']].copy()
    artifacts_file = os.path.join(results_dir, 'EXCLUDED_ARTIFACTS_12mo.csv')
    artifacts.to_csv(artifacts_file, index=False)
    
    # Export summary statistics
    summary_stats = {
        'Time_Window': '12_months',
        'Total_Features_Compared': total_compared,
        'Significant_Features_All': len(result),
        'Artifacts_Excluded': result['is_artifact'].sum(),
        'Clinical_Features': len(result_clinical),
        'Sig_Prevalence_Ztest': result_clinical['sig_prop'].sum(),
        'Sig_Prevalence_Chi2': result_clinical['sig_chi2'].sum() if 'sig_chi2' in result_clinical.columns else 0,
        'Sig_Mean_Difference': result_clinical['sig_mean'].sum(),
        'Sig_Variance_Difference': result_clinical['sig_var'].sum(),
        'Mean_Odds_Ratio': result_clinical['odds_ratio'].mean(),
        'Median_Odds_Ratio': result_clinical['odds_ratio'].median(),
        'Mean_Information_Gain': result_clinical['information_gain'].mean()
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_file = os.path.join(results_dir, 'summary_12mo.csv')
    summary_df.to_csv(summary_file, index=False)
