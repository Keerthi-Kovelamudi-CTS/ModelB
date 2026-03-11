# RUN THIS — verify haematuria, LUTS, dysuria, raised WBC in windowed data
# Uses clinical windowed input (data/{window}/). Raised WBC from cleaned feature matrix when available.
# Usage: python 7_verification.py [--window 3mo|6mo|12mo|12mo_250k]  (default: 12mo_250k)
#        python 7_verification.py --compare   (run for 3mo, 6mo, 12mo and print comparison table)

import os
import argparse
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description='Verify haematuria, LUTS, dysuria, raised WBC in windowed clinical data.')
parser.add_argument('--window', choices=['3mo', '6mo', '12mo', '12mo_250k'], default='12mo_250k', help='Data window (default: 12mo_250k)')
parser.add_argument('--compare', action='store_true', help='Run for 3mo, 6mo, 12mo and print comparison table')
args = parser.parse_args()
WINDOW = args.window
COMPARE = args.compare

CLEANUP_DIR = os.path.join(SCRIPT_DIR, 'cleanupfeatures')


def get_clinical_path(win):
    data_dir = os.path.join(SCRIPT_DIR, 'data', win)
    if win == '12mo':
        return os.path.join(data_dir, 'FE_bladder_clinical_windowed.csv')
    if win == '12mo_250k':
        return os.path.join(data_dir, 'FE_bladder_clinical_windowed_12m_250k.csv')
    suffix = '3m' if win == '3mo' else '6m'
    return os.path.join(data_dir, f'FE_bladder_clinical_windowed_{suffix}.csv')


def get_matrix_path(win):
    return os.path.join(CLEANUP_DIR, win, f'bladder_feature_matrix_{win}_cleaned.csv')


def compute_metrics(win):
    """Compute verification metrics for one window. Returns dict or None if clinical file missing."""
    clinical_path = get_clinical_path(win)
    if not os.path.isfile(clinical_path):
        return None
    clinical = pd.read_csv(clinical_path)
    total_pts = clinical['PATIENT_GUID'].nunique()
    total_records = len(clinical)
    cancer_pids = set(clinical[clinical['LABEL'] == 1]['PATIENT_GUID'].unique())
    non_cancer_pids = set(clinical[clinical['LABEL'] == 0]['PATIENT_GUID'].unique())

    # Haematuria
    haem = clinical[clinical['CATEGORY'] == 'HAEMATURIA']
    haem_pids = set(haem['PATIENT_GUID'].unique()) if len(haem) > 0 else set()
    haem_pct = (len(haem_pids) / total_pts * 100) if total_pts else 0
    cancer_haem = len(haem_pids & cancer_pids)
    non_cancer_haem = len(haem_pids & non_cancer_pids)
    cancer_haem_pct = (cancer_haem / len(cancer_pids) * 100) if cancer_pids else 0
    non_cancer_haem_pct = (non_cancer_haem / len(non_cancer_pids) * 100) if non_cancer_pids else 0

    # LUTS
    luts = clinical[clinical['CATEGORY'] == 'LUTS']
    luts_pids = set(luts['PATIENT_GUID'].unique()) if len(luts) > 0 else set()
    luts_pct = (len(luts_pids) / total_pts * 100) if total_pts else 0
    cancer_luts_pct = (len(luts_pids & cancer_pids) / len(cancer_pids) * 100) if cancer_pids else 0
    non_cancer_luts_pct = (len(luts_pids & non_cancer_pids) / len(non_cancer_pids) * 100) if non_cancer_pids else 0

    # Dysuria (category or term)
    dysuria_cat = clinical[clinical['CATEGORY'] == 'DYSURIA']
    dysuria_term = clinical[clinical['TERM'].str.contains('dysuria', case=False, na=False)]
    dysuria_pids = set(dysuria_cat['PATIENT_GUID'].unique()) | set(dysuria_term['PATIENT_GUID'].unique()) if (len(dysuria_cat) > 0 or len(dysuria_term) > 0) else set()
    dysuria_pct = (len(dysuria_pids) / total_pts * 100) if total_pts else 0
    cancer_dysuria_pct = (len(dysuria_pids & cancer_pids) / len(cancer_pids) * 100) if cancer_pids else 0
    non_cancer_dysuria_pct = (len(dysuria_pids & non_cancer_pids) / len(non_cancer_pids) * 100) if non_cancer_pids else 0

    # Raised WBC from cleaned feature matrix (LAB_WBC_HIGH, last value > 11)
    raised_wbc_pct = raised_wbc_cancer_pct = raised_wbc_non_pct = None
    matrix_path = get_matrix_path(win)
    if os.path.isfile(matrix_path):
        try:
            df = pd.read_csv(matrix_path, usecols=['PATIENT_GUID', 'LABEL', 'LAB_WBC_HIGH'], low_memory=False)
            if 'LAB_WBC_HIGH' in df.columns:
                raised = df[df['LAB_WBC_HIGH'] == 1]
                raised_wbc_pct = (len(raised) / len(df) * 100) if len(df) else 0
                n_cancer = (df['LABEL'] == 1).sum()
                n_non = (df['LABEL'] == 0).sum()
                raised_wbc_cancer_pct = (raised['LABEL'].eq(1).sum() / n_cancer * 100) if n_cancer else 0
                raised_wbc_non_pct = (raised['LABEL'].eq(0).sum() / n_non * 100) if n_non else 0
        except Exception:
            pass

    return {
        'window': win,
        'total_pts': total_pts,
        'total_records': total_records,
        'haem_pct': haem_pct,
        'cancer_haem_pct': cancer_haem_pct,
        'non_cancer_haem_pct': non_cancer_haem_pct,
        'luts_pct': luts_pct,
        'cancer_luts_pct': cancer_luts_pct,
        'non_cancer_luts_pct': non_cancer_luts_pct,
        'dysuria_pct': dysuria_pct,
        'cancer_dysuria_pct': cancer_dysuria_pct,
        'non_cancer_dysuria_pct': non_cancer_dysuria_pct,
        'raised_wbc_pct': raised_wbc_pct,
        'raised_wbc_cancer_pct': raised_wbc_cancer_pct,
        'raised_wbc_non_pct': raised_wbc_non_pct,
        'clinical': clinical,
        'cancer_pids': cancer_pids,
        'non_cancer_pids': non_cancer_pids,
    }


def run_single_window(win):
    """Full detailed output for one window."""
    clinical_path = get_clinical_path(win)
    if not os.path.isfile(clinical_path):
        raise FileNotFoundError(f"Clinical windowed file not found: {clinical_path}. Run pipeline for --window {win} first.")
    clinical = pd.read_csv(clinical_path)
    print(f"Loaded: {clinical_path} (window={win})\n")

    print("=" * 70)
    print("HAEMATURIA PREVALENCE CHECK")
    print("=" * 70)
    total_pts = clinical['PATIENT_GUID'].nunique()
    print(f"Total patients: {total_pts:,}")
    print(f"Total records:  {len(clinical):,}")
    haem = clinical[clinical['CATEGORY'] == 'HAEMATURIA']
    haem_pids = set(haem['PATIENT_GUID'].unique()) if len(haem) > 0 else set()
    cancer_pids = set(clinical[clinical['LABEL'] == 1]['PATIENT_GUID'].unique())
    non_cancer_pids = set(clinical[clinical['LABEL'] == 0]['PATIENT_GUID'].unique())
    print(f"\nHaematuria records: {len(haem):,}")
    print(f"Haematuria patients: {len(haem_pids):,}")
    print(f"Haematuria prevalence: {len(haem_pids) / total_pts * 100:.2f}%")
    print(f"\nCancer patients WITH haematuria:    {len(haem_pids & cancer_pids):,} / {len(cancer_pids):,} ({len(haem_pids & cancer_pids)/len(cancer_pids)*100:.1f}%)")
    print(f"Non-cancer patients WITH haematuria: {len(haem_pids & non_cancer_pids):,} / {len(non_cancer_pids):,} ({len(haem_pids & non_cancer_pids)/len(non_cancer_pids)*100:.1f}%)")

    print(f"\n" + "=" * 70)
    print("LUTS PREVALENCE")
    print("=" * 70)
    luts = clinical[clinical['CATEGORY'] == 'LUTS']
    luts_pids = set(luts['PATIENT_GUID'].unique()) if len(luts) > 0 else set()
    print(f"LUTS patients: {len(luts_pids):,} ({len(luts_pids) / total_pts * 100:.2f}%)")
    print(f"Cancer with LUTS:     {len(luts_pids & cancer_pids):,} / {len(cancer_pids):,} ({len(luts_pids & cancer_pids)/len(cancer_pids)*100:.1f}%)")
    print(f"Non-cancer with LUTS: {len(luts_pids & non_cancer_pids):,} / {len(non_cancer_pids):,} ({len(luts_pids & non_cancer_pids)/len(non_cancer_pids)*100:.1f}%)")

    print(f"\n" + "=" * 70)
    print("DYSURIA PREVALENCE (category or term containing 'dysuria')")
    print("=" * 70)
    dysuria_cat = clinical[clinical['CATEGORY'] == 'DYSURIA']
    dysuria_term = clinical[clinical['TERM'].str.contains('dysuria', case=False, na=False)]
    dysuria_pids = set(dysuria_cat['PATIENT_GUID'].unique()) | (set(dysuria_term['PATIENT_GUID'].unique()) if len(dysuria_term) > 0 else set())
    if len(dysuria_cat) > 0 or len(dysuria_term) > 0:
        print(f"Dysuria patients: {len(dysuria_pids):,} ({len(dysuria_pids) / total_pts * 100:.2f}%)")
        print(f"Cancer with dysuria:     {len(dysuria_pids & cancer_pids):,} / {len(cancer_pids):,} ({len(dysuria_pids & cancer_pids)/len(cancer_pids)*100:.1f}%)")
        print(f"Non-cancer with dysuria: {len(dysuria_pids & non_cancer_pids):,} / {len(non_cancer_pids):,} ({len(dysuria_pids & non_cancer_pids)/len(non_cancer_pids)*100:.1f}%)")
    else:
        print("No dysuria records found in this window.")

    matrix_path = get_matrix_path(win)
    if os.path.isfile(matrix_path):
        try:
            df = pd.read_csv(matrix_path, usecols=['PATIENT_GUID', 'LABEL', 'LAB_WBC_HIGH'], low_memory=False)
            if 'LAB_WBC_HIGH' in df.columns:
                raised = df[df['LAB_WBC_HIGH'] == 1]
                print(f"\n" + "=" * 70)
                print("RAISED WHITE CELL COUNT (blood test, LAB_WBC_HIGH from feature matrix)")
                print("=" * 70)
                print(f"Patients with raised WBC: {len(raised):,} / {len(df):,} ({len(raised)/len(df)*100:.2f}%)")
                n_cancer = (df['LABEL'] == 1).sum()
                n_non = (df['LABEL'] == 0).sum()
                print(f"Cancer with raised WBC:     {(raised['LABEL']==1).sum():,} / {n_cancer:,} ({(raised['LABEL']==1).sum()/n_cancer*100:.1f}%)")
                print(f"Non-cancer with raised WBC: {(raised['LABEL']==0).sum():,} / {n_non:,} ({(raised['LABEL']==0).sum()/n_non*100:.1f}%)")
            else:
                print(f"\n(LAB_WBC_HIGH not in cleaned matrix for {win}; skip raised WBC)")
        except Exception as e:
            print(f"\n(Could not read raised WBC from matrix: {e})")
    else:
        print(f"\n(Cleaned matrix not found: {matrix_path}; run 5_feature_cleanup.py --window {win} for raised WBC stats)")

    print(f"\n" + "=" * 70)
    print("ALL CATEGORY PREVALENCE")
    print("=" * 70)
    cats = clinical.groupby('CATEGORY')['PATIENT_GUID'].nunique().sort_values(ascending=False)
    for cat, count in cats.items():
        pct = count / total_pts * 100
        cat_pids = set(clinical[clinical['CATEGORY'] == cat]['PATIENT_GUID'].unique())
        cancer_in_cat = len(cat_pids & cancer_pids)
        cancer_rate = cancer_in_cat / max(len(cat_pids), 1) * 100
        marker = " ← SIGNAL" if cancer_rate > 15 else ""
        print(f"  {cat:50s} {count:6,} pts ({pct:5.1f}%)  cancer rate: {cancer_rate:.1f}%{marker}")

    print(f"\n" + "=" * 70)
    print("EVENT_TYPE DISTRIBUTION")
    print("=" * 70)
    print(clinical['EVENT_TYPE'].value_counts())

    print(f"\n" + "=" * 70)
    print("TIME_WINDOW DISTRIBUTION")
    print("=" * 70)
    print(clinical['TIME_WINDOW'].value_counts())

    print(f"\n" + "=" * 70)
    print("TERMS CONTAINING 'haem' or 'blood' or 'urin' or 'dysuria'")
    print("=" * 70)
    for pattern in ['haem', 'blood.*urin', 'urin.*blood', 'hematuria', 'dysuria']:
        matched = clinical[clinical['TERM'].str.contains(pattern, case=False, na=False)]
        if len(matched) > 0:
            print(f"\n  '{pattern}': {matched['PATIENT_GUID'].nunique():,} patients, {len(matched):,} records")
            print(f"    Categories: {matched['CATEGORY'].value_counts().head(5).to_dict()}")


def run_compare():
    """Run for 3mo, 6mo, 12mo and print comparison table."""
    windows = ['3mo', '6mo', '12mo']
    results = {}
    for win in windows:
        m = compute_metrics(win)
        results[win] = m

    print("=" * 80)
    print("WINDOW COMPARISON: 3mo  |  6mo  |  12mo  (haematuria, LUTS, dysuria, raised WBC)")
    print("=" * 80)

    def fmt(v, decimals=1):
        if v is None:
            return "   N/A  "
        if isinstance(v, int):
            return f"{v:>7,}"
        return f"{v:7.{decimals}f}"

    def row(label, key_3mo, key_6mo, key_12mo, suffix=""):
        r3 = results.get('3mo')
        r6 = results.get('6mo')
        r12 = results.get('12mo')
        v3 = r3.get(key_3mo) if r3 else None
        v6 = r6.get(key_6mo) if r6 else None
        v12 = r12.get(key_12mo) if r12 else None
        print(f"  {label:48s}  3mo: {fmt(v3)}  6mo: {fmt(v6)}  12mo: {fmt(v12)}  {suffix}")

    for win in windows:
        if results.get(win) is None:
            print(f"  [{win}]: clinical file missing — run FE pipeline for --window {win} first.")
    print()

    row("Total patients", 'total_pts', 'total_pts', 'total_pts', "")
    row("Total records", 'total_records', 'total_records', 'total_records', "")
    print()
    row("Haematuria prevalence (%)", 'haem_pct', 'haem_pct', 'haem_pct', "%")
    row("  Cancer with haematuria (%)", 'cancer_haem_pct', 'cancer_haem_pct', 'cancer_haem_pct', "%")
    row("  Non-cancer with haematuria (%)", 'non_cancer_haem_pct', 'non_cancer_haem_pct', 'non_cancer_haem_pct', "%")
    print()
    row("LUTS prevalence (%)", 'luts_pct', 'luts_pct', 'luts_pct', "%")
    row("  Cancer with LUTS (%)", 'cancer_luts_pct', 'cancer_luts_pct', 'cancer_luts_pct', "%")
    row("  Non-cancer with LUTS (%)", 'non_cancer_luts_pct', 'non_cancer_luts_pct', 'non_cancer_luts_pct', "%")
    print()
    row("Dysuria prevalence (%)", 'dysuria_pct', 'dysuria_pct', 'dysuria_pct', "%")
    row("  Cancer with dysuria (%)", 'cancer_dysuria_pct', 'cancer_dysuria_pct', 'cancer_dysuria_pct', "%")
    row("  Non-cancer with dysuria (%)", 'non_cancer_dysuria_pct', 'non_cancer_dysuria_pct', 'non_cancer_dysuria_pct', "%")
    print()
    row("Raised WBC prevalence (%) [from cleaned matrix]", 'raised_wbc_pct', 'raised_wbc_pct', 'raised_wbc_pct', "%")
    row("  Cancer with raised WBC (%)", 'raised_wbc_cancer_pct', 'raised_wbc_cancer_pct', 'raised_wbc_cancer_pct', "%")
    row("  Non-cancer with raised WBC (%)", 'raised_wbc_non_pct', 'raised_wbc_non_pct', 'raised_wbc_non_pct', "%")

    print("=" * 80)
    out_dir = os.path.join(SCRIPT_DIR, 'verification_results')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'verification_comparison_3mo_6mo_12mo.txt')
    with open(out_file, 'w') as f:
        r3, r6, r12 = results.get('3mo') or {}, results.get('6mo') or {}, results.get('12mo') or {}
        f.write("WINDOW COMPARISON: 3mo | 6mo | 12mo\n")
        f.write("Total patients: 3mo=%s 6mo=%s 12mo=%s\n" % (r3.get('total_pts') or 'N/A', r6.get('total_pts') or 'N/A', r12.get('total_pts') or 'N/A'))
        f.write("Haematuria %%: 3mo=%s 6mo=%s 12mo=%s\n" % (fmt(r3.get('haem_pct')), fmt(r6.get('haem_pct')), fmt(r12.get('haem_pct'))))
        f.write("Dysuria %%: 3mo=%s 6mo=%s 12mo=%s\n" % (fmt(r3.get('dysuria_pct')), fmt(r6.get('dysuria_pct')), fmt(r12.get('dysuria_pct'))))
        f.write("Raised WBC %%: 3mo=%s 6mo=%s 12mo=%s\n" % (fmt(r3.get('raised_wbc_pct')), fmt(r6.get('raised_wbc_pct')), fmt(r12.get('raised_wbc_pct'))))
    print(f"Comparison saved to: {out_file}")


if COMPARE:
    run_compare()
else:
    run_single_window(WINDOW)
