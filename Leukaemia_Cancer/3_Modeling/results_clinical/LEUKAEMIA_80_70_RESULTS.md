# LEUKAEMIA CANCER — CLINICAL-ONLY MODELING RESULTS

## Target: Sensitivity >= 80% + Specificity >= 70%

### Results at 80/70 Operating Point

| Window | Model    | Sensitivity | Specificity | Threshold | 80/70 Hit? |
|--------|----------|-------------|-------------|-----------|------------|
| 3mo    | CatBoost | 80.5%       | 78.2%       | 0.380     | YES        |
| 3mo    | Ensemble | 80.7%       | 77.9%       | 0.368     | YES        |
| 3mo    | LightGBM | 80.5%       | 77.7%       | 0.386     | YES        |
| 3mo    | XGBoost  | 80.7%       | 77.6%       | 0.362     | YES        |
| 6mo    | CatBoost | 80.4%       | 73.8%       | 0.356     | YES        |
| 6mo    | Ensemble | 80.4%       | 73.2%       | 0.356     | YES        |
| 6mo    | XGBoost  | 80.4%       | 72.6%       | 0.358     | YES        |
| 6mo    | LightGBM | 80.1%       | 70.2%       | 0.250     | YES        |
| 12mo   | XGBoost  | 78.7%       | 69.3%       | —         | NO (2% gap)|
| 12mo   | LightGBM | 77.4%       | 70.7%       | —         | NO         |
| 12mo   | CatBoost | 76.2%       | 70.0%       | —         | NO         |
| 12mo   | Ensemble | 77.7%       | 70.1%       | —         | NO         |

### AUC-ROC Summary

| Window | XGBoost | LightGBM | CatBoost | Ensemble |
|--------|---------|----------|----------|----------|
| 3mo    | 0.875   | 0.874    | 0.869    | 0.875    |
| 6mo    | 0.843   | 0.830    | 0.840    | 0.843    |
| 12mo   | 0.824   | 0.816    | 0.814    | 0.824    |

### Configuration
- **Features**: Clinical-only (~425 features after filtering)
- **Feature selection**: Top 225 by XGBoost importance + correlation filter
- **Seed**: 44
- **Split**: 75% train / 10% val / 15% test (stratified)
- **Optuna trials**: 75 per model
- **Models**: XGBoost, LightGBM, CatBoost + Ensemble (top 2 weighted)
- **Instance**: cts-ai-dev-cpu-01 (n2d-standard-16, 16 vCPUs)

### Key Findings
- **3mo and 6mo achieve 80/70 target across all models**
- Best 3mo: CatBoost — 80.5% sensitivity, 78.2% specificity
- Best 6mo: CatBoost — 80.4% sensitivity, 73.8% specificity
- 12mo is ~2% short on both sensitivity and specificity — text+embedding features may close the gap
- Leukaemia outperforms Ovarian Cancer (which could not hit 80/70)

### Patients
| Window | Total | Positive | Negative | Ratio |
|--------|-------|----------|----------|-------|
| 3mo    | 14,653| 2,495    | 12,158   | 4.9:1 |
| 6mo    | 14,494| 2,278    | 12,216   | 5.4:1 |
| 12mo   | 14,359| 2,126    | 12,233   | 5.8:1 |

---
Generated: 2026-03-31
Pipeline: Leukaemia Cancer clinical-only (seed 44)
Run on: GCloud cts-ai-dev-cpu-01 (n2d-standard-16)
