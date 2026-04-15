# Melanoma Cancer — Modeling Results (Filtered)

## 1. Clinical-Only Results (Single Seed 44)

|                    | 3-month              | 6-month              | 12-month             |
|--------------------|----------------------|----------------------|----------------------|
| Best Model         | XGBoost              | XGBoost              | XGBoost              |
| Cancer / Non-cancer| 3,724 / 11,998 (~1:3)| 2,932 / 11,655 (~1:4)| 2,726 / 11,340 (~1:4)|
| Train / Val / Test | 11,791 / 1,572 / 2,359| 10,940 / 1,458 / 2,189| 10,549 / 1,406 / 2,110|
| Cancer in splits   | 2,793 / 372 / 559    | 2,199 / 293 / 440    | 2,044 / 272 / 409    |
| AUC                | 0.797                | 0.760                | 0.763                |
| Sensitivity        | 75.3%                | 70.2%                | 70.2%                |
| Specificity        | 69.1%                | 67.7%                | 67.3%                |
| PPV / NPV          | 43.0% / 90.0%       | 35.4% / 90.0%       | 34.0% / 90.4%       |

## 2. Text + Embeddings Results (5 Seeds)

*Pending — currently running on GCloud. Will be updated when complete.*

## 3. Noise Filtering

22 observation codes and 20 medication codes were removed before feature engineering to reduce noise from routine dermatology conditions unrelated to melanoma detection.

**Observation codes removed (22):**
- Routine dermatology: Pruritus ani, Pyoderma, Vesicular eczema, Lichen sclerosus, Dermatitis/eczemas NOS, Impetigo, Nail dystrophy, Onychomycosis, Scalp psoriasis, Varicose veins with eczema
- Warts: Viral wart, Wart treated with liquid nitrogen, Hand wart, Verruca plantaris
- Generic skin symptoms: Skin symptoms NOS, Skin inflammatory disorders NOS, Skin disease, Symptoms of integumentary tissue
- Infections: Infected toe, Blister of foot, Open wound of lower limb, Insect bites

**Medication codes removed (20):**
- Topical steroids for routine conditions: Hydrocortisone/Miconazole, Daktacort, Elocon, Synalar, Diprosalic, Betnovate, Crotamiton
- Topical antibiotics for routine infections: Fucidin H, Fucibet, Mupirocin, Fusidic acid
- Misc: Salicylic acid gel, Fludroxycortide tape, Aciclovir cream

**Impact:** Total 15,643 rows removed across all windows. XGBoost 3mo AUC improved from 0.786 (unfiltered) to 0.797 (filtered) — a +1.1% gain.

Original data is preserved in `data/`; filtered data in `data_filtered/`.

## 4. All Models — Clinical-Only

### 3mo (target: Sensitivity >= 75%)

| Model    | AUC   | Sens  | Spec  | PPV   | NPV   |
|----------|-------|-------|-------|-------|-------|
| XGBoost  | 0.797 | 75.3% | 69.1% | 43.0% | 90.0% |
| Ensemble | 0.797 | 75.1% | 69.0% | 42.9% | 89.9% |
| CatBoost | 0.792 | 75.5% | 67.7% | 42.1% | 89.9% |
| LightGBM | 0.784 | 75.1% | 64.9% | 40.0% | 89.4% |

### 6mo (target: Sensitivity >= 70%)

| Model    | AUC   | Sens  | Spec  | PPV   | NPV   |
|----------|-------|-------|-------|-------|-------|
| XGBoost  | 0.760 | 70.2% | 67.7% | 35.4% | 90.0% |
| Ensemble | 0.761 | 70.0% | 67.5% | 35.1% | 89.9% |
| CatBoost | 0.756 | 70.5% | 63.9% | 32.9% | 89.6% |
| LightGBM | 0.743 | 72.5% | 60.9% | 31.8% | 89.8% |

### 12mo (target: Sensitivity >= 70%)

| Model    | AUC   | Sens  | Spec  | PPV   | NPV   |
|----------|-------|-------|-------|-------|-------|
| Ensemble | 0.764 | 70.2% | 67.6% | 34.2% | 90.4% |
| XGBoost  | 0.763 | 70.2% | 67.3% | 34.0% | 90.4% |
| CatBoost | 0.762 | 70.9% | 66.4% | 33.7% | 90.5% |
| LightGBM | 0.743 | 73.3% | 59.1% | 30.2% | 90.2% |

## 5. Comparison: Unfiltered vs Filtered (3mo XGBoost)

| Metric      | Unfiltered | Filtered | Change |
|-------------|-----------|----------|--------|
| AUC         | 0.786     | 0.797    | +1.1%  |
| Sensitivity | ~73%      | 75.3%    | +2.3%  |
| Specificity | ~66%      | 69.1%    | +3.1%  |

## 6. Configuration

- Features: Clinical-only (~430 features after filtering generic utilisation)
- Feature selection: Top 225 by XGBoost importance + correlation filter
- Seed: 44 (clinical-only), Seeds 42-46 (5-seed, pending)
- Data split: 75% train / 10% val / 15% test (stratified)
- Optuna trials: 75 per model
- Models: XGBoost, LightGBM, CatBoost + Ensemble (weighted average of top 2)
- Class imbalance: Handled via scale_pos_weight
- Early stopping: 50 rounds on validation set

## 7. Key Findings

- 3mo achieves 75/69 — close to 75/70 target
- 6mo and 12mo achieve 70/67 — reliable detection 6-12 months before diagnosis
- NPV consistently ~90% across all windows
- AUC: 0.797 (3mo) -> 0.760 (6mo) -> 0.763 (12mo)
- 6mo and 12mo have similar AUC — melanoma features (skin lesions, dermatology pathway) are stable over time
- Noise filtering improved 3mo AUC by +1.1% (0.786 -> 0.797)
- Melanoma is inherently harder to predict than leukaemia due to lack of lab biomarkers and high overlap of skin lesion codes between cancer and non-cancer populations

## 8. Run Details

- Instance: GCloud cts-ai-dev-cpu-01 (n2d-standard-16, 16 vCPUs)
- Date: 2026-03-31
