# 1_Top_Snomed — feature (code) scoring

Ranks which observation / medication codes separate cancer vs non-cancer, on the **same
cohort the model trains on**. Run on a VM with BigQuery access.

Both horizons (**12mo + 1mo**) are produced in **one run** — each from its cohort SQL in
`SQL/` (`12mo_1to1.sql`, `1mo_1to1.sql`) — and written to separate subfolders so they never
overlap: `Data/{12mo,1mo}/` and `output/{12mo,1mo}/`.

### Step 1 — build inputs  (both horizons)
```bash
python build_score_counts.py
```
→ for each horizon, runs `SQL/{horizon}_1to1.sql` on BigQuery and writes to `./Data/{horizon}/`:
- `Lung_{positive,negative}_{obs,meds}.csv` — per-code patient counts
- `Lung_patient_{obs,meds}_matrix.csv` — real per-patient data for the ML step

### Step 2 — score & rank  (both horizons)
```bash
python "Combined Scoring (ML+Stat).py"
```
→ for each horizon, reads `./Data/{horizon}/`, ranks codes (statistical OR/χ² + real-data ML,
rank-normalised), writes to `./output/{horizon}/Scores_lung/`:
- `lung_obs_all.csv`, `lung_meds_all.csv`, **`lung_combined_all.csv`** ← ranked deliverable

> Run **Step 1 before Step 2**. Each script builds/scores **both horizons in one run**. The
> matrices from Step 1 are **required** — the scorer stops if they're missing (it never
> fabricates patient data).

```
SQL/{12mo,1mo}_1to1.sql ──build_score_counts.py──▶ Data/{12mo,1mo}/ (counts + matrices)
                        ──"Combined Scoring (ML+Stat).py"──▶ output/{12mo,1mo}/Scores_lung/lung_combined_all.csv
```
