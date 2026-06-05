# Ovarian v4 SQL — TODO

Add 4 files here:
- 1mo_v4.sql
- 3mo_v4.sql
- 6mo_v4.sql
- 12mo_v4.sql

Template: copy from `Prostate_2.0_1to1/SQL Queries/v4/`, then change:
- Title comment: `(v4)` for Ovarian
- Cohort filter: `target_cancer_pattern = '%ovarian%'`
- Sex filter: `pp.sex = 'F'` (or remove the filter if both)
- `cancer_snomed_codes`: Ovarian-specific diagnosis SNOMEDs
- `months_before` per file (1, 3, 6, 12)
