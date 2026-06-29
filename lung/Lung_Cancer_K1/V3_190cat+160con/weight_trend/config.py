"""
Sliding window weight trend analysis config.
Astra 10yr cohort, Phase 1, 12mo model.
"""

# Input
PARQUET_PATH = "../data/Astra_10yr_withtext.parquet"

# Target terms — add any SNOMED term string here to include in pipeline + visuals
TERMS = [
    "Body weight",
    "Body mass index",
]

# Plausible value range per term — rows outside range are dropped as dirty data
# Set to (None, None) to skip range filtering for a term
TERM_VALUE_RANGES = {
    "Body weight":     (20, 300),   # kg
    "Body mass index": (10, 80),    # kg/m²
}

# Windowing
WINDOW_DAYS = 91       # ~3 months per window
LOOKBACK_DAYS = 3650   # 10 years
N_WINDOWS = LOOKBACK_DAYS // WINDOW_DAYS   # 40 windows; w00=most recent, w39=oldest

# Output
OUTPUT_PATH = "output/weight_features.parquet"
