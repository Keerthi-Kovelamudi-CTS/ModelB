"""Run the 12mo 1to1 Lung B+C pipeline (FE -> model). Outputs land in this folder.
Thin driver — shared logic in ../_run_pipeline.py (Option B)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from _run_pipeline import run_config

HERE = os.path.dirname(os.path.abspath(__file__))
run_config(window="12mo", ratio="1to1", out_dir=HERE)
