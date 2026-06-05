import argparse
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = SCRIPT_DIR / 'results' / '1_training'
WINDOWS = ['1mo', '2mo', '3mo', '6mo', '9mo', '12mo']


def load_window(results_dir, window):
    p = results_dir / window / f'selected_features_{window}.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if 'importance' in df.columns:
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    df['window'] = window
    return df


def main():
    ap = argparse.ArgumentParser(description='Extract top-N features per window from modeling outputs.')
    ap.add_argument('--top-n', type=int, default=300)
    ap.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS)
    ap.add_argument('--out-dir', type=Path, default=None)
    ap.add_argument('--windows', nargs='+', default=WINDOWS)
    args = ap.parse_args()

    out_dir = args.out_dir or (SCRIPT_DIR / 'results' / 'top_features')
    out_dir.mkdir(parents=True, exist_ok=True)

    per_window = []
    for w in args.windows:
        df = load_window(args.results_dir, w)
        if df is None:
            print(f"[{w}] missing — skipping")
            continue
        top = df.head(args.top_n).copy()
        top.to_csv(out_dir / f'top_{args.top_n}_{w}.csv', index=False)
        per_window.append(top)
        print(f"[{w}] {len(top)} features | top: {top.iloc[0]['feature']} ({top.iloc[0].get('importance', '?')})")

    if not per_window:
        print("No windows found.")
        return

    long = pd.concat(per_window, ignore_index=True)
    long.to_csv(out_dir / f'top_{args.top_n}_all_windows_long.csv', index=False)

    wide_rank = long.pivot(index='feature', columns='window', values='rank')
    wide_rank = wide_rank[[w for w in args.windows if w in wide_rank.columns]]
    wide_rank['n_windows'] = wide_rank.notna().sum(axis=1)
    wide_rank['avg_rank'] = wide_rank[[w for w in args.windows if w in wide_rank.columns]].mean(axis=1)
    wide_rank = wide_rank.sort_values(['n_windows', 'avg_rank'], ascending=[False, True])
    wide_rank.to_csv(out_dir / f'top_{args.top_n}_all_windows_wide.csv')

    consensus = wide_rank[wide_rank['n_windows'] == len(per_window)]
    print(f"\n  Consensus features (in top-{args.top_n} for all {len(per_window)} windows): {len(consensus)}")
    print(f"  Outputs: {out_dir}")


if __name__ == '__main__':
    main()
