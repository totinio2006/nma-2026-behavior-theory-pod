"""
Q9/Q6 - same PsyTrack fit as psytrack_single_mouse.py, looped across mice.

Point DATA_PATH at the full multi-mouse sheet once it's ready (Samyuktha's
pulling ~30 mice as of writing). Each mouse gets fit independently, and we
save the per-mouse weight stats to a CSV so we can compare across mice for
Q6 (are some mice more stationary than others?).
"""

import numpy as np
import pandas as pd
import psytrack as psy

DATA_PATH = 'mice - mice.csv.csv'
OUTPUT_PATH = 'psytrack_results.csv'
MIN_TRIALS = 50  # skip a mouse if it doesn't have enough usable trials


def get_mouse_column(df):
    # earlier version of the sheet had both mouse_id_x and mouse_id_y from
    # a merge - they matched, but worth double checking once more mice are in
    return 'mouse_id_x' if 'mouse_id_x' in df.columns else 'mouse_id'


def fit_one_mouse(mouse_df):
    """Collapse one mouse's raw rows to trials, fix the choice sign issue,
    and fit PsyTrack. Returns None if there isn't enough usable data.
    """
    trials = (mouse_df.groupby('trial').first()
              .reset_index().sort_values('trial').reset_index(drop=True))
    trials = trials[trials['choice'] != 0].reset_index(drop=True)

    if len(trials) < MIN_TRIALS:
        return None

    left = trials['contrastLeft'].fillna(0).values
    right = trials['contrastRight'].fillna(0).values
    contrast = (right - left).reshape(-1, 1)

    # choice == -1 is Right, choice == 1 is Left (checked against known-correct
    # single-side-contrast trials - opposite of the naive guess)
    y = np.where(trials['choice'].values == -1, 2, 1)

    prev_choice = np.roll(y, 1).astype(float)
    prev_choice[0] = 0
    prev_choice = prev_choice.reshape(-1, 1)

    dat = {'y': y, 'inputs': {'contrast': contrast, 'prevChoice': prev_choice}}
    weights = {'bias': 1, 'contrast': 1, 'prevChoice': 1}
    n_weights = sum(weights.values())
    hyper = {'sigma': [2 ** -4] * n_weights, 'sigInit': 2 ** 4}

    try:
        hyp, evd, wMode, _ = psy.hyperOpt(dat, hyper, weights, ['sigma'])
    except Exception as e:
        print(f"  fit failed: {e}")
        return None

    # sanity check - PsyTrack's optimizer throws some RuntimeWarnings mid-search
    # (divide-by-zero/overflow while it explores extreme candidate values) but
    # they don't affect the final result. Confirmed by checking for NaN/Inf here.
    if np.isnan(wMode).any() or np.isinf(wMode).any():
        print("  fit produced NaN/Inf, skipping")
        return None

    names = sorted(weights.keys())
    stats = {'n_trials': len(trials), 'log_evidence': evd}
    for i, name in enumerate(names):
        w = wMode[i]
        stats[f'{name}_mean'] = w.mean()
        stats[f'{name}_std'] = w.std()   # main stationarity metric - higher = more drift
        stats[f'{name}_range'] = w.max() - w.min()
    return stats


def main():
    df = pd.read_csv(DATA_PATH)
    mouse_col = get_mouse_column(df)
    mice = df[mouse_col].unique()
    print(f"{len(mice)} mice found")

    rows = []
    for mouse_id in mice:
        print(f"fitting {mouse_id}...")
        stats = fit_one_mouse(df[df[mouse_col] == mouse_id])
        if stats is None:
            print(f"  skipped {mouse_id}")
            continue
        stats['mouse_id'] = mouse_id
        rows.append(stats)

    results = pd.DataFrame(rows)
    print(f"\nfit {len(results)}/{len(mice)} mice successfully")
    print(results)

    results.to_csv(OUTPUT_PATH, index=False)
    print(f"\nsaved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
