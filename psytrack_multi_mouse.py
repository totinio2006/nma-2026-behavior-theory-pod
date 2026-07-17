"""
Q9/Q6 - same PsyTrack fit as psytrack_single_mouse.py, looped across mice.
Tests engagement and pupil (linear + quadratic) against contrast weight.
"""

import numpy as np
import pandas as pd
import psytrack as psy

DATA_PATH = 'mice_pupil.csv'
OUTPUT_PATH = 'psytrack_results.csv'
MIN_TRIALS = 50
ENGAGEMENT_WINDOW = 10


def get_mouse_column(df):
    return 'mouse_id_x' if 'mouse_id_x' in df.columns else 'mouse_id'


def compute_pretrial_pupil(mouse_df):
    pre = mouse_df[mouse_df['camera_time'] < mouse_df['goCue_times']]
    return pre.groupby('trial')['pupilDiameter_smooth'].mean()


def compute_engagement_score(mouse_df, window=ENGAGEMENT_WINDOW):
    all_trials = (mouse_df.groupby('trial').first()
                  .reset_index().sort_values('trial').reset_index(drop=True))
    is_miss = (all_trials['choice'] == 0).astype(float)
    miss_rate = is_miss.rolling(window=window, min_periods=1, center=True).mean()
    all_trials['engagement_score'] = 1 - miss_rate
    return all_trials[['trial', 'engagement_score']]


def fit_one_mouse(mouse_df):
    engagement_by_trial = compute_engagement_score(mouse_df)

    trials = (mouse_df.groupby('trial').first()
              .reset_index().sort_values('trial').reset_index(drop=True))
    trials = trials[trials['choice'] != 0].reset_index(drop=True)

    if len(trials) < MIN_TRIALS:
        return None

    pupil_by_trial = compute_pretrial_pupil(mouse_df)
    trials['pre_trial_pupil'] = trials['trial'].map(pupil_by_trial)
    trials = trials.merge(engagement_by_trial, on='trial', how='left')

    left = trials['contrastLeft'].fillna(0).values
    right = trials['contrastRight'].fillna(0).values
    contrast = (right - left).reshape(-1, 1)

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

    if np.isnan(wMode).any() or np.isinf(wMode).any():
        print("  fit produced NaN/Inf, skipping")
        return None

    names = sorted(weights.keys())
    stats = {'n_trials': len(trials), 'log_evidence': evd}
    for i, name in enumerate(names):
        w = wMode[i]
        stats[f'{name}_mean'] = w.mean()
        stats[f'{name}_std'] = w.std()
        stats[f'{name}_range'] = w.max() - w.min()

    contrast_traj = wMode[names.index('contrast')]

    pupil_vals = trials['pre_trial_pupil'].values
    valid = ~np.isnan(pupil_vals)
    if valid.sum() >= 10:
        x = pupil_vals[valid]
        y_fit = contrast_traj[valid]
        r_linear = np.corrcoef(x, y_fit)[0, 1]
        stats['pupil_contrast_corr'] = r_linear
        stats['pupil_n_valid'] = int(valid.sum())
        stats['r2_linear'] = r_linear ** 2

        coeffs = np.polyfit(x, y_fit, 2)
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
        stats['r2_quadratic'] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        stats['quad_coeff'] = coeffs[0]  # positive = U-shape, negative = inverted-U
    else:
        stats['pupil_contrast_corr'] = np.nan
        stats['pupil_n_valid'] = int(valid.sum())
        stats['r2_linear'] = np.nan
        stats['r2_quadratic'] = np.nan
        stats['quad_coeff'] = np.nan

    eng_vals = trials['engagement_score'].values
    eng_valid = ~np.isnan(eng_vals)
    if eng_valid.sum() >= 10:
        stats['engagement_contrast_corr'] = np.corrcoef(eng_vals[eng_valid], contrast_traj[eng_valid])[0, 1]
        stats['engagement_n_valid'] = int(eng_valid.sum())
    else:
        stats['engagement_contrast_corr'] = np.nan
        stats['engagement_n_valid'] = int(eng_valid.sum())

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
