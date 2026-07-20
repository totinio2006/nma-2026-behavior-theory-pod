"""
Q9/Q6 - PsyTrack fit looped across mice, pupil/engagement correlations,
non-linearity check, Q6 link test, shuffle control, and engagement
early/middle/late shape test (parallel to Ishayu's beta split).
"""

import numpy as np
import pandas as pd
import psytrack as psy
from scipy import stats as st

DATA_PATH = 'mice_pupil.csv'
OUTPUT_PATH = 'psytrack_results.csv'
MIN_TRIALS = 50
ENGAGEMENT_WINDOW = 10
N_SHUFFLES = 1000
np.random.seed(0)


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


def engagement_by_third(mouse_df, window=ENGAGEMENT_WINDOW):
    """Split a mouse's full trial sequence into early/middle/late thirds,
    same logic as Ishayu's beta split, and return mean engagement per third."""
    eng = compute_engagement_score(mouse_df, window)
    n = len(eng)
    third = n // 3
    early = eng.iloc[:third]['engagement_score'].mean()
    middle = eng.iloc[third:2*third]['engagement_score'].mean()
    late = eng.iloc[2*third:]['engagement_score'].mean()
    return early, middle, late


def fit_one_mouse(mouse_df):
    engagement_by_trial = compute_engagement_score(mouse_df)

    trials = (mouse_df.groupby('trial').first()
              .reset_index().sort_values('trial').reset_index(drop=True))
    trials = trials[trials['choice'] != 0].reset_index(drop=True)

    if len(trials) < MIN_TRIALS:
        return None, None

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
        return None, None

    if np.isnan(wMode).any() or np.isinf(wMode).any():
        print("  fit produced NaN/Inf, skipping")
        return None, None

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
        stats['quad_coeff'] = coeffs[0]
    else:
        stats['pupil_contrast_corr'] = np.nan
        stats['pupil_n_valid'] = int(valid.sum())
        stats['r2_linear'] = np.nan
        stats['r2_quadratic'] = np.nan
        stats['quad_coeff'] = np.nan

    eng_vals = trials['engagement_score'].values
    eng_valid = ~np.isnan(eng_vals)
    raw_for_shuffle = None
    if eng_valid.sum() >= 10:
        e = eng_vals[eng_valid]
        c = contrast_traj[eng_valid]
        stats['engagement_contrast_corr'] = np.corrcoef(e, c)[0, 1]
        stats['engagement_n_valid'] = int(eng_valid.sum())
        stats['engagement_std'] = np.nanstd(e)
        raw_for_shuffle = (e, c)
    else:
        stats['engagement_contrast_corr'] = np.nan
        stats['engagement_n_valid'] = int(eng_valid.sum())
        stats['engagement_std'] = np.nan

    early_e, mid_e, late_e = engagement_by_third(mouse_df)
    stats['engagement_early'] = early_e
    stats['engagement_middle'] = mid_e
    stats['engagement_late'] = late_e

    return stats, raw_for_shuffle


def main():
    df = pd.read_csv(DATA_PATH)
    mouse_col = get_mouse_column(df)
    mice = df[mouse_col].unique()
    print(f"{len(mice)} mice found")

    rows = []
    raw_data = {}
    for mouse_id in mice:
        print(f"fitting {mouse_id}...")
        stats, raw = fit_one_mouse(df[df[mouse_col] == mouse_id])
        if stats is None:
            print(f"  skipped {mouse_id}")
            continue
        stats['mouse_id'] = mouse_id
        rows.append(stats)
        if raw is not None:
            raw_data[mouse_id] = raw

    results = pd.DataFrame(rows)
    print(f"\nfit {len(results)}/{len(mice)} mice successfully")
    print(results)

    results.to_csv(OUTPUT_PATH, index=False)
    print(f"\nsaved to {OUTPUT_PATH}")

    valid_q6 = results.dropna(subset=['engagement_std', 'contrast_std'])
    if len(valid_q6) >= 10:
        r, p = st.pearsonr(valid_q6['engagement_std'], valid_q6['contrast_std'])
        print(f"\nEngagement variability vs contrast_std (Q6 link): n={len(valid_q6)}, r={r:.3f}, p={p:.3f}")

    real_corrs = results['engagement_contrast_corr'].dropna().values
    real_mean_r = real_corrs.mean()
    print(f"\nReal engagement-contrast mean r across {len(real_corrs)} mice: {real_mean_r:.4f}")
    print(f"Running {N_SHUFFLES} shuffles for permutation control...")

    null_means = []
    for i in range(N_SHUFFLES):
        shuffled_rs = []
        for mouse_id, (e, c) in raw_data.items():
            e_shuffled = np.random.permutation(e)
            r = np.corrcoef(e_shuffled, c)[0, 1]
            if not np.isnan(r):
                shuffled_rs.append(r)
        null_means.append(np.mean(shuffled_rs))
    null_means = np.array(null_means)

    p_empirical = np.mean(np.abs(null_means) >= np.abs(real_mean_r))
    print(f"\nNull distribution (shuffled): mean={null_means.mean():.4f}, std={null_means.std():.4f}")
    print(f"Empirical p-value (real vs shuffled null): {p_empirical:.4f}")

    valid_shape = results.dropna(subset=['engagement_early', 'engagement_late'])
    if len(valid_shape) >= 10:
        t, p = st.ttest_rel(valid_shape['engagement_early'], valid_shape['engagement_late'])
        print(f"\nEngagement early vs late (paired t-test): n={len(valid_shape)}, t={t:.3f}, p={p:.4f}")
        print(f"Mean engagement - early: {valid_shape['engagement_early'].mean():.3f}, "
              f"middle: {valid_shape['engagement_middle'].mean():.3f}, "
              f"late: {valid_shape['engagement_late'].mean():.3f}")


if __name__ == '__main__':
    main()
