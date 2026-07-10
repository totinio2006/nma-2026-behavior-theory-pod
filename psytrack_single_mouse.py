"""
Q9 - fitting mouse choice behaviour as a continuously drifting GLM
(PsyTrack, following Roy et al. 2021)

Uses the IBL trial data pulled from our shared sheet. Single mouse for now
(FD_28) as a proof of concept before scaling to the full set.
"""

import numpy as np
import pandas as pd
import psytrack as psy
import matplotlib.pyplot as plt

DATA_PATH = 'mice - mice.csv.csv'


def load_trials(path):
    """Collapse the raw per-frame pupil data down to one row per trial.

    The raw file has ~300 rows per trial (one per camera frame), but
    choice/contrast/feedback don't change within a trial - only the pupil
    trace does. groupby().first() is fine here since we just want the
    trial-level fields.
    """
    raw = pd.read_csv(path)
    trials = (raw.groupby('trial').first()
              .reset_index()
              .sort_values('trial')
              .reset_index(drop=True))

    # drop no-go trials, we're only modelling the binary L/R choice
    trials = trials[trials['choice'] != 0].reset_index(drop=True)
    return trials


def check_choice_convention(trials):
    """The raw 'choice' column isn't labelled the way you'd guess.

    Confirmed this by looking at trials where only one side had any
    contrast and the trial was correct - on those trials we know for
    certain which way the mouse went, since there's only one possible
    right answer.
    """
    only_right = (trials['contrastLeft'].fillna(0) == 0) & (trials['contrastRight'].fillna(0) > 0)
    only_left = (trials['contrastRight'].fillna(0) == 0) & (trials['contrastLeft'].fillna(0) > 0)
    correct = trials['feedbackType'] == 1

    print("choice value when only-right-contrast + correct (should be one value):")
    print(trials[only_right & correct]['choice'].value_counts())
    print("choice value when only-left-contrast + correct (should be the other):")
    print(trials[only_left & correct]['choice'].value_counts())
    # -> confirms choice == -1 is actually Right, choice == 1 is Left


def build_psytrack_inputs(trials):
    left = trials['contrastLeft'].fillna(0).values
    right = trials['contrastRight'].fillna(0).values
    contrast = (right - left).reshape(-1, 1)

    # -1 = Right, 1 = Left (see check_choice_convention) -> recode to PsyTrack's 1/2
    y = np.where(trials['choice'].values == -1, 2, 1)

    prev_choice = np.roll(y, 1).astype(float)
    prev_choice[0] = 0
    prev_choice = prev_choice.reshape(-1, 1)

    return {
        'y': y,
        'inputs': {'contrast': contrast, 'prevChoice': prev_choice}
    }


def fit(dat):
    weights = {'bias': 1, 'contrast': 1, 'prevChoice': 1}
    n_weights = sum(weights.values())

    hyper = {'sigma': [2 ** -4] * n_weights, 'sigInit': 2 ** 4}
    hyp, evd, wMode, hess_info = psy.hyperOpt(dat, hyper, weights, ['sigma'])

    # note: PsyTrack orders wMode rows alphabetically by weight name, not
    # by the order given in the `weights` dict above
    names = sorted(weights.keys())
    return names, wMode, evd


if __name__ == '__main__':
    trials = load_trials(DATA_PATH)
    print(f"{len(trials)} trials after cleaning")

    check_choice_convention(trials)

    dat = build_psytrack_inputs(trials)
    names, wMode, evd = fit(dat)

    print(f"\nlog-evidence: {evd:.2f}")
    for i, name in enumerate(names):
        w = wMode[i]
        print(f"  {name}: mean={w.mean():.2f}  range=[{w.min():.2f}, {w.max():.2f}]")

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(names):
        ax.plot(wMode[i], label=name)
    ax.set(xlabel='Trial', ylabel='Weight value', title='PsyTrack weights over trials (FD_28)')
    ax.legend()
    plt.show()
