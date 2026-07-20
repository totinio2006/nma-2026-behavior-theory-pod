# Q9 - PsyTrack results

Started from the FD_28 pilot (see `psytrack_single_mouse.py`), scaled up to
the full mouse sheet. Notes below, roughly in the order I found things.

## The fit itself

PsyTrack, 3 weights (bias, contrast, prevChoice), same setup as the single
mouse version. 37/38 mice fit clean - SWC_023 dropped, only 50 trials and
a 70% miss rate, not worth trying to fit.

Found and fixed a real bug early on: `choice == -1` is actually Right and
`choice == 1` is Left in the raw data, opposite of what I assumed. Verified
using trials where only one side had contrast and was marked correct, so
ground truth was known for sure.

## Q6 - are some mice more stationary than others

Yes, pretty clearly. `contrast_std` (how much the contrast weight moves
around within a session) ranges from basically 0 (DY_009 - flat, one
strategy the whole time) to ~7.4 (ibl_witten_13 - swings a lot). Full numbers
in `psytrack_results.csv`.

## Pupil vs contrast weight - null

FD_28 alone showed r = -0.544 between pre-trial pupil and contrast weight,
looked like a real effect. Didn't hold up: across 35 mice, mean r = 0.036,
p = 0.48, split ~50/50 in direction. Doesn't generalize. Pilot mouse was
probably just unusual.

## Engagement vs contrast weight - actually significant

Tried a different angle after the pupil thing didn't pan out - built a
rolling engagement score based on Steinmetz et al 2019 (they define
disengagement via miss/no-go streaks, same task as ours). Made it
continuous instead of their binary version.

Across 28 mice with usable data (9 had zero miss trials at all, so nothing
to correlate against - checked this, not a bug): mean r = -0.11, p = 0.008,
68% of mice same direction. Actually holds up, unlike pupil.

## Trained-mice check

Liza said only ~5 mice were trained with pupil data, my first check (best
ever training status per mouse) said all 37 were trained-tier - these
aren't the same question. Best-ever status doesn't tell you whether the
specific session was recorded before or after the mouse got there.

Pulled actual session dates + session-linked training status (see
`pull_training_status.py` - this took forever, ONE-api kept throwing
ambiguous errors, writeup of what actually worked is in the script comments)
and matched them properly (`check_trained_at_recording.py`). Result:
120/120 sessions across all 38 mice were recorded post-training. No
contamination. Pupil and engagement results above don't need any caveat.

## Still open

- mice_pupil.csv has no response_times/firstMovement_times, need these for
  anyone doing HDDM
- Q2 (lapse probability) is the one gap left in the core template questions
  that I haven't touched
- worth checking if engagement variability explains some of the Q6 spread -
  haven't tested this yet

## Files here

- `psytrack_single_mouse.py` - original FD_28 fit
- `psytrack_multi_mouse.py` - all mice + pupil/engagement correlations
- `pull_training_status.py` - session-linked training status pull
- `check_trained_at_recording.py` - the as-of merge for the trained-mice check
- `psytrack_results.csv` - per-mouse results
- `mice_status_by_session.csv` / `sessions_with_status_at_time.csv` - training status stuff

## Non-linearity check on pupil (prompted by Varad's question)

A null linear result doesn't rule out a non-linear one, so fit a quadratic
model (contrast_weight ~ pupil + pupil^2) across all 35 mice with valid pupil
data - relevant given Hulsey et al. 2024's HMM-based inverted-U finding
(this is the paper previously misfiled as "McGinley et al." - McCormick was
McGinley's PI, easy mixup, corrected here).

Mean R^2 improves modestly, 0.084 (linear) to 0.120 (quadratic). 24/35 mice
(69%) show a U-shape, not the inverted-U the literature predicts - only
11/35 (31%) go the expected direction.

Robustness check (IQR outlier removal, 9/35 mice flagged): direction holds
(69% U-shape either way), but strength doesn't - median drops to ~0, and
Wilcoxon signed-rank test is no longer significant (p = 0.136) without the
high-leverage mice. Honest read: modest, directionally-consistent lean
toward U-shape, not a solid effect on its own.

Doesn't necessarily contradict Hulsey et al. - they use pupil AND movement
together via discrete HMM states on raw performance; this is pupil alone,
continuous regression, against a model-derived weight. Different method,
different outcome variable.

Ran the same outlier check on the engagement result for comparison: zero
outliers detected, result completely unchanged (p = 0.008 either way).
Engagement is the solid, load-bearing finding; the pupil U-shape is a
weaker, secondary observation.

## Supporting literature (found by Varad)

Ortiz, Aziz & Hestrin (2020, Cell Reports), "Motivation and Engagement during
Visually Guided Behavior" - independent support for both main findings above,
from a completely different lab/setup (head-fixed 2AFC with V1 electrophysiology,
not IBL data).

They use running miss rate (10-trial blocks) as their engagement signal, same
basic idea as the rolling miss-rate score I built independently, and find a
sharp transition from near-zero to very high miss rate when mice disengage.

More importantly for the pupil result: they directly tested whether pre-trial
pupil diameter predicts contrast threshold, bias, or lapse rate (splitting
trials into small-pupil vs large-pupil groups) and found no difference on any
of the three. Pupil diameter didn't even decrease during disengagement (if
anything it was ~10% larger), arguing directly against reduced arousal driving
disengagement. Basically an independent replication of the null pupil result
above, from a different species-adjacent paradigm.

They also found V1 spike counts drop significantly during disengagement,
giving a plausible neural mechanism for why engagement (not pupil) tracks
performance.	

## Shuffle control on the engagement result

Ran a permutation control to rule out the engagement-contrast correlation being
a pipeline artifact or spurious shared-trend effect. For each mouse, shuffled its
engagement scores relative to its contrast-weight trajectory (breaking the real
trial-by-trial link while keeping each mouse's own data distribution intact),
recomputed the population mean correlation, repeated 1000 times to build a null
distribution.

Real mean r = -0.107. Null distribution: mean = 0.0001, std = 0.0074. Empirical
p-value = 0.0000 (real result more extreme than all 1000 shuffles).

This is a strong validation - not just significant, but a complete separation
from the shuffled null. The engagement-contrast relationship survives outlier
removal (unchanged, see earlier) AND this permutation control. This is the most
solid result to come out of Q9.

Also consistent with Saderi, Schwartz, Heller, Pennington & David (2021, eLife),
"Dissociation of task engagement and arousal effects in auditory cortex and
midbrain" (found by Varad) - they show that task engagement and pupil-indexed
arousal modulate largely independent neural populations, with engagement effects
more prominent in cortex (A1) and arousal effects more prominent in midbrain (IC).
This suggests engagement and arousal may be dissociable at the circuit level more
broadly, not just in this specific dataset - a plausible biological explanation for
why our engagement measure outperforms pupil: engagement may be closer to the
decision-to-act circuit than to the general arousal pathway pupil indexes. Note
this study used ferrets performing an auditory task, not mice on a visual task, so
it supports the general principle rather than being a direct replication.

## Engagement shape across a session (prompted by Ishayu's early/middle/late split)

Tested whether engagement follows the same early/middle/late pattern as Ishayu's
significant beta result (rise then fall). Split each mouse's full trial sequence
into thirds and averaged the engagement score in each.

n=37, paired t-test (early vs late): t=3.789, p=0.0006. Mean engagement - early:
0.999, middle: 0.995, late: 0.982.

Real effect, but a different shape than expected: engagement steadily declines
across the session rather than rising then falling. Doesn't confirm the
"settles in, then fatigues" hypothesis in its exact form - more like engagement
just steadily erodes from the start, no early warm-up bump.

Robustness check: removed 3 highest-leverage mice (IQR method) - result barely
changes (p=0.0011). Also confirmed non-parametrically (Wilcoxon signed-rank,
p=0.0001). Genuinely robust, unlike the pupil quadratic result.

12/37 mice show zero change at all (perfectly flat engagement all session) -
consistent with many mice having very few miss trials overall.
