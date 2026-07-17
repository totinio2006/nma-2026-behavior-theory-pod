# nma-2026-behavior-theory-pod

Neuromatch Academy 2026, comp neuro track, pod project - "State-dependent
decision-making in mice performing a 2AFC task" (IBL behavior dataset).

Working on Q9 (continuously-varying GLM via PsyTrack, following Roy et al
2021) and Q6 (cross-mouse stationarity) as part of the group project.

## TL;DR

Fit PsyTrack across 37 mice. Pupil doesn't predict contrast sensitivity at
the population level (null, p=0.48) but a Steinmetz et al 2019-style
engagement score does (p=0.008). Verified all sessions were recorded
post-training, not mid-training. Full writeup in `RESULTS.md`.

## Files

- `psytrack_single_mouse.py` - original single-mouse (FD_28) fit
- `psytrack_multi_mouse.py` - scaled to all mice + pupil/engagement correlations
- `pull_training_status.py` - pulls session-linked training status from IBL
- `check_trained_at_recording.py` - checks mice were actually trained when each session was recorded
- `psytrack_results.csv` - per-mouse results
- `mice_status_by_session.csv`, `sessions_with_status_at_time.csv`, `session_dates.csv` - training status data/verification
- `RESULTS.md` - full results writeup

## Setup

```
pip install psytrack ONE-api
```

Data included as `mice_pupil.csv` in this repo - no need to request it separately.
