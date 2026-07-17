"""
Was every pupil session actually recorded while the mouse was already
trained, or could some have snuck in mid-training?

Checking best-ever training status per mouse isn't good enough for this -
a mouse can eventually reach ready4ephysrig without every recorded session
having happened after that point. Need to match each session's actual date
against the status *at that date*, not ever.

Uses the output of pull_training_status.py (session-linked status history)
and does an as-of merge - for each pupil session, find the most recent
status change on or before that session's date.
"""

import pandas as pd
from one.api import ONE

ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')

df = pd.read_csv('mice_pupil.csv')
mouse_col = 'mouse_id_x' if 'mouse_id_x' in df.columns else 'mouse_id'
sessions = df[[mouse_col, 'session_id']].drop_duplicates()
print(f"{len(sessions)} unique (mouse, session) pairs")

session_dates = []
for _, row in sessions.iterrows():
    try:
        details = one.alyx.rest('sessions', 'read', id=row['session_id'])
        session_dates.append({
            'mouse_id': row[mouse_col],
            'session_id': row['session_id'],
            'session_date': pd.to_datetime(details['start_time']).normalize()
        })
    except Exception as e:
        print(f"  {row['session_id']} failed: {e}")

session_dates_df = pd.DataFrame(session_dates)
session_dates_df.to_csv('session_dates.csv', index=False)
print(f"saved {len(session_dates_df)} session dates")

status = pd.read_csv('mice_status_by_session.csv')
status['date'] = pd.to_datetime(status['date'])

results = []
for mouse in session_dates_df['mouse_id'].unique():
    mouse_sessions = session_dates_df[session_dates_df['mouse_id'] == mouse].sort_values('session_date')
    mouse_status = status[status['mouse_id'] == mouse].sort_values('date')

    if mouse_status.empty:
        continue

    # backward = most recent status ON OR BEFORE the session date
    merged = pd.merge_asof(mouse_sessions, mouse_status[['date', 'training_status']],
                            left_on='session_date', right_on='date', direction='backward')
    results.append(merged)

final = pd.concat(results, ignore_index=True)
final.to_csv('sessions_with_status_at_time.csv', index=False)

print("\nstatus AT TIME OF RECORDING, across all pupil sessions:")
print(final['training_status'].value_counts(dropna=False))

TRAINED_TIERS = ['trained 1a', 'trained 1b', 'ready4ephysrig', 'ready4recording', 'ready4delay']
n_trained_sessions = final['training_status'].isin(TRAINED_TIERS).sum()
trained_mice = final[final['training_status'].isin(TRAINED_TIERS)]['mouse_id'].unique()
print(f"\n{n_trained_sessions} of {len(final)} sessions recorded while already trained")
print(f"{len(trained_mice)} mice had at least one trained-status session: {list(trained_mice)}")
