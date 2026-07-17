"""
Grabs session-linked training status for our mice via IBL's public
2021_Q1_IBL_et_al_Behaviour tag (same tag K used in their Q1 notebook).

Took way longer than it should have. load_aggregate kept throwing an
AssertionError with no message, turned out to be a multi-collection
ambiguity (public/aggregates vs aggregates paths both exist for some
subjects). Downgrading ONE-api didn't fix it, REST filters didn't fix it.
What actually works: grab the flatironinstitute.org URL specifically
(not the S3 mirror - that one fails ONE's own auth check) and pull it
through one.alyx.download_file() instead of a bare pd.read_parquet(url).
"""

import pandas as pd
from one.api import ONE

PUPIL_DATA_PATH = 'mice_pupil.csv'
OUTPUT_PATH = 'mice_status_by_session.csv'

ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')

df = pd.read_csv(PUPIL_DATA_PATH)
mouse_col = 'mouse_id_x' if 'mouse_id_x' in df.columns else 'mouse_id'
mouse_ids = set(df[mouse_col].unique())

training_datasets = one.alyx.rest(
    'datasets', 'list',
    tag='2021_Q1_IBL_et_al_Behaviour',
    name='_ibl_subjectTraining.table.pqt'
)

# has to be the flatiron url specifically, the s3 one fails download_file's
# internal check against HTTP_DATA_SERVER
training_urls = {}
for d in training_datasets:
    subject_name = d['file_records'][0]['relative_path'].split('/')[2]
    for fr in d['file_records']:
        if 'flatironinstitute.org' in fr['data_url']:
            training_urls[subject_name] = fr['data_url']
            break

print(f"{len(training_urls)} subjects available, pulling our {len(mouse_ids)} mice")

all_status = []
failed = []

for subject in mouse_ids:
    if subject not in training_urls:
        print(f"  {subject}: no flatiron url found")
        failed.append(subject)
        continue
    try:
        local_path = one.alyx.download_file(training_urls[subject])
        training_df = pd.read_parquet(local_path)
        training_df = training_df.reset_index()  # session becomes a column
        training_df['mouse_id'] = subject
        all_status.append(training_df)
        print(f"  {subject}: {len(training_df)} session records")
    except Exception as e:
        print(f"  {subject} failed: {type(e).__name__}: {e}")
        failed.append(subject)

if all_status:
    status_df = pd.concat(all_status, ignore_index=True)
    status_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nsaved {len(status_df)} rows to {OUTPUT_PATH}")
    print(f"columns: {status_df.columns.tolist()}")
else:
    print("\nnothing pulled, something's still broken")

if failed:
    print(f"\n{len(failed)} mice failed: {failed}")
