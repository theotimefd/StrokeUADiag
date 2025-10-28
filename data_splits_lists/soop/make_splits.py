import numpy as np
import pandas as pd
import glob
import os

#ROOT_DIR = "/home/fehrdelt/bettik/"
ROOT_DIR = "/bettik/PROJECTS/pr-gin5_aini/fehrdelt/"

SUB_EXPERIMENT_NAME = "densenet3d_exp_0_0"
SPLIT_SEED = 42

participants_tsv_path = ROOT_DIR+"datasets/final_soop_dataset_small/participants.tsv"

participants_df = pd.read_csv(participants_tsv_path, sep="\t")

# drop rows where 'nihss' is NaN
participants_df = participants_df.dropna(subset=["nihss"])

participants_df["high_nihss"] = (participants_df["nihss"] > 15).astype(np.int64)



counts = participants_df["high_nihss"].value_counts()
count_high_nihss_1 = int(counts.get(1, 0))
count_high_nihss_0 = int(counts.get(0, 0))

# drop up to 600 rows where high_nihss == 0
n_drop = min(600, count_high_nihss_0)
if n_drop > 0:
    drop_idx = participants_df[participants_df["high_nihss"] == 0].sample(n=n_drop, random_state=42).index
    participants_df = participants_df.drop(drop_idx).reset_index(drop=True)
    # update counts if needed
    counts = participants_df["high_nihss"].value_counts()
    count_high_nihss_1 = int(counts.get(1, 0))
    count_high_nihss_0 = int(counts.get(0, 0))

# merge clinical data df with image ids df
image_paths = glob.glob(ROOT_DIR+"datasets/StrokeUADiag_classification_inputs/stacked_*.nii.gz")
image_ids = [os.path.basename(path).replace(".nii.gz", "").split('_')[-1] for path in image_paths]

image_ids_df = pd.DataFrame(image_ids, columns=['participant_id'])

# make a new dataframe merging image_ids_df and participants_df on 'participant_id'
final_df = pd.merge(image_ids_df, participants_df, on='participant_id', how='left')
final_df.dropna(subset=["high_nihss"], inplace=True)

# drop rows with exclude files

# read the csv to get files to exclude
exclude_csv_path = ROOT_DIR+"StrokeUADiag/data_splits_lists/soop/exclude_failed_registration.csv"
exclude_df = pd.read_csv(exclude_csv_path, header=None)

exclude_files = exclude_df[0].tolist()
final_df = final_df[~final_df['participant_id'].isin(exclude_files)].reset_index(drop=True)

# Stratified split of participants_df into train/val/test
# Fractions can be adjusted if needed
train_frac, val_frac, test_frac = 0.7, 0.15, 0.15

assert "high_nihss" in final_df.columns, "Column 'high_nihss' not found."

rng = np.random.default_rng(SPLIT_SEED)
train_idx, val_idx, test_idx = [], [], []

for cls in final_df["high_nihss"].unique():
    cls_idx = final_df.index[final_df["high_nihss"] == cls].to_numpy()
    cls_idx = rng.permutation(cls_idx)

    n = len(cls_idx)
    n_train = int(np.floor(n * train_frac))
    n_val = int(np.floor(n * val_frac))
    n_test = n - n_train - n_val

    train_idx.extend(cls_idx[:n_train])
    val_idx.extend(cls_idx[n_train:n_train + n_val])
    test_idx.extend(cls_idx[n_train + n_val:])

# Build dataframes
train_final_df = final_df.loc[train_idx].reset_index(drop=True)
val_final_df = final_df.loc[val_idx].reset_index(drop=True)
test_final_df = final_df.loc[test_idx].reset_index(drop=True)

train_final_df = train_final_df.iloc[rng.permutation(len(train_final_df))].reset_index(drop=True)
val_final_df = val_final_df.iloc[rng.permutation(len(val_final_df))].reset_index(drop=True)
test_final_df = test_final_df.iloc[rng.permutation(len(test_final_df))].reset_index(drop=True)

# ---------------------

output_train_df = train_final_df[["participant_id", "high_nihss"]]
output_csv_path = os.path.join(ROOT_DIR+"StrokeUADiag/data_splits_lists/soop", "train.csv")
output_train_df.to_csv(output_csv_path, index=False)

output_val_df = val_final_df[["participant_id", "high_nihss"]]
output_csv_path = os.path.join(ROOT_DIR+"StrokeUADiag/data_splits_lists/soop", "val.csv")
output_val_df.to_csv(output_csv_path, index=False)

output_test_df = test_final_df[["participant_id", "high_nihss"]]
output_csv_path = os.path.join(ROOT_DIR+"StrokeUADiag/data_splits_lists/soop", "test.csv")
output_test_df.to_csv(output_csv_path, index=False)