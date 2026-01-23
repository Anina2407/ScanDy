import os
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

import scripts.scanpath_utils as su

## scanpath evaluation
# create dataframe consisting of one row per gaze event

path_control = "/mnt/raid/data/anina/ScanDy/data/BOFA_eval_data/controls_5s_eval/controls_5s_eval"#"data/BOFA_eval_data/controls_5s_eval"
path_with = "/mnt/raid/data/anina/ScanDy/data/BOFA_eval_data/parkinson_with_5s_eval/parkinson_with_5s_eval"#"data/BOFA_eval_data/parkinson_with_5s_eval"
path_without = "/mnt/raid/data/anina/ScanDy/data/BOFA_eval_data/parkinson_without_5s_eval"#
path_results = "/mnt/raid/data/anina/ScanDy/data/new_generation_2"

image_size = (1080, 1920)
display_size = (1080, 1920)
#display_w_dva = 47.7 for young healthy participants
display_w_dva = 40.45   # following Nicos code, fix_obj_summary_psycsci
max_scaling = 0.8

px2dva = su.calc_px2dva(image_size, display_size, display_w_dva, max_scaling)

# same order as in healthy young
column_order = [
    "subj_id",
    "block",
    "trial_in_block",
    "trial_id",
    "scene",
    "animate", # actually also extra column
    "video",
    "t_start",
    "t_end",
    "duration_ms",
    "x",
    "y",
    "obj",
    "sac_amp_dva",
    "fov_cat",
]

extra_columns = ["sac_angle_h", "sac_angle_p"]



###############################################
# loop over all 3 groups

for path_group, group_name in zip([path_control, path_with, path_without],
                               ['controls', 'parkinson_with', 'parkinson_without']):
                            
    # read in all subjects (all files ending with _hpc.csv.gz)
    subject_files = [f for f in os.listdir(path_group) if f.endswith("_hpc.csv.gz")]

    # create dataframe to fill in all data
    df_all = pd.DataFrame()

    # loop over all subjects
    for i, subject_file in enumerate(subject_files):
        print(f"Processing subject file: {subject_file} in group: {group_name}")
        print(f"{i} / {len(subject_files)} subjects")

        df_data = pd.read_csv(os.path.join(path_group, subject_file), compression="gzip")

        df_data["trial_id"] = (
            df_data["subj_id"].astype(str)
            + "_"
            + df_data["block"].astype(int).map("{:02d}".format)
            + "_"
            + df_data["trial_in_block"].astype(int).map("{:02d}".format)
        )

        df_sbj = pd.DataFrame()

        # loop over all trials for this subject
        for trial in df_data["trial_id"].unique():

            df_trial = df_data[df_data["trial_id"] == trial].copy()

             # --- SAFETY CHECK 1: empty trial ---
            if df_trial.empty:
                print(f"Skipping empty trial: {trial}")
                continue
            
            subj = df_trial["subj_id"].iloc[0]
            
            # convert PSO to FOV
            df_trial["em_rv"].replace(["PSO"], ["FOV"], inplace=True)

            # exclude all events that are not FOV or SAC
            df_trial = df_trial[
                df_trial["em_rv"].isin(["FOV", "SAC"])
            ].reset_index(drop=True)

            # --- SAFETY CHECK 2: after filtering, may be empty ---
            if df_trial.empty:  
                print(f"Trial {trial} (subj {subj}) has no valid events. Skipping.")
                continue

            # --- SAFETY CHECK 3: Check for FOV events before slicing ---
            fov_indices_check = df_trial[df_trial['em_rv'] == "FOV"].index
            if len(fov_indices_check) == 0:  # ← CHECK BEFORE ANY SLICING
                print(f"No FOV in trial {trial} (subject {subj}). Skipping.")
                continue

            # start and end with Foveation (crashed for some trials otherwise)
            first_fov = fov_indices_check[0]  
            last_fov = fov_indices_check[-1]
            df_trial = df_trial.loc[first_fov:last_fov, :].copy()  

            # Reset index after slicing
            df_trial = df_trial.reset_index(drop=True)
            
            group = df_trial.groupby(
                df_trial["em_rv"].ne(df_trial["em_rv"].shift()).cumsum()
            )

            df_fov = group.apply(
                lambda entry: pd.DataFrame(
                    {
                        "t_start": [int(entry["t"].min())],
                        "t_end": [int(entry["t"].max())],
                        "duration_ms": [int(entry["t"].max() - entry["t"].min())],
                        "event": [entry["em_rv"].iloc[0]],
                        "x_start": [entry["x"].iloc[0]],
                        "x_end": [entry["x"].iloc[-1]],
                        "y_start": [entry["y"].iloc[0]],
                        "y_end": [entry["y"].iloc[-1]],
                        "x": [int(round(entry["x"].mean() if not np.isnan(entry["x"].mean()) else 0, 0))],
                        "y": [int(round(entry["y"].mean() if not np.isnan(entry["y"].mean()) else 0, 0))],
                        "obj": [su.dominant_obj(entry)],
                    }
                )
            ).reset_index(drop=True)

            # remove saccades from df
            df_fov = df_fov[(df_fov.event == "FOV")]
            df_fov.drop('event', axis=1, inplace=True)
            df_fov.reset_index(inplace=True, drop = True)

            df_fov.insert(0, "subj_id", df_trial['subj_id'].iloc[0])
            df_fov.insert(1, "block", df_trial['block'].iloc[0])
            df_fov.insert(2, "trial_in_block", df_trial['trial_in_block'].iloc[0])
            df_fov.insert(3, "trial_id", df_trial['trial_id'].iloc[0])
            df_fov.insert(4, "scene", df_trial['scene'].iloc[0])
            df_fov.insert(5, "animate", df_trial['animate'].iloc[0])
            df_fov.insert(6, "video", df_trial['video'].iloc[0])

            gaze_shifts_x = np.array(np.array(df_fov["x_start"].iloc[1:]) - np.array(df_fov["x_end"].iloc[:-1]))
            gaze_shifts_y = np.array(np.array(df_fov["y_start"].iloc[1:]) - np.array(df_fov["y_end"].iloc[:-1]))
            
            df_fov["gaze_shifts_x"] = list(gaze_shifts_x) + [np.nan]
            df_fov["gaze_shifts_y"] = list(gaze_shifts_y) + [np.nan]

            df_fov["sac_amp_dva"] = list(
                np.sqrt(gaze_shifts_x**2 + gaze_shifts_y**2) * px2dva
            ) + [np.nan]

            df_fov["sac_angle_h"] = list(
                    -np.arctan2(gaze_shifts_x, gaze_shifts_y) / np.pi * 180
                ) + [np.nan]
            
            df_fov["sac_angle_p"] =  [np.nan] + [
                    su.angle_limits(
                        df_fov["sac_angle_h"].iloc[i + 1] - df_fov["sac_angle_h"].iloc[i]
                    )
                    for i in range(len(df_fov) - 1)
                ]
            
            df_fov["fov_cat"] = su.fov_cat(df_fov)

             # --- SAFETY CHECK 4: Check for NaN objects - skip entire trial if any found ---
            if df_fov["obj"].isna().any(): 
                print(f"Trial {trial} (subj {subj}) contains NaN objects. Skipping entire trial.")
                continue

            if df_fov["obj"].any() == 'nan' or df_fov["obj"].any() == '':
                print(f"Trial {trial} (subj {subj}) contains NaN objects. Skipping entire trial.")
                continue
            # select only the desired columns in the desired order
            df_fov = df_fov[column_order + extra_columns]

            df_sbj = pd.concat([df_sbj, df_fov])

        df_all = pd.concat([df_all, df_sbj])

    bad_trials = (
    df_all.groupby("trial_id")["obj"]
      .apply(lambda s: s.isna().any())
    )
    removed_trials = bad_trials[bad_trials].index.tolist()

    print(f"Removing {len(removed_trials)} trials, as they contain NaN objects:")
    print(removed_trials)

    df_all_clean = df_all.loc[~df_all["trial_id"].isin(bad_trials[bad_trials].index)]

    # Write file per group
    df_all_clean.to_csv(f"{path_results}/df_all_fovs_{group_name}_corrected.csv.gz", compression="gzip", index = False)
    print(f"Written df_all_fovs_{group_name}_corrected.csv.gz to {path_results}/")