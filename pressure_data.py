import re
import warnings
from itertools import product
from pathlib import Path

import pandas as pd
from labtools.analyses.kinetics.event_detection import get_force_events_treadmill
from labtools.batch_processor import BatchProcessor
from labtools.systems.zebris.spatio_temnporal_parameters import analyze
from labtools.systems.zebris.utils import get_force
from labtools.utils.c3d import load_c3d

from common import get_path_root, get_shoe_sequence_df, path_data_root
from demographics import get_demographics_master

CONDITION_PATTERN = re.compile(r"_Laufanalyse_(AFT|NonAFT|INT)_")


def extract_condition(trial: str) -> str | None:
    m = CONDITION_PATTERN.search(trial)
    return m.group(1) if m else None


def process_trials(row: pd.Series) -> dict:
    out = analyze(row.path)
    return out


def get_events(row: pd.Series) -> dict:
    data, meta = load_c3d(row.path)
    sample_rate = data['analog_rate']
    f_z_r, f_z_l = get_force(data, True)

    evt_r = get_force_events_treadmill(f_z=f_z_r,
                                     sample_rate=sample_rate)
    # convert from np.int to int
    evt_r['ic'] = [int(x) for x in evt_r['ic']]
    evt_r['tc'] = [int(x) for x in evt_r['tc']]
    evt_l = get_force_events_treadmill(f_z=f_z_l,
                                       sample_rate=sample_rate)
    evt_l['ic'] = [int(x) for x in evt_l['ic']]
    evt_l['tc'] = [int(x) for x in evt_l['tc']]

    return {"left": evt_l, "right": evt_r}


def process():
    path_data_root = get_path_root()
    path_root = path_data_root / "pressure" / "raw"

    bp = BatchProcessor(
        path_root=path_root,
        file_pattern=".c3d",
        level_names=["participant", "session", "condition", "trial"],
        allow_existing_output=True,
    )
    print("Skipped files:", bp.skipped)
    print(bp.summary())
    print(bp.index.head())

    # foo = 1
    # copy = bp.index.copy()
    # copy["suffix"] = bp.index.apply(lambda row: row.trial.split(row.condition)[-1].split("_")[1], axis=1)
    # copy["numeric"] = pd.to_numeric(copy["suffix"], "coerce")
    # copy["is_numeric"] = pd.to_numeric(copy["suffix"], "coerce").notna()
    #
    # copy[copy["is_numeric"] == False]
    #
    #
    # counts = bp.index.groupby(["participant", "session", "condition"]).size()
    # print(counts[counts > 2])
    #
    # mask = pd.to_numeric(copy["col"], errors="coerce").notna()

    # results_metrics = bp.apply(process_trials,
    #                    multiprocess=True)
    # df_results_metrics = pd.json_normalize(results_metrics)
    # df_dict_metrics = pd.concat([bp.index.reset_index(drop=True), df_results_metrics], axis=1)
    # df_dict_metrics.drop(columns=["path"], inplace=True)
    # save_pressure_data(df_dict_metrics)


    # calculate events and save
    results_events = bp.apply(get_events,
                       multiprocess=True)
    df_results_events = pd.json_normalize(results_events)
    df_dict_events = pd.concat([bp.index.reset_index(drop=True), df_results_events], axis=1)
    df_dict_events.drop(columns=["path"], inplace=True)

    save_events(df_dict_events)


def analyze_results():
    import seaborn as sns
    import matplotlib.pyplot as plt
    path_data_root = get_path_root()
    path_df_results = path_data_root / "pressure" / "results.xlsx"
    df_results = pd.read_excel(path_df_results, index_col=0)
    # plot pressure parameters
    params = ['steps_per_minute', 'contact_time_ms', 'flight_time_ms',
              'normalized_ground_contact_time']
    for param in params:
        print(param)
        fig, ax = plt.subplots()
        sns.boxplot(data=df_results, ax=ax,
                    x='condition',
                    y=param,
                    hue='session'
                    )
        plt.show()


def get_pressure_data_path(path: Path):
    return path_data_root / "pressure" / "results_pressure.xlsx"


def load_pressure_data():
    path_df_out = get_pressure_data_path(path_data_root)
    df_pressure = pd.read_excel(path_df_out)
    if "participant" in df_pressure.columns:
        df_pressure.rename(columns={"participant": "participant_id"}, inplace=True)

    return df_pressure


def save_pressure_data(df_pressure: pd.DataFrame):
    path_df_out = get_pressure_data_path(path_data_root)
    df_pressure.to_excel(path_df_out, index=False)


def save_events(df_events: pd.DataFrame):
    path_data_root = get_path_root()
    path_df_out = path_data_root / "pressure" / "events.xlsx"
    df_events.to_excel(path_df_out, index=False)


# def get_shoe_sequence_df():
#     path_root = get_path_root()
#     path_shoe_sequence_pre = path_root / "balanced_shoe_sequence_PRE.xlsx"
#     df_shoe_sequence_pre = pd.read_excel(path_shoe_sequence_pre, nrows=70)
#     df_shoe_sequence_pre["session"] = "PRE"
#     path_shoe_sequence_post = path_root / "balanced_shoe_sequence_POST.xlsx"
#     df_shoe_sequence_post = pd.read_excel(path_shoe_sequence_post, nrows=70)
#     df_shoe_sequence_post["session"] = "POST"
#
#     df_shoe_sequence = pd.concat([df_shoe_sequence_pre, df_shoe_sequence_post], sort=False)
#
#     pd.melt(df_shoe_sequence, ["Trial_1", "Trial_2", "Trial_3", "Trial_4", "Trial_5", "Trial_6"])
#     df_shoe_sequence_long = pd.melt(df_shoe_sequence, ["participant_id", "session"], var_name="trial_no", value_name="shoe_condition")
#     df_shoe_sequence_long["trial_no"] = df_shoe_sequence_long.apply(lambda row: int(row["trial_no"].split("_")[-1]), axis=1)
#
#     df_shoe_sequence_long.sort_values(by=["participant_id", "session", "trial_no"], ascending=[True, False, True], inplace=True)
#     #
#     # path_master_out = path_data_root / "shoe_sequence_master.xlsx"
#     # df_shoe_sequence_long.to_excel(path_master_out, index=False)
#
#     return df_shoe_sequence_long
#


def _parse(row):
    parts = row.trial.split(row.condition)[-1].split("_")
    return parts[1]


def add_trial_column(df: pd.DataFrame):
    df_shoe_sequence = get_shoe_sequence_df()
    df_shoe_sequence.drop(columns=["shoe_condition", "Unnamed: 5"], inplace=True)
    ## make sure it's ordered correctly:
    df_shoe_sequence.sort_values(by=["participant_id", "session", "trial_no"], inplace=True)
    if "trial_no" in df.columns:
        warnings.warn("Data already contains trial_no column")
        return df
    # add trial column based on the time stamp in the "trial" name
    # df['trial_no'] = None  # initialize None
    # do per participant per session
    participants = df['participant_id'].unique()
    sessions = df['session'].unique()
    results = []
    for p_id, session in product(participants, sessions):
        if (p_id == "HAB13") & (session == "POST"):
            print("stop")
        df_sub = df[(df['participant_id'] == p_id) & (df['session'] == session)]
        if df_sub.empty:
            continue
        df_shoe_sequence_sub = df_shoe_sequence[(df_shoe_sequence['participant_id'] == p_id) & (df_shoe_sequence['session'] == session)]

        # create sortable datetime column from the filename (lives in 'trial')
        df_sub["dt"] = pd.to_datetime(
            df_sub['trial'].str.split("_").str[0],
            format="%Y-%m-%d-%H-%M"
        )
        df_sub_ordered = df_sub.sort_values(by=["dt"], ascending=True)
        # add a "file_no" column
        df_sub_ordered["file_no"] = df_sub_ordered["dt"].rank(method="first").astype(int)

        # add a column for the occurrence (1,2) of each
        df_sub_ordered["condition_occurrence"] = df_sub_ordered.groupby('condition').cumcount() + 1

        # occurrence according to the file name
        print(p_id, session)
        try:
            df_sub_ordered["condition_occurrence_alt"] = df_sub_ordered.apply(lambda row: row.trial.split(row.condition)[-1].split("_")[1], axis=1).astype(int)
        except:
            foo = 1
        mask = df_sub_ordered["condition_occurrence_alt"] != df_sub_ordered["condition_occurrence"]
        if any(mask):
            df_sub_ordered.loc[mask, "condition_occurrence"] = df_sub_ordered.loc[mask, "condition_occurrence_alt"]

        # same for the master table:
        df_shoe_sequence_sub["condition_occurrence"] = df_shoe_sequence_sub.groupby('true_condition').cumcount() + 1

        # # merge on the new condition occurrence column AND condition
        # df_shoe_sequence_sub.rename(columns={"shoe_condition": "condition"}, inplace=True)
        df_shoe_sequence_sub.rename(columns={"true_condition": "condition"}, inplace=True)

        df_merged = pd.merge(df_shoe_sequence_sub, df_sub_ordered, on=["participant_id", "session", "condition", "condition_occurrence"], how="left")
        results.append(df_merged)
    df_out = pd.concat(results, ignore_index=True)
    df_out.drop(columns=["dt", "file_no", "condition_occurrence", "condition_occurrence_alt", "condition"], inplace=True)
    df_out.dropna(subset="trial", inplace=True)

    return df_out


def check_datetime_stamp(df_pressure, df_demographics):
    # add "session_date_per_trial_name" column to pressure df
    df_pressure["session_date_per_trial_name"] = pd.to_datetime(
        df_pressure['trial'].str.split("_").str[0],
        format="%Y-%m-%d-%H-%M"
    ).dt.date

    # add session_date_per_demographics_table column
    def get_demographics_date(row):
        if row.session == "PRE":
            return df_demographics.loc[df_demographics["participant_id"] == row.participant_id, "date_session_1"].iloc[0]
        elif row.session == "POST":
            return df_demographics.loc[df_demographics["participant_id"] == row.participant_id, "   date_session_2"].iloc[0]

    df_pressure["session_date_per_demographics_table"] = df_pressure.apply(lambda row: get_demographics_date(row), axis=1)
    df_pressure["session_date_check"] = df_pressure["session_date_per_trial_name"] == df_pressure["session_date_per_demographics_table"]
    df_pressure["session_date_check"].value_counts()

    df_wrong = df_pressure[df_pressure["session_date_check"] == False]


if __name__ == "__main__":
    RECALC = True
    if RECALC:
        process()
    # df_pressure = load_pressure_data()
    # df_demographics = get_demographics_master()
    # # check_datetime_stamp(df_pressure, df_demographics)
    # #
    # df_pressure_trial_no = add_trial_column(df_pressure)
    #
    # # add a "matching" column for testing
    #
    # # df_pressure_trial_no["matching"] = df_pressure_trial_no.apply(lambda row: row.file_no == row.trial_no, axis=1)
    # # df_pressure_trial_no.dropna(subset=["trial"], inplace=True)
    #
    # path_root = get_path_root()
    # path_pressure_trial_no = path_root / "pressure" / "results_pressure_trial_no.xlsx"
    # df_pressure_trial_no.to_excel(path_pressure_trial_no, index=False)
    # # df_pressure.sort_values(by=['participant_id', 'session', 'trial_no'], inplace=True)
    # # # reorder cols
    # # first_cols = ["participant_id", "session", "trial_no", "condition"]
    # # df_pressure = df_pressure[first_cols+[c for c in df_pressure.columns if c not in first_cols]]
    # # save_pressure_data(df_pressure)
