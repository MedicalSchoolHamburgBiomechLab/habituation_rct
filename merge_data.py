import pandas as pd

from common import get_path_root, get_shoe_sequence_df
from spiro import get_demographics, get_spiro_path_root


def add_shoe_condition_by_trial_no(df):
    # use the shoe/trial sequence master table to add the shoe condition to each trial
    df_shoe_sequence = get_shoe_sequence_df()
    df = merge_new_dataframe(df_shoe_sequence, df)
    first_cols = ["participant_id", "session", "trial_no", "shoe_condition"]
    df = df[first_cols + [c for c in df.columns if c not in first_cols]]
    return df



def add_demographics(df):
    # add demographic data that will be used as covariates for the analysis
    df_demographics = get_demographics()
    add_columns = ["participant_id", "disposition", "height_cm", "weight_kg", "bmi", "age_session_1", "sex", "group"]
    df_demographics = df_demographics[add_columns]
    df_demographics.rename(columns={"age_session_1": "age"}, inplace=True)
    return df.merge(df_demographics, on="participant_id", how="left")


def get_lactate_df():
    path_root = get_path_root()
    path_lactate_file = path_root / "borg_lactate_clean.xlsx"
    return pd.read_excel(path_lactate_file)


def merge_new_dataframe(df_new, df_merged) -> pd.DataFrame:
    # merge multiple frames from different data sources based on participan_id, session, and trial number
    df_merged = df_merged.merge(df_new, on=["participant_id", "session", "trial_no"], how="left")
    return df_merged


def load_pressure_data_with_trial_no():
    path_data_root = get_path_root()
    path_pressure_with_trial_no = path_data_root / "pressure" / "results_pressure_trial_no.xlsx"
    df_pressure = pd.read_excel(path_pressure_with_trial_no)
    if "participant" in df_pressure.columns:
        df_pressure.rename(columns={"participant": "participant_id"}, inplace=True)
    if "trial" in df_pressure.columns:
        df_pressure.drop(columns="trial", inplace=True)
    return df_pressure

def add_lactate_borg(df):
    df_lactate = get_lactate_df()
    if "comments" in df_lactate.columns:
        df_lactate = df_lactate.drop(columns=["comments"])
    return merge_new_dataframe(df_lactate, df)

def load_spiro_with_eco():
    path_spiro_root = get_spiro_path_root()

    path_with_eco = path_spiro_root / "results_spiro_plus_eco.xlsx"
    return pd.read_excel(path_with_eco)


def make_repeated_measures_dataframe():
    df_spiro = load_spiro_with_eco()
    df_spiro.drop(columns="file", inplace=True)
    df_pressure = load_pressure_data_with_trial_no()
    # merge main dataframes
    df_merged = merge_new_dataframe(df_pressure, df_spiro)
    # add lactate and borg values
    df_merged = add_lactate_borg(df_merged)
    # add demographic data
    df_merged = add_demographics(df_merged)
    # add shoe condition
    df_merged = add_shoe_condition_by_trial_no(df_merged)

    path_root = get_path_root()
    path_merged = path_root / "merged.xlsx"
    df_merged.to_excel(path_merged)

def merge_strava_and_demographics():
    df_demo = get_demographics()
    path_root = get_path_root()
    path_df_strava = path_root / "strava" / 'metrics.xlsx'
    df_strava = pd.read_excel(path_df_strava)
    if "participant" in df_strava.columns:
        df_strava.rename(columns={"participant": "participant_id"}, inplace=True)

    df_merged = df_strava.merge(df_demo, on=["participant_id"], how="left")
    file_out = path_root / "demographics_plus_strava.xlsx"
    df_merged.to_excel(file_out, index=False)


def load_kinematics_data():
    path_root = get_path_root()
    path_kinematics = path_root / "kinematics" / "results_kinematics.xlsx"
    df_kinematics = pd.read_excel(path_kinematics)
    if "participant" in df_kinematics.columns:
        df_kinematics.rename(columns={"participant": "participant_id"}, inplace=True)
    if "trial" in df_kinematics.columns:
        df_kinematics.drop(columns="trial", inplace=True)
    return df_kinematics

def make_kinematics_dataframe():
    df_kinematics = load_kinematics_data()
    df_kinematics.dropna(subset=["trial_no"], inplace=True)
    df_kinematics.drop(columns="filename", inplace=True)

    # add demographic data
    df_merged = add_demographics(df_kinematics)
    # add shoe condition
    df_merged = add_shoe_condition_by_trial_no(df_merged)

    path_root = get_path_root()
    path_merged = path_root / "kinemaics_merged.xlsx"
    df_merged.to_excel(path_merged)


if __name__ == '__main__':
    make_kinematics_dataframe()