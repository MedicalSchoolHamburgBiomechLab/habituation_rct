import re
import warnings
from itertools import product
from pathlib import Path

import pandas as pd
from labtools.batch_processor import BatchProcessor
from labtools.systems.zebris.spatio_temnporal_parameters import analyze
from matplotlib import pyplot as plt

from common import get_path_root, path_data_root

CONDITION_PATTERN = re.compile(r"_Laufanalyse_(AFT|NonAFT|INT)_")


def extract_condition(trial: str) -> str | None:
    m = CONDITION_PATTERN.search(trial)
    return m.group(1) if m else None


def process_trials(row: pd.Series) -> dict:
    out = analyze(row.path)
    return out


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

    results = bp.apply(process_trials,
                       multiprocess=True)
    df_results = pd.json_normalize(results)
    df_dict = pd.concat([bp.index.reset_index(drop=True), df_results], axis=1)
    df_dict.drop(columns=["path"], inplace=True)

    save_pressure_data(df_dict)

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
    return path_data_root / "pressure" / "results.xlsx"


def load_pressure_data():
    path_df_out = get_pressure_data_path(path_data_root)
    df_pressure = pd.read_excel(path_df_out)
    if "participant" in df_pressure.columns:
        df_pressure.rename(columns={"participant": "participant_id"}, inplace=True)

    return df_pressure


def save_pressure_data(df_pressure: pd.DataFrame):
    path_df_out = get_pressure_data_path(path_data_root)
    df_pressure.to_excel(path_df_out, index=False)


def add_trial_column(df: pd.DataFrame):
    if "trial_no" in df.columns:
        warnings.warn("Data already contains trial_no column")
        return df
    # add trial column based on the time stamp in the "trial" name
    df['trial_no'] = None  # initialize None
    # do per participant per session
    participants = df['participant_id'].unique()
    sessions = df['session'].unique()
    for p_id, session in product(participants, sessions):
        mask = (df['participant_id'] == p_id) & (df['session'] == session)
        dt = pd.to_datetime(
            df.loc[mask, 'trial'].str.split("_").str[0],
            format="%Y-%m-%d-%H-%M"
        )
        df.loc[mask, 'trial_no'] = dt.rank(method="first").astype(int)

    return df


if __name__ == "__main__":
    process()
    # analyze_results()
    df_pressure = load_pressure_data()

    df_pressure = add_trial_column(df_pressure)
    df_pressure.sort_values(by=['participant_id', 'session', 'trial_no'], inplace=True)
    # reorder cols
    first_cols = ["participant_id", "session", "trial_no", "condition"]
    df_pressure = df_pressure[first_cols+[c for c in df_pressure.columns if c not in first_cols]]




    save_pressure_data(df_pressure)
