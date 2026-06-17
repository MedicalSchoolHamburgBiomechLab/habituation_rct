import pandas as pd

from common import get_path_root, path_data_root
from pressure_data import load_pressure_data
from spiro import get_demographics, load_df_spiro


if __name__ == '__main__':
    df_spiro = load_df_spiro()
    df_pressure = (
        load_pressure_data())
    df_demographics = get_demographics()
    add_columns = ["participant_id", "disposition", "height_cm", "bmi", "age_session_1"]
    df_demographics = df_demographics[add_columns]
    df_demographics.rename(columns={"age_session_1": "age"}, inplace=True)
    df_merged = df_spiro.merge(df_demographics, on="participant_id", how="left")
    # add pressure data
    df_pressure = load_pressure_data()
    df_merged = df_merged.merge(df_pressure, on=["participant_id","session", "trial_no" ], how="left")
    print(df_merged.head())

    path_root = get_path_root()
    path_merged = path_root / "merged.xlsx"
    df_merged.to_excel(path_merged)