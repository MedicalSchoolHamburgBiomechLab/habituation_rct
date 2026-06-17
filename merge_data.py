import pandas as pd

from common import get_path_root, path_data_root
from spiro import get_demographics, load_df_spiro

def load_pressure_data():
    path_df_out = path_data_root / "pressure" / "results.xlsx"
    df_pressure = pd.read_excel(path_df_out)
    if "participant" in df_pressure.columns:
        df_pressure.rename(columns={"participant": "participant_id"}, inplace=True)

    return df_pressure

if __name__ == '__main__':
    df_spiro = load_df_spiro()
    df_pressure = load_pressure_data()
    df_demographics = get_demographics()
    add_columns = ["participant_id", "disposition"]
    df_demographics = df_demographics[add_columns]
    df_merged = df_spiro.merge(df_demographics, on="participant_id", how="left")
    # add pressure data
    df_pressure = load_pressure_data()
    df_merged = df_merged.merge(df_pressure, on="participant_id", how="left")
    print(df_merged.head())

    path_root = get_path_root()
    path_merged = path_root / "merged.xlsx"
    df_merged.to_excel(path_merged)