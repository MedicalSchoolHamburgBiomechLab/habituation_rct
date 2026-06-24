from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from os import getenv

PARTICIPANT_IDS = [f"HAB{i:02d}" for i in range(1, 71)]
SESSIONS = ["pre", "post"]
DROPOUTS = ["HAB01", "HAB02", "HAB05",
            "HAB14", "HAB15", "HAB17",
            "HAB24", "HAB25", "HAB26", "HAB27", "HAB29",
            "HAB31", "HAB35", "HAB37", "HAB38",
            "HAB41", "HAB42", "HAB43", "HAB46", "HAB48",
            "HAB51", "HAB56", "HAB57", "HAB59",
            "HAB62", "HAB63", "HAB65", "HAB68"]


def get_path_root() -> Path:
    load_dotenv()

    path_data_root = getenv("PATH_DATA_ROOT")
    if path_data_root is None:
        raise EnvironmentError("PATH_DATA_ROOT is not set. Copy .env.example to .env and set the path.")
    path = Path(path_data_root)
    if not path.exists():
        raise FileNotFoundError(f"PATH_DATA_ROOT points to non-existent path: {path}")
    return path


path_data_root = get_path_root()


def get_demographics_session_info() -> pd.DataFrame:
    path_data_root = get_path_root()
    path_demographics = path_data_root / "demographics_session_info.xlsx"
    return pd.read_excel(path_demographics)


def get_shoe_sequence_df():
    path_root = get_path_root()
    path_master_out = path_root / "shoe_sequence_master.xlsx"
    df = pd.read_excel(path_master_out)
    df.drop(columns=["shoe_condition", "Unnamed: 5"], inplace=True)
    df.rename(columns={"true_condition": "shoe_condition"}, inplace=True)
    return df