from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from os import getenv

PARTICIPANT_IDS = [f"HAB{i:02d}" for i in range(1, 71)]
SESSIONS = ["pre", "post"]
DROPOUTS = ["HAB01", "HAB02", "HAB25", "HAB29", "HAB31", "HAB35", "HAB38", "HAB46", "HAB57", "HAB62", "HAB65", "HAB68"]


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


def get_demographics() -> pd.DataFrame:
    path_data_root = get_path_root()
    path_demographics = path_data_root / "demographics_session_info.xlsx"
    return pd.read_excel(path_demographics)
