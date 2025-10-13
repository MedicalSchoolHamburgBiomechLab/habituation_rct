from pathlib import Path

import pandas as pd

PARTICIPANT_IDS = [f"HAB{i:02d}" for i in range(1, 71)]
SESSIONS = ["pre", "post"]


def get_path_root() -> Path:
    # return Path(r"C:\Users\dominik.fohrmann\OneDrive - MSH Medical School Hamburg - University of Applied Sciences and Medical University\Dokumente\Projects\AFT_Habituation_2\data")
    return Path(r"E:\project\AFT_Habituation_2\data")


def get_demographics() -> pd.DataFrame:
    path_data_root = get_path_root()
    path_demographics = path_data_root / "demographics_session_info.xlsx"
    return pd.read_excel(path_demographics)
