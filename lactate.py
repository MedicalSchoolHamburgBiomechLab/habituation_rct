import pandas as pd

from common import get_path_root



if __name__ == '__main__':
    path_root= get_path_root()
    path_lactate_file = path_root / "borg_lactate.xlsx"


    df_lactate_pre = pd.read_excel(path_lactate_file, sheet_name="PRE")
    df_lactate_pre.dropna(subset=["shoe_condition"], inplace=True)
    df_lactate_post= pd.read_excel(path_lactate_file, sheet_name="POST")
    df_lactate_post.dropna(subset=["shoe_condition"], inplace=True)

    df_lactate = pd.concat([df_lactate_post, df_lactate_pre], axis=0)
    df_lactate.sort_values(by=["participant_id", "session", "trial_no"], inplace=True)
    df_lactate['trial_no'] = df_lactate['trial_no'].astype(int)

    # remove unnecessary cols:
    df_lactate.drop(columns=["shoe_condition", "is_aft", "comments"], inplace=True)

    new_filename = f"{path_lactate_file.stem}_clean.xlsx"
    path_new = path_root / new_filename
    df_lactate.to_excel(path_new, index=False)
    foo = 1