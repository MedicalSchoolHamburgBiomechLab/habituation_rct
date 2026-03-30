from pathlib import Path
import pandas as pd

if __name__ == '__main__':
    path_root = Path(r"C:\Users\dominik.fohrmann\OneDrive - MSH Medical School Hamburg - University of Applied Sciences and Medical University\Dokumente\Projects\AFT_Habituation_2\data\kinematics\reports")
    folders = list(path_root.glob("*"))
    cols = ["participant_id", "pre_AFT", "pre_NonAFT", "pre_INT", "post_AFT", "post_NonAFT", "post_INT"]
    df_out = pd.DataFrame(columns=cols)
    # pre allocate a row for each participant with participant_id and empty values for the other columns
    df_out["participant_id"] = [folder.name for folder in folders]
    for folder in folders:
        p_id = folder.name
        print(p_id)
        sessions = list(folder.glob("*"))
        for session in sessions:
            print(f"  {session.name}")
            conditions = list(session.glob("*"))
            for condition in conditions:
                print(f"    {condition.name}")
                sess = session.name.lower()
                cond = condition.name
                col = f"{sess}_{cond}"
                if col not in cols:
                    print(f"Warning: column {col} not in expected columns {cols}")
                    continue
                files = list(condition.glob("*.cmz"))
                n_files = len(files)
                df_out.loc[df_out["participant_id"] == p_id, col] = n_files
    # save dataframe to excel
    path_out = path_root / "report_files_check.xlsx"
    df_out.to_excel(path_out, index=False)
