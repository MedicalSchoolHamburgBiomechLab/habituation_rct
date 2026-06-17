from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from labtools.batch_processor import BatchProcessor

from common import get_path_root


def _unwrap_matlab_scipy_result(x):
    """Flatten one MATLAB cell value into a scalar (or NaN if empty)."""
    arr = np.asarray(x).ravel()
    return arr[0] if arr.size else np.nan


def read_metrics_mat(row: pd.Series) -> dict:
    """Read selected gait metrics from the V3D Running Report mat-file exports.

    Cells with mixed singleton/empty entries are unwrapped to plain
    floats, then averaged across trials within the file.
    """
    data = loadmat(str(row.path))
    params = [
        "Stride_Length_Mean",
        "Stride_Width_Mean",
        "Steps_Per_Minute_Mean",
        "Stance_Time_Mean_MEAN",
        "Flight_Time_Mean_MEAN",
        "Pelvis_Height_RANGE_cm_MEAN",
    ]
    out: dict = {}
    for p in params:
        values = data.get(p)
        if values is None:
            out[p] = np.nan
            continue
        out[p] = np.nanmean([_unwrap_matlab_scipy_result(cell) for cell in values.ravel()])
    return out


if __name__ == "__main__":
    path_data_root = get_path_root()
    path_root = path_data_root / "kinematics" / "mat"

    bp = BatchProcessor(
        path_root=path_root,
        file_pattern="metrics.mat",
        level_names=["participant", "session", "condition", "trial"],
        allow_existing_output=True,
    )

    print("Index summary:", bp.summary())
    print("Skipped files:", bp.skipped)

    results = bp.apply(read_metrics_mat, multiprocess=False)

    metrics_df = pd.json_normalize(results)
    df = pd.concat([bp.index.reset_index(drop=True), metrics_df], axis=1)

    out_path = path_root.parent / "metrics.xlsx"
    df.to_excel(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")
    print(df.head())
    print("Errors:", bp.errors)
