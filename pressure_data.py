import re
from pathlib import Path

import pandas as pd
from labtools.batch_processor import BatchProcessor
from labtools.systems.zebris.spatio_temnporal_parameters import analyze
from matplotlib import pyplot as plt

from common import get_path_root

CONDITION_PATTERN = re.compile(r"_Laufanalyse_(AFT|NonAFT|INT)_")


def extract_condition(trial: str) -> str | None:
    m = CONDITION_PATTERN.search(trial)
    return m.group(1) if m else None


def process_trials(path: Path) -> dict:
    out = analyze(path)
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

    results = bp.apply(process_trials, multiprocess=True)
    df_results = pd.json_normalize(results)
    df_dict = pd.concat([bp.index.reset_index(drop=True), df_results], axis=1)

    path_df_out = path_data_root / "pressure" / "results.xlsx"
    df_dict.to_excel(path_df_out)

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


if __name__ == "__main__":
    # process()
    analyze_results()
