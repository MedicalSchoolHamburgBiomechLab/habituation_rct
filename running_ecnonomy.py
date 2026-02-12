from pathlib import Path

from common import PARTICIPANT_IDS, get_demographics, get_path_root
from spiro import get_spiro_path_root, load_df_spiro

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


def make_bar_graph(df_participant: pd.DataFrame, participant_id: str, path_plot: Path):
    # make a bargraph for each participant with average values of the
    # two measurements (per session, currently only PRE) for the three shoe conditions
    df_avg = df_participant.groupby("shoe_condition_long").agg(avg_vo2kg=('avg_vo2kg', 'mean'),
                                                          ocot=('ocot', 'mean')).reset_index()
    vO2_ref = df_avg['avg_vo2kg'][df_avg['shoe_condition_long'] == 'NonAFT'].values
    df_avg['rel_avg_vo2kg'] = df_avg['avg_vo2kg'] / vO2_ref * 100 - 100
    fig, ax = plt.subplots(figsize=(10, 6))
    lst = df_avg['shoe_condition_long'].unique()
    order = ['AFT', 'NonAFT']
    order = sorted(lst, key=lambda x: order.index(x) if x in order else len(order))
    sns.barplot(ax=ax, data=df_participant, x='shoe_condition_long', y='avg_vo2kg', hue='session', errorbar=("pi", 100),
                order=order)
    ax.set_ylabel('Average VO2/kg (mL/min/kg)')
    ax.set_xlabel('')
    ax.set_ylim(df_participant['avg_vo2kg'].min() - 1, df_participant['avg_vo2kg'].max() + 1)
    # print the relative difference of vo2kg compared to NonAFT above each bar
    y_pos = ax.get_ylim()[0] + 0.5
    for _, row in df_avg.iterrows():
        # get the correct x position for the current shoe condition (they might be in different order)
        x_pos = [text_obj.get_text() for text_obj in ax.get_xticklabels()].index(row['shoe_condition_long'])
        ax.text(x_pos, y_pos, f"%-Diff: {row['rel_avg_vo2kg']:.1f}%", ha='center', va='bottom', fontsize=14, fontweight='bold', color='black')
    plt.title(f'Average VO2/Kg for {participant_id}')
    plt.savefig(path_plot / f"{participant_id}_bargraph_vo2kg.png", dpi=200)
    plt.close()


def main():
    df = load_df_spiro()
    path_plot = get_spiro_path_root() / "plots" / "bar_graphs"
    path_plot.mkdir(parents=True, exist_ok=True)
    for participant_id in PARTICIPANT_IDS:
        df_participant = df[df['participant_id'] == participant_id]
        if df_participant.empty:
            print(f"No data for participant {participant_id}")
            continue
        make_bar_graph(df_participant, participant_id, path_plot)


if __name__ == '__main__':
    main()
