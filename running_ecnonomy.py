from pathlib import Path

from common import PARTICIPANT_IDS, get_demographics, get_path_root
from spiro import get_spiro_path_root, load_df_spiro

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


def make_point_plot(df_participant: pd.DataFrame, participant_id: str, path_plot: Path):
    # make a bargraph for each participant with average values of the
    # two measurements for the three shoe conditions per session,
    # with error bars representing the standard deviation of the two measurements

    # calculate the relative difference between the shoe conditions compared to NonAFT for each session separately
    df_avg = df_participant.groupby(["session", "shoe_condition_long"]).agg(avg_vo2kg=('avg_vo2kg', 'mean'),
                                                                            ocot=('ocot', 'mean')).reset_index()
    vO2_ref_pre = df_avg['avg_vo2kg'][(df_avg['shoe_condition_long'] == 'NonAFT') & (df_avg['session'] == 'PRE')].values[0]
    vO2_ref_post = df_avg['avg_vo2kg'][(df_avg['shoe_condition_long'] == 'NonAFT') & (df_avg['session'] == 'POST')].values[0]
    # make relative values for pre and post separately

    df_avg[df_avg['session'] == 'PRE']['rel_avg_vo2kg'] = df_avg[df_avg['session'] == 'PRE']['avg_vo2kg'] / vO2_ref_pre * 100 - 100

    df_avg.loc[df_avg['session'] == 'PRE', 'rel_avg_vo2kg'] = df_avg.loc[df_avg['session'] == 'PRE', 'avg_vo2kg'] / vO2_ref_pre * 100 - 100
    df_avg.loc[df_avg['session'] == 'POST', 'rel_avg_vo2kg'] = df_avg.loc[df_avg['session'] == 'POST', 'avg_vo2kg'] / vO2_ref_post * 100 - 100

    fig, ax = plt.subplots(figsize=(10, 6))
    # make sure to always plot the shoes in the correct order (NonAFT as reference, then AFT and INT), regardless of the order in the dataframe
    lst = df_avg['shoe_condition_long'].unique()
    int_name = [name for name in lst if "INT" in name][0]
    shoe_order = ['NonAFT', 'AFT', int_name]  # adjust if your INT label differs
    shoe_palette = {'NonAFT': sns.color_palette()[1],  # orange-ish
                    'AFT': sns.color_palette()[0],  # blue
                    int_name: sns.color_palette()[2]}  # green

    sns.pointplot(ax=ax,
                  data=df_participant,
                  x='session',
                  y='avg_vo2kg',
                  hue='shoe_condition_long',
                  hue_order=shoe_order,
                  palette=shoe_palette,
                  errorbar=("pi", 100))
    ax.set_ylabel('Average VO2/kg (mL/min/kg)')
    ax.set_xlabel('')
    ax.set_ylim(df_participant['avg_vo2kg'].min() - 1, df_participant['avg_vo2kg'].max() + 1)
    # print the relative difference of vo2kg compared to NonAFT above each bar
    for _, row in df_avg.iterrows():
        if row['shoe_condition_long'] == 'NonAFT':
            continue
        # get the correct x position for the current shoe condition (they might be in different order)
        x_pos = [text_obj.get_text() for text_obj in ax.get_xticklabels()].index(row['session'])
        x_pos = x_pos - 0.1 if row['session'] == 'PRE' else x_pos + 0.1
        ha = 'right' if row['session'] == 'PRE' else 'left'
        # get the color from the current shoe condition palette
        # blue for AFT, green for INT (NonAFT is not plotted, so we can ignore it)
        try:
            color = shoe_palette[row['shoe_condition_long']]
        except KeyError:
            print(row['shoe_condition_long'], participant_id)
            color = 'black'  # fallback color if shoe condition is not found in palette

        y_pos = row['avg_vo2kg']
        ax.text(x_pos,
                y_pos,
                f"{row['rel_avg_vo2kg']:.1f}%",
                ha=ha,
                va='center',
                fontsize=14,
                fontweight='bold',
                color=color)
    plt.title(f'Average VO2/min/kg for {participant_id}')
    plt.legend(title='Shoe Condition')
    plt.savefig(path_plot / f"{participant_id}_pointplot_vo2kg.png", dpi=200)
    plt.close()


def main():
    df = load_df_spiro()
    path_plot = get_spiro_path_root() / "plots" / "point_plots"
    path_plot.mkdir(parents=True, exist_ok=True)
    for participant_id in PARTICIPANT_IDS:
        df_participant = df[df['participant_id'] == participant_id]
        if df_participant.empty:
            print(f"No data for participant {participant_id}")
            continue
        make_point_plot(df_participant, participant_id, path_plot)


if __name__ == '__main__':
    main()
