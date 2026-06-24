import pandas as pd

from common import get_path_root
from spiro import load_df_spiro, get_path_spiro_plots
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    print("This is a placeholder for the plotting module.")
    df = load_df_spiro()
    path_plots = get_path_spiro_plots()
    path_plots.joinpath("stats").mkdir(parents=True, exist_ok=True)

    # fig, ax = plt.subplots()
    # sns.histplot(ax=ax, data=df, x="ecot", hue="sex")
    # ax.set_xlabel("ECOT (mL/kg/km)")
    # ax.set_title("ECOT Distribution by Sex")
    #
    # plt.savefig(path_plots / "stats" / "ecot_distribution_by_sex.png", dpi=200)
    # plt.close()

    shoe_group = "AFT"

    ax = sns.catplot(
        data=df[df['shoe_condition'] == shoe_group],
        kind="bar",
        x="int_group",
        y="ecot",
        hue="session",
        errorbar="sd",
        palette="dark",
        alpha=.6,
        height=6
    )
    ax.set_axis_labels("Intervention Group", "ECOT (mL/kg/km)")
    fig = ax.fig
    fig.suptitle(f"ECOT by Intervention Group and Session ({shoe_group} Shoes)")
    # plt.savefig(path_plots / "stats" / "ecot_by_intervention_group_and_session_aft.png", dpi=200)
    # plt.close()

    plt.show()


def make_param_vs_trial_no_plot(df, param_name):
    df = df[df['shoe_condition'] != "INT"]
    fig, ax = plt.subplots()

    sns.boxplot(
        data=df,
        x="trial_no",
        y=param_name,
        # hue="shoe_condition_long",
        hue="session",
    )
    fig.suptitle(f"{param_name} by Trial No")
    path_out = get_path_root()
    path_out = path_out / "plots" / "param_vs_trial_no"
    path_out.mkdir(parents=True, exist_ok=True)
    path_plot = path_out / f"{param_name}_pre-post.png"
    fig.savefig(path_plot)
    plt.close(fig)





def param_plots():
    path = get_path_root()
    path_df = path / "merged.xlsx"
    df = pd.read_excel(path_df)
    print(df.head())
    param_names = [
        # from cpet
        "avg_vo2kg",
        "energetic_cost_W_kg",
        "ocot",
        "avg_vent",
        "avg_hr",
        # from pressure data
        "steps_per_minute",
        "contact_time_ms",
        "flight_time_ms",
        "normalized_ground_contact_time"
    ]
    for param in param_names:
        make_param_vs_trial_no_plot(df, param)


if __name__ == '__main__':
    # main()
    param_plots()
