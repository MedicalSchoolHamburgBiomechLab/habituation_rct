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
        data=df[df['shoe_condition']==shoe_group],
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

if __name__ == '__main__':
    main()
