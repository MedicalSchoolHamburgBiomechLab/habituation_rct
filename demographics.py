import pandas as pd

from common import get_demographics_session_info, get_path_root


def make_demographics_master():
    df_info = get_demographics_session_info()
    foo = 1
    return


def get_demographics_master():
    path_data_root = get_path_root()
    path_demographics = path_data_root / "demographics_master.xlsx"
    df_demo = pd.read_excel(path_demographics)
    return df_demo


def get_demographics_plus_strava():
    path_data_root = get_path_root()
    path_demographics = path_data_root / "demographics_plus_strava.xlsx"
    df_demo = pd.read_excel(path_demographics)
    return df_demo


def print_disposition(df_demo):
    def print_info(df: pd.DataFrame):
        count_aft = len(df[df["group"] == "AFT"])
        count_non_aft = len(df[df["group"] == "NonAFT"])
        print(f"({count_aft} AFT, {count_non_aft} NonAFT)")
        reasons = df["reason"].value_counts()
        if len(reasons) > 0:
            print("Reasons: ")
            for reason, count in reasons.items():
                print(f"\t{count}x {reason}")
        cols = ["age_session_1", "bmi", "weight_kg", "height_cm"]

    def print_subgroup_info(df_demo: pd.DataFrame, subgroup_disposition: str):
        df_subgroup = df_demo[df_demo["disposition"] == subgroup_disposition]
        print(f"Total number of {subgroup_disposition} cases: {len(df_subgroup)}")
        print_info(df_subgroup)

    def print_count_aft_non_aft(df):
        count_aft = len(df[df["group"] == "AFT"])
        count_non_aft = len(df[df["group"] == "NonAFT"])
        print(f"({count_aft} AFT, {count_non_aft} NonAFT)")

    print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
    print('All')
    print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
    print_info(df_demo)
    print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
    print('Completer')
    print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
    print_subgroup_info(df_demo, "completer")
    print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
    print('Lost To Follow Up')
    print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
    print_subgroup_info(df_demo, "lost_to_follow_up")
    print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
    print('Adherence Violation')
    print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
    print_subgroup_info(df_demo, "adherence_violation")


def get_summary(df_demo: pd.DataFrame):
    def get_param_table_mean(df, param, name, unit):
        m = df[param].mean()
        std = df[param].std()

        group_means = df.groupby("group")[param].mean()
        mean_aft = group_means["AFT"]
        mean_non_aft = group_means["NonAFT"]

        group_stds = df.groupby("group")[param].std()
        std_aft = group_stds["AFT"]
        std_non_aft = group_stds["NonAFT"]

        data = {
            "param_name": f"{name} ({unit})",
            "AFT": f"{mean_aft:.2f} ± {std_aft:.2f}",
            "NonAFT": f"{mean_non_aft:.2f} ± {std_non_aft:.2f}",
            "Total": f"{m:.2f} ± {std:.2f}"}
        df_out = pd.DataFrame(data, index=[0])
        return df_out

    def get_table_count(df):
        count_total = len(df)
        count_aft = len(df[df["group"] == "AFT"])
        count_non_aft = len(df[df["group"] == "NonAFT"])
        data = {
            "param_name": "n",
            "AFT": str(count_aft),
            "NonAFT": str(count_non_aft),
            "Total": str(count_total)
        }
        df_out = pd.DataFrame(data, index=[0])
        return df_out

    def get_param_table_count(df, param=None, name=None, categories=None):
        if param is None:
            return get_table_count(df)

        df[param] = pd.Categorical(df[param], categories=categories)
        counts_total = df[param].value_counts()
        group_counts = df.groupby("group")[param].value_counts()
        counts_aft = group_counts["AFT"]
        counts_non_aft = group_counts["NonAFT"]

        counts_total_str = ""
        counts_aft_str = ""
        counts_non_aft_str = ""
        for cat in categories:
            counts_total_str += str(counts_total[cat]) + "/"
            counts_aft_str += str(counts_aft[cat]) + "/"
            counts_non_aft_str += str(counts_non_aft[cat]) + "/"

        counts_total_str = counts_total_str[:-1]
        counts_aft_str = counts_aft_str[:-1]
        counts_non_aft_str = counts_non_aft_str[:-1]

        data = {
            "param_name": name,
            "AFT": counts_aft_str,
            "NonAFT": counts_non_aft_str,
            "Total": counts_total_str
        }
        df_out = pd.DataFrame(data, index=[0])
        return df_out

        group_counts = df.groupby("group")["sex"].count()

    def get_params_df(df):
        list_print_params = [
            ("age_session_1", "Age", "years"),
            ("weight_kg", "Bodymass", "kg"),
            ("height_cm", "Height", "cm"),
            ("bmi", "BMI", "kg/m^2"),
            ("WA_points_max", "WA Points", ""),
            # strava metrics:
            ("kilometers_all", "Total distance", "km"),
            ("kilometers_intervention", "Distance in intervention shoes", "km"),
            ("distance_intervention_percent", "Proportion distance in intervention shoes", "%"),
            ("number_weeks_intervention", "Number of weeks with at least one run in intervention shoes", ""),
        ]
        df_all = pd.DataFrame()
        for param, name, unit in list_print_params:
            df_param = get_param_table_mean(df, param, name, unit)
            df_all = pd.concat([df_all, df_param], axis=0)

        df_sex = get_param_table_count(df, "sex", "(female/male)", ["f", "m"])
        df_all = pd.concat([df_all, df_sex], axis=0)

        df_count = get_param_table_count(df)
        df_all = pd.concat([df_all, df_count], axis=0)

        df_all.set_index("param_name", inplace=True)

        return df_all

    # all
    df_summary_all = get_params_df(df_demo)
    # modified intention-to-tread
    df_mitt = df_demo[df_demo["disposition"] != "lost_to_follow_up"]
    df_summary_mitt = get_params_df(df_mitt)
    # lost to follow up
    df_ltfu = df_demo[df_demo["disposition"] == "lost_to_follow_up"]
    df_summary_ltfu = get_params_df(df_ltfu)
    # non adherence
    df_adherence_viol = df_demo[df_demo["disposition"] == "adherence_violation"]
    df_summary_adherence_viol = get_params_df(df_adherence_viol)
    # completer
    df_completer = df_demo[df_demo["disposition"] == "completer"]
    df_summary_completer = get_params_df(df_completer)

    # dropouts
    df_dropouts = df_demo[df_demo["disposition"] != "completer"]
    df_summary_dropouts = get_params_df(df_dropouts)

    d = {"Completer": df_summary_completer, "Dropouts": df_summary_dropouts, "All": df_summary_all, "mITT": df_summary_mitt}
    df_summary = pd.concat(d.values(), axis=1, keys=d.keys())

    return df_summary


if __name__ == '__main__':
    df_demo = get_demographics_master()
    df_demo_plus_strava = get_demographics_plus_strava()

    df = df_demo_plus_strava
    print_disposition(df)
    df_summary = get_summary(df)
    path_root = get_path_root()
    path_summary = path_root / "demographics_summary_plus_strava.xlsx"
    df_summary.to_excel(path_summary)
