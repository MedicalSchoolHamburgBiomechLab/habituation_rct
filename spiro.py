from labtools.systems.cosmed.convenience import read_cosmed_excel
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import warnings
import pandas as pd

from common import SESSIONS, get_path_root, PARTICIPANT_IDS


def analyze_file(data, meta, participant_id, session, path_plots=None) -> dict:
    # Placeholder for analysis logic
    meta_id = meta["Nachname"]
    if meta_id != participant_id:
        return dict()
    # rename necessary columns
    wrong_column_names = ["VO2/kg (mL/min/Kg)", "VO2/Kg (mL/min/Kg)"]
    if any([wcn in data.keys() for wcn in wrong_column_names]):
        i_wrong = [wcn in data.keys() for wcn in wrong_column_names].index(True)
        data = data.rename(columns={wrong_column_names[i_wrong]: "VO2/kg (mL/min/kg)"})

    # convert time column to actual seconds
    data["t (s)"] = data["t (s)"].dt.total_seconds()

    # conditionally correct VO2/kg (mL/min/kg) values where the "factor-3-error" occurred
    is_adjusted = False
    if data["VO2/kg (mL/min/kg)"].max() < 30:  # This will break after the new cosmed update because they changed it to "VO2/kg" (lowercase kg)...
        is_adjusted = True
        data["VO2/kg (mL/min/kg)"] = data["VO2/kg (mL/min/kg)"] * 3
        data["VCO2 (mL/min)"] = data["VCO2 (mL/min)"] * 3

    marker = data["Marker (---)"]
    marker_end_bout_mask = marker == 0
    marker_start_bout_mask = marker == 1

    n_markers = marker_end_bout_mask.value_counts().get(True)
    #
    # bout_starts = data["t (s)"][marker_start_bout_mask]
    # bout_ends = data["t (s)"][marker_end_bout_mask]
    # n_bouts_expected = 6
    #
    # if n_markers != n_bouts_expected:
    #     # bouts were excluded. Find out which one(s)
    #     bout_missing = n_bouts_expected - n_markers
    #     if bout_missing < 2:
    #         raise ValueError("Unexpected number of bouts missing")
    #     expected_diff_min_seconds = 400
    #     expected_diff_max_seconds = 600
    #     diffs = data["t (s)"][marker_end_bout_mask].diff()
    #     is_within = any((diffs.values < expected_diff_min_seconds) & (diffs.values > expected_diff_max_seconds))
    #     if not is_within:
    #         # easy case: either first n or last n bouts were excluded
    #         avg_bout_duration = diffs.mean()
    #         expected_first_bout_start_max_seconds = 300
    #         expected_last_bout_end_min_seconds = data["t (s)"].max() - 200
    #         first_bout_start = bout_starts.iloc[0]
    #         last_bout_end = bout_starts.iloc[-1]
    #         late_start = first_bout_start > expected_first_bout_start_max_seconds
    #         early_end = last_bout_end < expected_last_bout_end_min_seconds
    #         if late_start and not early_end:
    #             print(f"  -> {bout_missing} bout(s) likely excluded at the start.")
    #             bout_starts = bout_starts - (avg_bout_duration * bout_missing)
    #         else:
    #             print(f"  -> {bout_missing} bout(s) likely excluded at the end.")
    #             bout_starts = bout_starts + (avg_bout_duration * bout_missing)

    # return 0
    plt.close()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x="t (s)", y="VO2/kg (mL/min/kg)")
    average_window_seconds = 180
    keys = ["trial_no", "avg_vo2kg", "avg_vco2kg"]
    res = {k: [] for k in keys}
    for it, t in enumerate(data["t (s)"][marker_end_bout_mask]):
        plt.axvline(t, color="red", linestyle="--", alpha=0.5)
        # go back average_window_seconds and plot average
        start_time = t - average_window_seconds
        end_time = t
        roi_vo2_rel = data.loc[(data["t (s)"] >= start_time) & (data["t (s)"] < t), "VO2/kg (mL/min/kg)"]
        roi_vco2_abs = data.loc[(data["t (s)"] >= start_time) & (data["t (s)"] < t), "VCO2 (mL/min)"]
        roi_vco2_rel = roi_vco2_abs / meta["Gewicht (kg)"]
        avg_rel_vo2 = roi_vo2_rel.mean()
        avg_rel_vco2 = roi_vco2_rel.mean()
        plt.plot([start_time, t], [avg_rel_vo2, avg_rel_vo2], color="red", linestyle="--", alpha=0.5)
        plt.text(t + 10, avg_rel_vo2, f"{avg_rel_vo2:.1f}", color="red", fontsize=8)
        # add regression line and text with the slope
        sns.regplot(x="t (s)", y="VO2/kg (mL/min/kg)", data=data.loc[(data["t (s)"] >= start_time) & (data["t (s)"] < t)], scatter=False, color="red", line_kws={"alpha": 0.5})
        time_series = data["t (s)"][(data["t (s)"] >= start_time) & (data["t (s)"] < t)]
        data_series = data["VO2/kg (mL/min/kg)"][(data["t (s)"] >= start_time) & (data["t (s)"] < t)]
        slope = np.polyfit(time_series, data_series, 1)[0] * 60  # slope in mL/min/kg/min
        plt.text(t + 5, avg_rel_vo2 - 5, f"{slope:.1f} 1/min", color="red", fontsize=8)
        # Print bouts number in the middle of the bout
        n_bout = it + 1
        if participant_id == "HAB39" and session == "pre":
            n_bout += 2  # two bouts were excluded at the start
        plt.text((start_time + t) / 2, 5, str(n_bout), color="black", fontsize=12, ha="center")
        res["trial_no"].append(n_bout)
        res["avg_vo2kg"].append(avg_rel_vo2)
        res["avg_vco2kg"].append(avg_rel_vco2)

    for t in data["t (s)"][marker_start_bout_mask]:
        plt.axvline(t, color="green", linestyle="--", alpha=0.5)
    title = f"{participant_id} - {session}"
    if is_adjusted:
        title += " (VO2/kg adjusted x3)"
    plt.title(title)
    plt.gca().set_ylim(0, 55)

    if path_plots:
        path_plots.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_plots / f"{participant_id}_{session}_vo2kg.png")
    return res


def get_spiro_path_root():
    return get_path_root() / "spiro"


def get_path_spiro_results():
    return get_spiro_path_root() / "results_spiro.xlsx"


def add_shoe_condition(df: pd.DataFrame) -> pd.DataFrame:
    if 'shoe_condition' in df.columns:
        df.drop(columns='shoe_condition', inplace=True, axis=1)
    if 'shoe_condition_long' in df.columns:
        df.drop(columns='shoe_condition_long', inplace=True, axis=1)
    path_data_root = get_path_root()
    # The correct shoe order is actually in "borg_lactate.xlsx"
    path_shoe_order_pre = path_data_root / "borg_lactate.xlsx"
    df_shoe_order_pre = pd.read_excel(path_shoe_order_pre, sheet_name="PRE")
    df_shoe_order_post = pd.read_excel(path_shoe_order_pre, sheet_name="POST")
    # merge pre and post shoe order dataframes, remove nan rows
    cols = ["participant_id", "session", "trial_no", "shoe_condition", "is_aft"]
    df_shoe_order_pre = df_shoe_order_pre[cols].dropna(axis=0, how='any')
    df_shoe_order_post = df_shoe_order_post[cols].dropna(axis=0, how='any')
    df_shoe_order = pd.concat([df_shoe_order_pre, df_shoe_order_post], ignore_index=True)
    # merge columns shoe_condition and is_aft into one
    df_shoe_order['is_aft'] = df_shoe_order['is_aft'].astype(bool)
    def add_shoe_condition_long(row):
        if row['shoe_condition'] == "AFT":
            return "AFT"
        elif row['shoe_condition'] == "NonAFT":
            return "NonAFT"
        elif row['shoe_condition'] == "INT" and row['is_aft']:
            return "INT (AFT)"
        elif row['shoe_condition'] == "INT" and not row['is_aft']:
            return "INT (Non AFT)"
        else:
            return row['shoe_condition']
    df_shoe_order['shoe_condition_long'] = df_shoe_order.apply(add_shoe_condition_long, axis=1)

    # df_long = (df_shoe_order_pre.melt(id_vars='participant_id', var_name='trial_no', value_name='shoe_condition')
    #            .assign(trial_no=lambda d: d['trial_no'].str.extract(r'(\d+)').astype(int))
    #            .sort_values(['participant_id', 'trial_no'])
    #            .reset_index(drop=True))
    # df_long['session'] = 'pre'
    # convert trial_no to int
    df_shoe_order['trial_no'] = df_shoe_order['trial_no'].astype(int)
    # convert lower case to upper case in session column
    df['session'] = df['session'].str.upper()
    # merge with existing df on session, participant_id, trial_no skipping missing trial_no in df
    df_merged = pd.merge(df, df_shoe_order, on=['participant_id', 'session', 'trial_no'], how='inner')
    # df_merged["participant_id"].value_counts().max().nunique()
    # reorder:
    df_merged = df_merged[['participant_id', 'session', 'trial_no', 'shoe_condition', 'shoe_condition_long', 'avg_vo2kg', 'avg_vco2kg']]
    # df_merged = df_merged.rename(columns={"shoe": "shoe_condition"})
    return df_merged


def load_df_spiro():
    path_spiro_results = get_path_spiro_results()
    df = pd.read_excel(path_spiro_results)
    return df


def safe_path_spiro_results(df_spiro_results):
    path_spiro_results = get_path_spiro_results()
    df_spiro_results.to_excel(path_spiro_results, index=False)


def calc_spiro_metrics() -> pd.DataFrame:
    path_spiro_root = get_spiro_path_root()
    path_plots = path_spiro_root / "plots"

    wrong_id_counter = 0
    df_out = pd.DataFrame()
    for participant_id, session in product(PARTICIPANT_IDS, SESSIONS):
        print(f"Analyzing {participant_id} {session}")
        path_participant = path_spiro_root / "raw" / participant_id / session.upper()
        spiro_files = list(path_participant.glob("*.xlsx"))
        if not spiro_files:
            print(f"No file found for {participant_id} {session}")
            continue
        spiro_file = spiro_files[0]
        print(f"Processing {spiro_file.stem}")
        try:
            data, meta = read_cosmed_excel(spiro_file)
        except Exception as e:
            warnings.warn(f"Could not read {spiro_file}: {e}")
            continue
        values_dict = analyze_file(data, meta, participant_id, session)  # , path_plots=path_plots)
        if not values_dict:
            warnings.warn(f"No values returned from analyze_file for {participant_id} {session}. Possibly wrong ID?")

        df_temp = pd.DataFrame(values_dict)
        df_temp["participant_id"] = participant_id
        df_temp["session"] = session

        # add "bout_no" column and add res values
        df_out = pd.concat([df_out, df_temp], ignore_index=True)
    return df_out




def get_demographics():
    path_data_root = get_path_root()
    path_demographics = path_data_root / "demographics_session_info.xlsx"
    df_demo = pd.read_excel(path_demographics, nrows=70)

    df_out = df_demo[['participant_id', 'sex', 'DOB']]
    df_out['sex'][df_out['sex'] == "w"] = 'f'  # recode 'w' to 'f'
    # add running speed column based on sex column
    df_out["speed"] = df_demo["sex"].map({"f": 12.0, "m": 14.0, "w": 12.0})
    return df_out


def peronnet_massicotte_1991(VO2, VCO2):
    """Table of nonprotein respiratory quotient: an update. Peronnet F, Massicotte D. Can J Sport Sci. 1991;16(
    1):23-29.
     VO2 and VCO2 required in L/s"""
    return 16.89 * VO2 + 4.84 * VCO2


def add_running_economy(df_spiro, df_demo) -> pd.DataFrame:
    df_eco = pd.merge(df_spiro, df_demo, on=['participant_id'], how='left')
    # add oxygen cost of transport (ml/kg/km)
    df_eco['ocot'] = df_eco['avg_vo2kg'] / (df_eco['speed'] / 60)
    # add energetic cost in W/kg according to Peronnet & Massicotte 1991
    df_eco['energetic_cost_W_kg'] = peronnet_massicotte_1991(df_eco['avg_vo2kg'] / 60000, df_eco['avg_vco2kg'] / 60000) * 1000
    # add energetic cost of transport in J/kg/m
    df_eco['ecot'] = df_eco['energetic_cost_W_kg'] / (df_eco['speed'] / 3.6)
    return df_eco


if __name__ == '__main__':
    RECALC = False
    if RECALC:
        df_spiro = calc_spiro_metrics()
        safe_path_spiro_results(df_spiro)
    else:
        df_spiro = load_df_spiro()
    df_spiro = add_shoe_condition(df_spiro)
    # safe_path_spiro_results(df_merged)
    df_demographics = get_demographics()
    df_spiro = add_running_economy(df_spiro, df_demographics)
    safe_path_spiro_results(df_spiro)
