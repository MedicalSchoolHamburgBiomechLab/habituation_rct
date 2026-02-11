import datetime

from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
import numpy as np

from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from common import PARTICIPANT_IDS, get_demographics, get_path_root
import warnings


def get_path_strava_root() -> Path:
    path_data_root = get_path_root()
    path_strava_root = path_data_root / "strava"
    return path_strava_root


def get_path_strava_raw() -> Path:
    path_data_strava_root = get_path_strava_root()
    path_strava_raw = path_data_strava_root / "raw"
    return path_strava_raw


def get_runs_count(df_strava: pd.DataFrame) -> int:
    # Count the number of running activities in the dataframe
    run_activities = df_strava[df_strava['sport_type'] == 'Run']
    return len(run_activities)


def get_total_running_distance(df_strava: pd.DataFrame) -> float:
    # Calculate the total distance of all runs in kilometers
    df_runs = get_runs_only(df_strava)
    total_distance_m = df_runs['distance'].sum()
    total_distance_km = total_distance_m / 1000  # Convert meters to kilometers
    return total_distance_km


def get_total_running_distance_per_week(df_strava: pd.DataFrame, start_date: pd.Timestamp) -> pd.DataFrame:
    # returns a dataframe with total running distance per week
    df_runs = get_runs_only(df_strava)
    df_runs['start_date_local'] = pd.to_datetime(df_runs['start_date_local'])
    df_runs.set_index('start_date_local', inplace=True)
    try:
        df_weekly = (
            df_runs
            .resample('7D', origin=start_date, label='left', closed='left')  # 7-day bins starting at `anchor`
            .agg(distance=('distance', 'sum'))
            .rename_axis('week_start').
            reset_index()
        )
    except Exception as e:
        print(f"Error in resampling for start_date {start_date}: {e}")
        raise e

    df_weekly['distance_km'] = df_weekly['distance'] / 1000  # Convert meters to kilometers
    return df_weekly[['week_start', 'distance_km']]


def get_intervention_shoe_running_distance_per_week(df_strava: pd.DataFrame, start_date: pd.Timestamp) -> pd.DataFrame:
    # returns a dataframe with total running distance per week in the intervention shoe
    df_runs = get_runs_only(df_strava)
    # Check where column "gear_model_name" has "MSH" in it
    df_msh = df_runs[df_runs['gear_model_name'].str.contains('MSH', na=False)]
    df_msh['start_date_local'] = pd.to_datetime(df_msh['start_date_local'])
    df_msh.set_index('start_date_local', inplace=True)
    df_weekly = (
        df_msh
        .resample('7D', origin=start_date, label='left', closed='left')  # 7-day bins starting at `anchor`
        .agg(distance=('distance', 'sum'))
        .rename_axis('week_start')
        .reset_index()
    )
    df_weekly['distance_km'] = df_weekly['distance'] / 1000  # Convert meters to kilometers
    return df_weekly[['week_start', 'distance_km']]


def get_running_distance_in_msh_footwear(df_strava: pd.DataFrame) -> float:
    # Calculate the total distance of all runs with the intervention shoe in kilometers
    df_runs = get_runs_only(df_strava)
    # Check where column "gear_model_name" has "MSH" in it
    df_msh = df_runs[df_runs['gear_model_name'].str.contains('MSH', na=False)]
    intervention_distance_m = df_msh['distance'].sum()
    intervention_distance_km = intervention_distance_m / 1000  # Convert meters to kilometers
    return intervention_distance_km


def get_runs_only(df_strava: pd.DataFrame) -> pd.DataFrame:
    # Filter the dataframe to include only running activities
    df_runs = df_strava[df_strava['type'] == 'Run'].copy()
    return df_runs


def check_date_range(df_strava: pd.DataFrame, participant_id: str):
    # Check the range of the activities' dates to verify that all activities are present for the 12 week period
    df_strava['start_date_local'] = pd.to_datetime(df_strava['start_date_local'])
    min_date = df_strava['start_date_local'].min()
    max_date = df_strava['start_date_local'].max()
    days_passed = (max_date - min_date).days
    weeks_passed = days_passed / 7


def get_pre_post_dates(participant_id: str) -> (pd.Timestamp, pd.Timestamp):
    # 1. look up the demographics excel and check for the PRE- and POST-Session dates
    df_demo = get_demographics()
    start_date = df_demo[(df_demo['participant_id'] == participant_id) & (df_demo['session'] == 'PRE')]['session_date'].values[0]
    # 1.1 Convert strings to datetime if necessary
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    end_date = df_demo[(df_demo['participant_id'] == participant_id) & (df_demo['session'] == 'POST')]['session_date'].values[0]
    if pd.isna(start_date):
        raise ValueError(f"Missing session dates for participant {participant_id}")
    if pd.isna(end_date):
        # TODO: FOR NOW!!! REMOVE WHEN DATA IS COMPLETE
        # end_date = start_date + pd.Timedelta(weeks=12)
        raise ValueError(f"Missing session dates for participant {participant_id}")
    # check that diff between start and end date is at least 8 weeks
    full_weeks = int((end_date - start_date) / np.timedelta64(1, 'W'))
    if full_weeks < 8:
        raise ValueError(f"Session dates for participant {participant_id} are less than 8 weeks apart ({full_weeks} weeks)")

    return start_date, end_date


def cut_to_study_period(df_strava: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    # cut the dataframe to only include activities between these two dates (inclusive)
    df_strava['start_date_local'] = pd.to_datetime(df_strava['start_date_local'])
    mask = (df_strava['start_date_local'] >= start_date) & (df_strava['start_date_local'] <= end_date)
    df_cut = df_strava[mask].copy()
    return df_cut


def plot_intervention_percentage(df_running_weekly: pd.DataFrame, participant_id: str, path_out):
    # expects columns: week_start, distance_km_total, distance_km_intervention, percentage_intervention (0-100)
    df = df_running_weekly.copy()
    df = df.sort_values('week_start')
    df['distance_km_other'] = (df['distance_km_total'] - df['distance_km_intervention']).clip(lower=0)

    x = pd.to_datetime(df['week_start'])
    width = np.min(np.diff(x.values).astype('timedelta64[D]').astype(float), initial=7.0) * 0.6 if len(x) > 1 else 4.0
    width = np.clip(width, 2.0, 6.0)  # days, just to keep bars readable

    fig, ax = plt.subplots(figsize=(12, 6))

    # stacked bars: other + intervention
    ax.bar(x, df['distance_km_other'], width=width, align='center', label='Other Shoes (km)')
    ax.bar(x, df['distance_km_intervention'], width=width, align='center', bottom=df['distance_km_other'], label='Intervention Shoes (km)')

    ax.set_ylabel('Distance (km)')

    # right axis: percentage line
    ax2 = ax.twinx()
    ax2.plot(x, df['percentage_intervention'], marker='o', linewidth=2, label='Intervention (%)', color='black', zorder=5)
    ax2.set_ylabel('Percentage (%)')
    ax2.set_ylim(-5, 105)

    # RECTANGULAR PATCHES (fix): use fill_between over date spans (axvspan can't limit y-range)
    # weeks 0–3:10–15%, 3–6:15–20%, 6–9:20–25%, 9–12:25–30%
    start_date = pd.to_datetime(df['week_start'].min()).normalize()
    patch_specs = [(0, 3, 10, 15),
                   (3, 6, 15, 20),
                   (6, 9, 20, 25),
                   (9, 12, 25, 30)]
    for w0, w1, pmin, pmax in patch_specs:
        x0 = start_date + pd.to_timedelta(w0, unit='W')
        x1 = start_date + pd.to_timedelta(w1, unit='W')
        ax2.fill_between([x0, x1], [pmin, pmin], [pmax, pmax], alpha=0.18, zorder=4, color='gray')
        xm = x0 + (x1 - x0) / 2
        ax2.text(xm, (pmin + pmax) / 2, f"{pmin}-{pmax}%", ha='center', va='center', fontsize=8, zorder=4)

    # x-axis formatting
    locator = AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)

    # combined legend
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc='upper left', frameon=False)

    ax.set_title(f'Weekly Distance & Intervention Share — {participant_id}')
    fig.tight_layout()
    fig.savefig(path_out / f"{participant_id}_running_summary.png", dpi=200)
    plt.close(fig)


def main(participant_id: str = None):
    path_strava_raw = get_path_strava_raw()
    print(f"Strava data root path: {path_strava_raw}")
    if participant_id is not None:
        p_ids = [participant_id]
    else:
        p_ids = PARTICIPANT_IDS
    for participant_id in p_ids:
        path_participant = path_strava_raw / participant_id
        if not path_participant.exists():
            warnings.warn(f"Participant path does not exist: {path_participant}")
            continue
        strava_files = list(path_participant.glob("*.xlsx"))
        if len(strava_files) != 1:
            print(f"Expected one strava file for {participant_id}, found {len(strava_files)}")
            continue
        df_strava = pd.read_excel(strava_files[0])
        start_date, end_date = get_pre_post_dates(participant_id)
        #
        #
        # Make a new excel sheet per participant, containing only runs from 12 weeks prior to study start data until study end date
        #
        #
        df_runs_only = get_runs_only(df_strava)
        if len(df_runs_only) == 0:
            print(f"No running activities found for participant {participant_id}")
            continue
        pre_study_check_date = start_date - pd.Timedelta(weeks=12)
        df_reduced = cut_to_study_period(df_runs_only, pre_study_check_date, end_date)
        columns = ['start_date_local', 'name', 'distance', 'moving_time', 'type', 'gear_model_name']
        df_reduced = df_reduced[columns].copy()
        # add the absolute week number (always starting on monday) to the dataframe, relative to the study start date (study start date week is week 1)
        df_reduced['week_number'] = ((df_reduced['start_date_local'] - start_date) / np.timedelta64(1, 'W')).apply(np.ceil).astype(int)
        # add a column indicating whether the run was done in the intervention shoe
        df_reduced['in_intervention_shoe'] = df_reduced['gear_model_name'].str.contains('MSH', na=False)
        # save to excel
        filename_out = f"{participant_id}_runs_study_period.xlsx"
        path_out = get_path_strava_root() / "runs_only"
        path_out.mkdir(parents=True, exist_ok=True)
        df_reduced.to_excel(path_out / filename_out, index=False)

        #
        #
        #
        #

        # df_strava = cut_to_study_period(df_strava, start_date, end_date)
        # running_km_total = get_total_running_distance(df_strava)
        # running_km_intervention = get_running_distance_in_msh_footwear(df_strava)
        # df_running_total_weekly = get_total_running_distance_per_week(df_strava, start_date=start_date)
        # df_running_intervention_weekly = get_intervention_shoe_running_distance_per_week(df_strava, start_date=start_date)
        # # combine both weekly dataframes
        # df_running_weekly = pd.merge(df_running_total_weekly, df_running_intervention_weekly, on='week_start', how='outer', suffixes=('_total', '_intervention'))
        # # add percentage in intervention shoe column
        # df_running_weekly['percentage_intervention'] = (df_running_weekly['distance_km_intervention'] / df_running_weekly['distance_km_total']) * 100
        # # safe to excel
        # filename_out = f"{participant_id}_running_summary.xlsx"
        # path_out = get_path_strava_root() / "summary"
        # path_out.mkdir(parents=True, exist_ok=True)
        # df_running_weekly.to_excel(path_out / filename_out, index=False)
        # plot_intervention_percentage(df_running_weekly, participant_id, path_out)


# todo:
# 1. Get the total mileage in the intervention shoe per participant during the study period (12w)
# 2. Get the overall average weekly running mileage during the 12 weeks prior to the study start date and compare to the weekly mileage during the study period
# 3. Get the total overall adherence to the intervention shoe as percentage of total running distance
# 4. Adherence: percentage of running distance in the intervention shoe over the total running distance during the study period
# 5. Get the nuber of weeks where at least one run was done in the intervention shoe
# 6. Get the total number of days of the study period per participant


def data_reduction():
    path_strava_raw = get_path_strava_raw()
    p_ids = PARTICIPANT_IDS
    for participant_id in p_ids:
        path_participant = path_strava_raw / participant_id
        if not path_participant.exists():
            warnings.warn(f"Participant path does not exist: {path_participant}")
            continue
        strava_files = list(path_participant.glob("*.xlsx"))
        if len(strava_files) != 1:
            print(f"Expected one strava file for {participant_id}, found {len(strava_files)}")
            continue
        df_strava = pd.read_excel(strava_files[0])
        try:
            start_date, end_date = get_pre_post_dates(participant_id)
        except ValueError as e:
            print(f"Error getting session dates for participant {participant_id}: {e}")
            continue
        #
        #
        # Make a new excel sheet per participant, containing only runs from 12 weeks prior to study start data until study end date
        #
        #
        df_runs_only = get_runs_only(df_strava)
        if len(df_runs_only) == 0:
            print(f"No running activities found for participant {participant_id}")
            continue
        pre_study_check_date = start_date - pd.Timedelta(weeks=12)
        df_reduced = cut_to_study_period(df_runs_only, pre_study_check_date, end_date)
        columns = ['start_date_local', 'name', 'distance', 'moving_time', 'type', 'gear_model_name']
        df_reduced = df_reduced[columns].copy()
        # add the absolute week number (always starting on monday) to the dataframe, relative to the study start date (study start date week is week 1)
        df_reduced['week_number'] = ((df_reduced['start_date_local'] - start_date) / np.timedelta64(1, 'W')).apply(np.ceil).astype(int)
        # add a column indicating whether the run was done in the intervention shoe
        df_reduced['in_intervention_shoe'] = df_reduced['gear_model_name'].str.contains('MSH', na=False)
        # save to excel
        filename_out = f"{participant_id}_runs_study_period.xlsx"
        path_out = get_path_strava_root() / "runs_only"
        path_out.mkdir(parents=True, exist_ok=True)
        df_reduced.to_excel(path_out / filename_out, index=False)
    return


def check_pre_study_data():
    path_root = get_path_root()
    path_data = path_root / "strava" / "runs_only"
    p_ids = PARTICIPANT_IDS
    df_out = pd.DataFrame(columns=['participant_id', 'days_pre_study', 'days_study', 'diff', 'mileage_pre_study_km', 'mileage_study_km', 'ratio'])
    for participant_id in p_ids:
        print(f"Checking pre-study data for participant {participant_id}...")
        path_participant = path_data / f"{participant_id}_runs_study_period.xlsx"
        if not path_participant.exists():
            warnings.warn(f"Participant file does not exist: {path_participant}")
            continue
        df_strava = pd.read_excel(path_participant)

        try:
            start_date, end_date = get_pre_post_dates(participant_id)
        except ValueError as e:
            print(f"Error getting session dates for participant {participant_id}: {e}")
            continue
        pre_study_check_date = start_date - pd.Timedelta(weeks=12)
        df_pre_study_period = cut_to_study_period(df_strava, pre_study_check_date, start_date - pd.Timedelta(days=1))
        df_study_period = cut_to_study_period(df_strava,  start_date, end_date)
        days_pre_study = (start_date - pre_study_check_date).days
        days_study = int((end_date - start_date).astype('timedelta64[D]') / np.timedelta64(1, 'D'))

        mileage_pre_study = df_pre_study_period['distance'].sum() / 1000
        mileage_study = df_study_period['distance'].sum() / 1000
        data = {'participant_id': participant_id,
                                'days_pre_study': days_pre_study,
                                'days_study': days_study,
                                'diff': days_study - days_pre_study,
                                'mileage_pre_study_km': mileage_pre_study,
                                'mileage_study_km': mileage_study,
                                'ratio': mileage_study / mileage_pre_study if mileage_pre_study > 0 else np.nan
                }

        df_add = pd.DataFrame(data, index=[0])
        df_out = pd.concat([df_out, df_add], ignore_index=True)
    df_out.to_excel(path_root / "strava" / "pre_study_data_check.xlsx", index=False)


if __name__ == '__main__':
    # main(participant_id="HAB10")
    # data_reduction()
    # main()
    check_pre_study_data()
