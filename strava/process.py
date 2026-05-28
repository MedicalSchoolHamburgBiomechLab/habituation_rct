import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from labtools.batch_processor import BatchProcessor

from common import get_demographics, DROPOUTS
from strava.strava_analysis import get_path_strava_root


def get_runs_only(df_strava: pd.DataFrame) -> pd.DataFrame:
    # Filter the dataframe to include only running activities
    df_runs = df_strava[df_strava['type'] == 'Run'].copy()
    return df_runs


def process_get_runs_only(path: Path) -> pd.DataFrame:
    df_all = pd.read_excel(path, index_col=0)
    return get_runs_only(df_all)


def get_pre_post_dates(participant_id: str) -> (pd.Timestamp, pd.Timestamp):
    # 1. look up the demographics excel and check for the PRE- and POST-Session dates
    df_demo = get_demographics()
    start_date = df_demo[(df_demo['participant_id'] == participant_id) & (df_demo['session'] == 'PRE')]['session_date'].values[0]
    # 1.1 Convert strings to datetime if necessary
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    end_date = df_demo[(df_demo['participant_id'] == participant_id) & (df_demo['session'] == 'POST')]['session_date'].values[0]
    if pd.isna(start_date):
        raise ValueError(f"Missing PRE session date for participant {participant_id}")
    if pd.isna(end_date):
        end_date = start_date + np.timedelta64(12, 'W')  # assume 12 weeks later if POST date is missing, but warn about it

        warnings.warn(f"Missing POST session date for participant {participant_id}")
    # check that diff between start and end date is at least 8 weeks
    full_weeks = int((end_date - start_date) / np.timedelta64(1, 'W'))
    if full_weeks < 8:
        raise ValueError(f"Session dates for participant {participant_id} are less than 8 weeks apart ({full_weeks} weeks)")

    return pd.to_datetime(start_date), pd.to_datetime(end_date)


def cut_to_study_period(df_strava: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    # cut the dataframe to only include activities between these two dates (inclusive)
    df_strava['start_date_local'] = pd.to_datetime(df_strava['start_date_local'])
    mask = (df_strava['start_date_local'] >= start_date) & (df_strava['start_date_local'] <= end_date)
    df_cut = df_strava[mask].copy()
    return df_cut


def get_activities_period_days(df_all: pd.DataFrame) -> int:
    df_all["date"] = pd.to_datetime(df_all['start_date_local'])
    diff = (df_all['date'].max() - df_all['date'].min()).days
    return diff


def make_activities_summary(path: Path) -> dict:
    p_id = path.parent.stem
    date_start, date_end = get_pre_post_dates(p_id)
    out = dict()
    df_all = pd.read_excel(path, index_col=0)
    df_all["date"] = pd.to_datetime(df_all['start_date_local'])

    date_first_activity = df_all['date'].min()
    date_last_activity = df_all['date'].max()

    out["count_all_activities"] = len(df_all)
    out["days_all_activities"] = (date_last_activity - date_first_activity).days
    out["date_first_activity"] = date_first_activity.strftime('%d.%m.%Y')
    out["date_start_period"] = date_start.strftime('%d.%m.%Y')
    out["days_before_start"] = (date_start - date_first_activity).days
    out["date_end_period"] = date_end.strftime('%d.%m.%Y')
    out["days_period"] = (date_end - date_start).days
    out["date_last_activity"] = date_last_activity.strftime('%d.%m.%Y')
    out["days_after_end"] = (date_last_activity - date_end).days

    return out


def summary(batch_processor: BatchProcessor):
    results = batch_processor.apply(make_activities_summary
                                    , multiprocess=True
                                    )
    df_results = pd.json_normalize(results)
    df_merged = pd.concat([batch_processor.index.reset_index(drop=True), df_results], axis=1)
    df_merged.drop(columns=["path"], inplace=True)

    path_df_out = batch_processor.path_root.parent / 'summary.xlsx'
    df_merged.to_excel(path_df_out, index=False)


def get_intervention_shoe_running_distance_per_week(
        df_strava: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    # returns a dataframe with total running distance per week in the intervention shoe
    df_runs = get_runs_only(df_strava)
    # Check where column "gear_model_name" has "MSH" in it
    df_msh = df_runs[df_runs['gear_model_name'].str.contains('MSH', na=False)]
    df_msh['start_date_local'] = pd.to_datetime(df_msh['start_date_local'])
    df_msh.set_index('start_date_local', inplace=True)
    df_weekly = (
        df_msh
        .resample('7D', origin=start_date, label='left', closed='left')
        .agg(distance=('distance', 'sum'))
    )

    # Build the full week range and fill missing weeks with 0
    last_week = end_date if end_date is not None else df_weekly.index.max()
    full_weeks = pd.date_range(start=start_date, end=last_week, freq='7D')
    df_weekly = (
        df_weekly
        .reindex(full_weeks, fill_value=0)
        .rename_axis('week_start')
        .reset_index()
    )
    df_weekly['distance_km'] = df_weekly['distance'] / 1000  # Convert meters to kilometers
    return df_weekly[['week_start', 'distance_km']]


def make_metrics(path: Path) -> dict:
    # 1. Get the total mileage in the intervention shoe per participant during the study period (12w) ✅
    # 2. Get the overall average weekly running mileage during the 12 weeks prior to the study start date and compare to the weekly mileage during the study period
    # 3. Get the total overall adherence to the intervention shoe as percentage of total running distance ✅
    # 4. Adherence: percentage of running distance in the intervention shoe over the total running distance during the study period ✅
    # 5. Get the nuber of weeks where at least one run was done in the intervention shoe ✅
    # 6. Get the total number of days of the study period per participant ✅

    out = dict()
    p_id = path.parent.stem
    date_start, date_end = get_pre_post_dates(p_id)

    df_all = pd.read_excel(path, index_col=0)
    df_all["date"] = pd.to_datetime(df_all['start_date_local'])
    df_period_runs = get_runs_only(cut_to_study_period(df_all, date_start, date_end))
    df_intervention = df_period_runs[df_period_runs["gear_name"].str.contains("MSH", na=False)]

    km_all = df_period_runs["distance"].sum() / 1000
    km_intervention = df_intervention["distance"].sum() / 1000

    out["days_period"] = (date_end - date_start).days
    out['number_runs_all'] = len(df_period_runs)
    out['number_runs_intervention'] = len(df_intervention)
    out['kilometers_all'] = km_all
    out['kilometers_intervention'] = km_intervention
    out['distance_intervention_percent'] = np.round(km_intervention / km_all * 100, 1)

    # some checks:
    out['date_period_start'] = date_start.strftime('%d.%m.%Y')
    out['date_first_activity_intervention'] = df_intervention['date'].min().strftime('%d.%m.%Y')  # date of the first activity in the intervention shoe
    out['days_till_first_activity_intervention'] = (df_intervention['date'].min() - date_start).days  # days between PRE session and first intervention activity

    df_weekly_intervention = get_intervention_shoe_running_distance_per_week(df_intervention, date_start)
    out['date_week_period_start'] = df_weekly_intervention['week_start'][0].strftime('%d.%m.%Y')
    out['number_weeks'] = len(df_weekly_intervention)
    out['number_weeks_intervention'] = len(df_weekly_intervention[df_weekly_intervention['distance_km'] > 0])

    return out


def metrics(batch_processor: BatchProcessor):
    results = batch_processor.apply(make_metrics,
                                    multiprocess=True)
    df_results = pd.json_normalize(results)
    df_merged = pd.concat([batch_processor.index.reset_index(drop=True), df_results], axis=1)
    adherence_percent_thresh = 15  # %
    adherence_km_thresh = 20  # km
    adherence_n_weeks_thresh = 9  # weeks/75% (where km in intervention shoe was greater than 0)

    adherence_thresholds = {
        "kilometers_intervention": adherence_km_thresh,  # km
        "distance_intervention_percent": adherence_percent_thresh,  # %
        "number_weeks_intervention": adherence_n_weeks_thresh,  # weeks
    }

    # individual adherence overview:
    for adherence_criterion, value in adherence_thresholds.items():
        df_merged[f'adhered_{adherence_criterion}'] = df_merged[adherence_criterion] >= value
    # sum the number of adherence criteria violations:
    adhered_cols = [col for col in df_merged.columns if 'adhered_' in col]
    df_merged['num_violations'] = (~df_merged[adhered_cols]).sum(axis=1)

    df_merged.drop(columns=["path"], inplace=True)

    path_df_out = batch_processor.path_root.parent / 'metrics.xlsx'
    df_merged.to_excel(path_df_out, index=False)
    # some group stats:
    metric_cols = [
        "days_period",
        "kilometers_all",
        "kilometers_intervention",
        "distance_intervention_percent",
        "number_weeks",
        "number_weeks_intervention",
    ]

    df_stats = df_merged[metric_cols].describe().T[["count", "mean", "std", "min", "max"]]

    df_stats["adherence_threshold"] = df_stats.index.map(adherence_thresholds)

    df_stats["n_adhered"] = df_stats.index.map(
        lambda c: (df_merged[c] >= adherence_thresholds[c]).sum() if c in adherence_thresholds else pd.NA
    )

    df_stats.to_excel(batch_processor.path_root.parent / "group_stats.xlsx", index_label="metric")


if __name__ == '__main__':
    path_root = get_path_strava_root()
    path_raw = path_root / 'raw'
    bp = BatchProcessor(path_raw,
                        level_names=["participant", "activities_file"],
                        file_pattern="activities*.xlsx")
    bp.filter(participant=DROPOUTS, method="remove", inplace=True)
    print(bp.summary())
    summary(bp)
    metrics(bp)
