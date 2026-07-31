import ast
import datetime
import json
from itertools import product

import numpy as np
import pandas as pd
from labtools.batch_processor import BatchProcessor
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.io import loadmat

from common import get_path_root

SAVE_JSON = False


def get_trial_no(filename: str) -> str:
    trial_no = filename.split("_signals")[0][-1]
    if trial_no is None:
        foo = 1
    return trial_no


def get_events(p_id: str, session: str, condition: str, filename: str) -> list:
    path_root = get_path_root()
    path_events_file = path_root / "pressure" / "events.xlsx"
    df_events = pd.read_excel(path_events_file)
    df_sub = df_events[(df_events["participant"] == p_id)
                       & (df_events["session"] == session)
                       & (df_events["condition"] == condition)]
    trial_no = get_trial_no(filename)
    row = df_sub[df_sub["trial"].str.contains(f"_{trial_no}", regex=False)]
    if row.empty:
        return []
    # format signal to tuples (frame_no, side, type)
    list_of_tuples = []
    for side, e_type in product(["left", "right"], ["ic", "tc"]):
        events = row[f"{side}.{e_type}"].apply(ast.literal_eval).values[0]

        evt_tuples = [(t, side, e_type) for t in events]
        list_of_tuples = list_of_tuples + evt_tuples
    list_of_tuples.sort(key=lambda x: x[0])

    return list_of_tuples


def convert_theia_signal(signal: np.ndarray, new_rate: int = 300) -> np.ndarray:
    t_85 = np.arange(len(signal)) / 85.0
    t_300 = np.arange(start=0, stop=t_85[-1], step=1 / new_rate)
    spline = CubicSpline(t_85, signal)
    return spline(t_300)


def make_cycles(signal: np.ndarray, events: pd.DataFrame, normalize: bool = True) -> list:
    ics = events[events["event"] == "ic"]["frame"].to_numpy()
    cycles = []
    for ic, next_ic in zip(ics[:-1], ics[1:]):
        if next_ic > len(signal):
            break
        cycle = signal[ic:next_ic, :]
        if normalize:
            spline = CubicSpline(np.arange(len(cycle)), cycle)
            cycle = spline(np.linspace(0, len(cycle) - 1, 101))
        cycles.append(cycle)
    return cycles


def get_values_at_ic(signal: np.ndarray, events: pd.DataFrame) -> list:
    ics = events[events["event"] == "ic"]["frame"].to_numpy()
    values = []
    for ic in ics:
        if ic > len(signal):
            break
        values.append(signal[ic, :])
    return values


def get_peak_to_peak_per_step(signal: np.ndarray, events: pd.DataFrame, side: str) -> list:
    ics = events[(events["event"] == "ic") & (events["side"] == side)]["frame"].to_numpy()
    cl_side = "left" if side == "right" else "right"
    cl_ics = events[(events["event"] == "ic") & (events["side"] == cl_side)]["frame"].to_numpy()
    while cl_ics[0] < ics[0]:
        cl_ics = cl_ics[1:]

    ptps = []
    for ic, cl_ic in zip(ics, cl_ics):
        if cl_ic > len(signal):
            break
        ptp = np.ptp(signal[ic:cl_ic, :], axis=0)
        ptps.append(ptp)
    return ptps


def get_rom_during_stance(signal: np.ndarray, events: pd.DataFrame) -> list:
    ics = events[events["event"] == "ic"]["frame"].to_numpy()
    tcs = events[events["event"] == "tc"]["frame"].to_numpy()
    roms = []
    while tcs[0] < ics[0]:
        tcs = tcs[1:]
    for ic, tc in zip(ics, tcs):
        if tc > len(signal):
            break
        stance_phase = signal[ic:tc, :]
        roms.append(np.ptp(stance_phase, axis=0))
    return roms


def get_peak_during_stance(signal: np.ndarray, events: pd.DataFrame) -> list:
    ics = events[events["event"] == "ic"]["frame"].to_numpy()
    tcs = events[events["event"] == "tc"]["frame"].to_numpy()
    roms = []
    while tcs[0] < ics[0]:
        tcs = tcs[1:]
    for ic, tc in zip(ics, tcs):
        if tc > len(signal):
            break
        stance_phase = signal[ic:tc, :]
        roms.append(np.max(stance_phase, axis=0))
    return roms


def func_kinematics(row: pd.Series) -> dict:
    out = {}
    if "Stand" in row.filename:
        return out
    data = loadmat(row.path)
    if "Thorax_Angles" in data.keys():
        print(row.particpant_id)

    # print(f"{row.session}, {row.condition}, {row.filename}")

    events = get_events(p_id=row.participant_id,
                        session=row.session,
                        condition=row.condition,
                        filename=row.filename)
    if len(events) == 0:
        raise Exception("No events found")
    df_ev = pd.DataFrame(events, columns=["frame", "side", "event"])

    joints = ["hip", "knee", "ankle"]
    segments = ["pelvis", "foot"]
    sides = ["left", "right"]

    json_data = {}

    # {
    #     "meta": {
    #         "filename": "...",
    #         "date": "...",
    #         "n_samples": 101,
    #     },
    #     "joints": {
    #         "hip": {"axes": ["Flexion/Extension", "Adduction/Abduction", "Internal/External Rotation"],
    #                 "left": [[[1, 2, 3], ... [1, 2, 3]], [[1, 2, 3], .., [1, 2, 3]]...],
    #                 "right": [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3] ..., [1, 2, 3]]
    #                 },
    #         "knee": {"axes": ["...", "...", "..."], "left": [...], "right": [...]}
    #         "ankle": {"axes": ["...", "...", "..."], "left": [...], "right": [...]}
    #     }
    # }
    json_data["meta"] = {}
    json_data["meta"]["filename"] = row.filename
    json_data["meta"]["processed"] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    json_data["meta"]["n_samples"] = 101
    json_data["joints"] = {j: {} for j in joints}
    json_data["segments"] = {s: {} for s in segments}

    axes_dict = {
        "hip": ["Flexion/Extension", "Add./Abd.", "int./Ext. Rotation"],
        "knee": ["Flexion/Extension", "Add./Abd.", "int./Ext. Rotation"],
        "ankle": ["Flexion/Extension", "Eversion/Inversion", "int./Ext. Rotation"]
    }
    params = {j: {} for j in joints}

    #
    # Pelvis
    #
    json_data["segments"]["pelvis"]["com_position"] = {}
    params["pelvis"] = {}
    params["pelvis"]["vertical_motion"] = {}
    for side in sides:
        #
        # MAKE NORMALIZED SIGNALS
        #
        signal_specifier = "Pelvis_COM_Position"
        pelvis_com_signal = data[signal_specifier][0][0]
        pelvis_com_signal_300 = convert_theia_signal(pelvis_com_signal)
        cycles = make_cycles(pelvis_com_signal_300, df_ev[df_ev["side"] == side.lower()])
        for cycle in cycles:
            plt.plot(cycle[:, 2])
        json_data["segments"]["pelvis"]["com_position"][side] = [c.tolist() for c in cycles]

        ptp = get_peak_to_peak_per_step(signal=pelvis_com_signal_300,
                                        events=df_ev,
                                        side=side)
        pelvis_vertical_motion = float(round(np.mean([v[2] * 100 for v in ptp]), 3))
        params["pelvis"]["vertical_motion"][side] = pelvis_vertical_motion

    for joint in joints:
        json_data["joints"][joint]["axes"] = axes_dict[joint]
        params[joint]["flexion_at_ic"] = {}
        params[joint]["flexion_rom"] = {}
        params[joint]["flexion_peak"] = {}
        for side in sides:
            signal_specifier = f"{side.capitalize()}_{joint.capitalize()}_Angles"
            angle_signal = data[signal_specifier][0][0]
            #
            # MAKE NORMALIZED SIGNALS
            #
            angle_signal_300 = convert_theia_signal(angle_signal)
            # # theia files may have been cut at the end. -> filter pressure-based events that are beyond the signal length
            # df_ev = df_ev[df_ev["frame"] <= len(angle_signal_300)]
            cycles = make_cycles(angle_signal_300, df_ev[df_ev["side"] == side.lower()])
            # for cycle in cycles:
            #     plt.plot(cycle[:, 0])
            # plt.title(joint)
            # plt.show()
            json_data["joints"][joint][side] = [c.tolist() for c in cycles]
            #
            # GET DISCRETE VALUES
            #
            values_at_ic = get_values_at_ic(signal=angle_signal_300, events=df_ev[df_ev["side"] == side.lower()])
            flexion_at_ic = float(round(np.mean([v[0] for v in values_at_ic]), 3))
            params[joint]["flexion_at_ic"][side] = flexion_at_ic

            rom_during_stance = get_rom_during_stance(signal=angle_signal_300,
                                                      events=df_ev[df_ev["side"] == side.lower()])
            flexion_rom = float(round(np.mean([r[0] for r in rom_during_stance]), 3))
            params[joint]["flexion_rom"][side] = flexion_rom

            peak_during_stance = get_peak_during_stance(signal=angle_signal_300,
                                                        events=df_ev[df_ev["side"] == side.lower()])
            flexion_peak = float(round(np.mean([r[0] for r in peak_during_stance]), 3))
            params[joint]["flexion_peak"][side] = flexion_peak
    #
    # Overstriding
    #
    params["overstriding"] = {}
    params["overstriding"]["oh"] = {}
    params["overstriding"]["ok"] = {}

    for side in sides:

        hjc = convert_theia_signal(data[f"{side.capitalize()}_Hip_Center"][0][0])
        kjc = convert_theia_signal(data[f"{side.capitalize()}_Knee_Center"][0][0])
        ajc = convert_theia_signal(data[f"{side.capitalize()}_Ankle_Center"][0][0])

        oh_signal = ajc - hjc
        oh_values = get_values_at_ic(signal=oh_signal, events=df_ev[df_ev["side"] == side.lower()])
        params["overstriding"]["oh"][side] = float(round(np.mean([ohv[0]*100 for ohv in oh_values]),2))


        ok_signal = ajc - kjc
        ok_values = get_values_at_ic(signal=ok_signal, events=df_ev[df_ev["side"] == side.lower()])
        params["overstriding"]["ok"][side] = float(round(np.mean([okv[0]*100 for okv in ok_values]),2))



    # safe data
    trial_no = get_trial_no(row.filename)
    if SAVE_JSON:
        # print(f"{row.filename} - trial_no: {trial_no}")
        path_root = get_path_root()
        path_kinematics = path_root / "kinematics"
        path_out_root = path_kinematics / "json"
        path_out = path_out_root / row.participant_id / row.session / row.condition
        path_out.mkdir(exist_ok=True, parents=True)
        filename = f"{row.participant_id}_{row.session}_{row.condition}_{trial_no}.json"
        path_file_out = path_out / filename
        with open(path_file_out, "w") as f:
            json.dump(json_data, f)

    params["trial_no"] = trial_no

    return params


def func_plot_hip_flexion(row: pd.Series) -> dict:
    out = {}
    if "Stand" in row.filename:
        return out
    data = loadmat(row.path)

    events = get_events(p_id=row.participant_id,
                        session=row.session,
                        condition=row.condition,
                        filename=row.filename)
    if len(events) == 0:
        raise Exception("No events found")
    df_ev = pd.DataFrame(events, columns=["frame", "side", "event"])

    fig, ax = plt.subplots(figsize=(12, 9))
    pct = np.arange(101)

    for side, color in [("left", "C3"), ("right", "C0")]:
        sig = data[f"{side.capitalize()}_Hip_Angles"][0][0]
        t_85 = np.arange(len(sig)) / 85.0
        spline = CubicSpline(t_85, sig)

        ev_side = df_ev[df_ev["side"] == side]
        ic = ev_side.loc[ev_side["event"] == "ic", "frame"].to_numpy()
        tc = ev_side.loc[ev_side["event"] == "tc", "frame"].to_numpy()
        ic = ic[ic / 300.0 <= t_85[-1]]

        cycles, tc_pct = [], []
        for s, e in zip(ic[:-1], ic[1:]):
            cycles.append(spline(np.linspace(s, e, 101) / 300.0))
            tc_in = tc[(tc > s) & (tc < e)]
            if len(tc_in) == 1:
                tc_pct.append((tc_in[0] - s) / (e - s) * 100)

        cycles = np.stack(cycles)  # (n_cycles, 101, 3)
        flex = cycles[:, :, 0]
        m, sd = flex.mean(axis=0), flex.std(axis=0, ddof=1)

        # einzelne TC-Events (dünn, hinten)
        for p in tc_pct:
            ax.axvline(p, color=color, lw=0.6, alpha=0.25, zorder=0)

        # Einzelzyklen
        ax.plot(pct, flex.T, color=color, alpha=0.15, lw=0.6, zorder=1)

        # Mittelwert ± SD
        ax.fill_between(pct, m - sd, m + sd, color=color, alpha=0.25, zorder=2)
        ax.plot(pct, m, color=color, lw=2, zorder=3,
                label=f"{side} (n={len(flex)})")

        # mittleres TC (dick, vorne)
        tc_m = float(np.mean(tc_pct))
        ax.axvline(tc_m, color=color, ls="--", lw=2.0, zorder=4)

        out[f"{side}_hip_flex_mean"] = m
        out[f"{side}_hip_flex_sd"] = sd
        out[f"{side}_tc_pct"] = tc_m
        out[f"{side}_tc_pct_sd"] = float(np.std(tc_pct, ddof=1))
        out[f"{side}_n_cycles"] = len(flex)

    title = f"{row.participant_id} - {row.session} - {row.condition} - {row.filename}"
    ax.set_title(title)
    ax.set_xlabel("Gait Cycle [%]")
    ax.set_ylabel("Hip Flexion [°]")
    ax.set_xlim(0, 100)
    ax.axhline(0, color="k", lw=0.5)
    ax.legend()
    fig.tight_layout()

    path_root = get_path_root()
    path_kinematics = path_root / "kinematics"
    path_plot_out = path_kinematics / "plots"

    filename = f"{row.participant_id}_{row.session}_{row.condition}_{row.filename}_hip_flexion.png"
    fig.savefig(path_plot_out / filename)

    plt.close(fig)
    return out


def print_errors(errors):
    for err in errors:
        path = err[0]
        p = path.parent.parent.parent.stem
        s = path.parent.parent.stem
        c = path.parent.stem
        f = path.stem
        print(f"{p} {s} {c} {f}: {err[1]}")


if __name__ == '__main__':
    path_root = get_path_root()
    path_kinematics = path_root / "kinematics"
    path_signals = path_root / "kinematics" / "mat"
    bp_kin = BatchProcessor(path_signals, "signals.mat", ["participant_id", "session", "condition", "filename"])

    # bp_kin.filter(inplace=True, participant_id=["HAB11"])
    # res_kinematics = bp_kin.apply(func_plot_hip_flexion,
    #                               multiprocess=True)
    SAVE_JSON = True

    res_kinematics = bp_kin.apply(func_kinematics,
                                  multiprocess=True)

    ind = bp_kin.index

    print_errors(bp_kin.errors)

    # raise NotImplementedError

    df_results_kinematics = pd.json_normalize(res_kinematics)
    df_dict_kinematics = pd.concat([bp_kin.index.reset_index(drop=True), df_results_kinematics], axis=1)
    df_dict_kinematics.drop(columns=["path"], inplace=True)
    # drop stand files
    df_dict_kinematics = df_dict_kinematics[df_dict_kinematics["filename"] != "Stand_signals"]

    metrics = [col.replace(".right", "") for col in df_dict_kinematics.columns if "right" in col]

    missing_trial_no = df_dict_kinematics[pd.isna(df_dict_kinematics["trial_no"])]
    for r, row in missing_trial_no.iterrows():
        print(row)

    df_long = pd.wide_to_long(
        df_dict_kinematics,
        stubnames=metrics,
        i=['participant_id', 'session', "condition", "trial_no"],
        j='side',
        sep='.',
        suffix='(left|right)'
    ).reset_index()

    filename = "results_kinematics.xlsx"
    path_file_results = path_kinematics / filename

    df_long.to_excel(path_file_results, index=False)

    print(bp_kin.index.head())

    # 362 + 353
