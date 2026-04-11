import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Parameters, minimize, fit_report


# ============================================================
# Column handling
# ============================================================
def find_best_column(df, candidates, exclude_keywords=None):
    exclude_keywords = exclude_keywords or []
    cols = list(df.columns)
    cols_lower = [c.lower().strip() for c in cols]

    # exact match first
    for cand in candidates:
        cand_l = cand.lower().strip()
        for c, cl in zip(cols, cols_lower):
            if any(ex in cl for ex in exclude_keywords):
                continue
            if cl == cand_l:
                return c

    # substring match second
    for cand in candidates:
        cand_l = cand.lower().strip()
        for c, cl in zip(cols, cols_lower):
            if any(ex in cl for ex in exclude_keywords):
                continue
            if cand_l in cl:
                return c

    raise KeyError(f"Cannot find column from candidates={candidates}, exclude={exclude_keywords}")


def load_hppc_csv(csv_path):
    df = pd.read_csv(csv_path)

    time_col = find_best_column(df, ["time"], exclude_keywords=["timestamp", "stamp"])
    voltage_col = find_best_column(df, ["voltage", "volt"])
    current_col = find_best_column(df, ["current", "curr"])
    ah_col = find_best_column(df, ["ah"])

    out = pd.DataFrame({
        "time_s": pd.to_numeric(df[time_col], errors="raise"),
        "voltage_v": pd.to_numeric(df[voltage_col], errors="raise"),
        "current_a": pd.to_numeric(df[current_col], errors="raise"),
        "ah": pd.to_numeric(df[ah_col], errors="raise"),
    })

    out["time_s"] = out["time_s"] - out["time_s"].iloc[0]
    return out


# ============================================================
# Basic helpers
# ============================================================
def estimate_nominal_capacity_ah(df, default_capacity=2.9):
    ah_max = np.nanmax(df["ah"].values)
    if np.isfinite(ah_max) and ah_max > 0.5:
        return max(ah_max, default_capacity)
    return default_capacity


def add_soc(df, nominal_capacity_ah):
    soc = 1.0 - df["ah"].values / nominal_capacity_ah
    soc = np.clip(soc, 0.0, 1.0)
    out = df.copy()
    out["soc_est"] = soc
    return out


# ============================================================
# Pulse detection
# ============================================================
def detect_discharge_pulses(
    df,
    current_threshold_a=0.5,
    min_pulse_duration_s=2.0,
    pre_rest_s=8.0,
    post_rest_s=25.0,
):
    """
    Detect discharge pulses.
    Assumes discharge current is negative in this dataset.
    """
    t = df["time_s"].to_numpy()
    i = df["current_a"].to_numpy()

    dt = np.median(np.diff(t))
    in_pulse = i < -abs(current_threshold_a)

    starts = np.where((~in_pulse[:-1]) & (in_pulse[1:]))[0] + 1
    ends = np.where((in_pulse[:-1]) & (~in_pulse[1:]))[0] + 1

    if in_pulse[0]:
        starts = np.insert(starts, 0, 0)
    if in_pulse[-1]:
        ends = np.append(ends, len(in_pulse) - 1)

    rows = []
    pulse_id = 1
    for s, e in zip(starts, ends):
        duration = t[e] - t[s]
        if duration < min_pulse_duration_s:
            continue

        pre_n = int(round(pre_rest_s / dt))
        post_n = int(round(post_rest_s / dt))

        ws = max(0, s - pre_n)
        we = min(len(df) - 1, e + post_n)

        i_seg = i[s:e]
        soc_seg = df["soc_est"].iloc[s:e].to_numpy()

        rows.append({
            "pulse_id": pulse_id,
            "pulse_start_idx": int(s),
            "pulse_end_idx": int(e),
            "window_start_idx": int(ws),
            "window_end_idx": int(we),
            "time_start_s": float(t[s]),
            "time_end_s": float(t[e]),
            "pulse_duration_s": float(duration),
            "current_mean_a": float(np.mean(i_seg)),
            "current_abs_mean_a": float(np.mean(np.abs(i_seg))),
            "soc_test": float(np.mean(soc_seg)),
        })
        pulse_id += 1

    return pd.DataFrame(rows)


# ============================================================
# Plateau grouping by pulse sequence
# ============================================================
def assign_plateau_groups_by_sequence(pulse_df, pulses_per_group=5):
    """
    Group pulses by chronological order, not by rounded SOC.
    Example:
        pulses 1-5   -> group 1
        pulses 6-10  -> group 2
        ...
    """
    out = pulse_df.sort_values("pulse_id").copy().reset_index(drop=True)
    out["soc_group_id"] = (np.arange(len(out)) // pulses_per_group) + 1
    return out


# ============================================================
# 2RC simulation with pulse-specific initial states
# ============================================================
def simulate_2rc_window(t, i, ocv, r0, r1, c1, r2, c2, v1_init=0.0, v2_init=0.0):
    """
    Terminal voltage model:
        V = OCV - I_d*R0 - v1 - v2
    where discharge current is positive in model coordinates:
        I_d = -i_file
    """
    t = np.asarray(t, dtype=float)
    i = np.asarray(i, dtype=float)
    I_d = -i

    v1 = float(v1_init)
    v2 = float(v2_init)
    v = np.zeros_like(t)

    v[0] = ocv - I_d[0] * r0 - v1 - v2

    for k in range(1, len(t)):
        dt = max(t[k] - t[k - 1], 1e-9)

        tau1 = max(r1 * c1, 1e-9)
        tau2 = max(r2 * c2, 1e-9)

        a1 = np.exp(-dt / tau1)
        a2 = np.exp(-dt / tau2)

        v1 = a1 * v1 + r1 * (1.0 - a1) * I_d[k]
        v2 = a2 * v2 + r2 * (1.0 - a2) * I_d[k]

        v[k] = ocv - I_d[k] * r0 - v1 - v2

    return v


# ============================================================
# Dataset builder
# ============================================================
def build_group_datasets(df, pulse_group_df):
    datasets = []
    for _, row in pulse_group_df.iterrows():
        ws = int(row["window_start_idx"])
        we = int(row["window_end_idx"])
        seg = df.iloc[ws:we + 1].copy().reset_index(drop=True)

        t = seg["time_s"].to_numpy()
        t = t - t[0]

        i = seg["current_a"].to_numpy()
        v = seg["voltage_v"].to_numpy()

        # Estimate pre-pulse baseline from samples before pulse onset in the window
        pulse_start_global = int(row["pulse_start_idx"])
        local_pulse_start = pulse_start_global - ws
        local_pulse_start = max(local_pulse_start, 0)

        if local_pulse_start >= 3:
            v_pre = float(np.mean(v[max(0, local_pulse_start - 5):local_pulse_start]))
        else:
            v_pre = float(v[0])

        datasets.append({
            "pulse_id": int(row["pulse_id"]),
            "soc_test": float(row["soc_test"]),
            "current_mean_a": float(row["current_mean_a"]),
            "time_s": t,
            "current_a": i,
            "voltage_v": v,
            "v_pre_est": v_pre,
        })

    return datasets


# ============================================================
# Weighted residual
# ============================================================
def build_weights(ds):
    """
    Put higher weight on:
    - pulse onset
    - early relaxation after pulse
    """
    t = ds["time_s"]
    i = ds["current_a"]

    w = np.ones_like(t)

    # detect active-pulse region
    active = np.abs(i) > 0.2
    idx_active = np.where(active)[0]

    if len(idx_active) > 0:
        s = idx_active[0]
        e = idx_active[-1]

        # onset emphasis
        onset_end = min(len(t), s + 10)
        w[s:onset_end] *= 6.0

        # pulse body modest emphasis
        w[s:e+1] *= 2.0

        # early post-pulse relaxation
        post_end = min(len(t), e + 20)
        w[e+1:post_end] *= 4.0

    return w


# ============================================================
# Joint objective
# ============================================================
def residual_group(params, datasets):
    r0 = params["r0"].value
    r1 = params["r1"].value
    c1 = params["c1"].value
    r2 = params["r2"].value
    c2 = params["c2"].value

    res_all = []

    for ds in datasets:
        pid = ds["pulse_id"]
        ocv = params[f"ocv_p{pid}"].value
        v1_init = params[f"v1_init_p{pid}"].value
        v2_init = params[f"v2_init_p{pid}"].value

        v_model = simulate_2rc_window(
            ds["time_s"],
            ds["current_a"],
            ocv, r0, r1, c1, r2, c2,
            v1_init=v1_init,
            v2_init=v2_init,
        )
        res = v_model - ds["voltage_v"]
        w = build_weights(ds)
        res_all.append(res * w)

    return np.concatenate(res_all)


def make_initial_guess(datasets):
    """
    Use the strongest pulse for rough shared RC initialization.
    Keep OCV and initial states pulse-specific.
    """
    ds = max(datasets, key=lambda x: abs(x["current_mean_a"]))
    t = ds["time_s"]
    i = ds["current_a"]
    v = ds["voltage_v"]

    i_abs = np.abs(i)
    pulse_mask = i_abs > 0.5 * np.max(i_abs)
    if not np.any(pulse_mask):
        pulse_mask = i_abs > 0.2

    first_pulse_idx = np.argmax(pulse_mask)
    ocv0 = ds["v_pre_est"]

    if first_pulse_idx + 1 < len(v):
        dv0 = ocv0 - v[first_pulse_idx]
        di0 = abs(i[first_pulse_idx]) if abs(i[first_pulse_idx]) > 1e-6 else 1.0
        r0_guess = np.clip(dv0 / di0, 0.001, 0.05)
    else:
        r0_guess = 0.015

    return {
        "r0": float(r0_guess),
        "r1": 0.005,
        "c1": 200.0,
        "r2": 0.010,
        "c2": 2000.0,
    }


def fit_one_soc_group(df, pulse_group_df):
    datasets = build_group_datasets(df, pulse_group_df)
    guess = make_initial_guess(datasets)

    params = Parameters()

    # shared RC parameters
    params.add("r0", value=guess["r0"], min=1e-5, max=0.08)
    params.add("r1", value=guess["r1"], min=1e-6, max=0.10)
    params.add("c1", value=guess["c1"], min=1.0, max=1e5)
    params.add("r2", value=guess["r2"], min=1e-6, max=0.20)
    params.add("c2", value=guess["c2"], min=10.0, max=1e6)

    # pulse-specific OCV and initial states
    for ds in datasets:
        pid = ds["pulse_id"]
        v_pre = ds["v_pre_est"]

        params.add(f"ocv_p{pid}", value=v_pre, min=max(2.0, v_pre - 0.15), max=min(4.3, v_pre + 0.15))
        params.add(f"v1_init_p{pid}", value=0.0, min=-0.2, max=0.2)
        params.add(f"v2_init_p{pid}", value=0.0, min=-0.2, max=0.2)

    result = minimize(residual_group, params, args=(datasets,), method="least_squares")

    fitted = {name: result.params[name].value for name in result.params.keys()}
    r0 = fitted["r0"]
    r1 = fitted["r1"]
    c1 = fitted["c1"]
    r2 = fitted["r2"]
    c2 = fitted["c2"]

    tau1 = r1 * c1
    tau2 = r2 * c2

    per_pulse = []
    traces = []

    for ds in datasets:
        pid = ds["pulse_id"]
        ocv = fitted[f"ocv_p{pid}"]
        v1_init = fitted[f"v1_init_p{pid}"]
        v2_init = fitted[f"v2_init_p{pid}"]

        v_fit = simulate_2rc_window(
            ds["time_s"], ds["current_a"],
            ocv, r0, r1, c1, r2, c2,
            v1_init=v1_init, v2_init=v2_init
        )

        err = v_fit - ds["voltage_v"]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))

        per_pulse.append({
            "pulse_id": pid,
            "soc_test": ds["soc_test"],
            "current_mean_a": ds["current_mean_a"],
            "ocv_v": ocv,
            "v1_init_v": v1_init,
            "v2_init_v": v2_init,
            "r0_ohm": r0,
            "r1_ohm": r1,
            "c1_f": c1,
            "r2_ohm": r2,
            "c2_f": c2,
            "tau1_s": tau1,
            "tau2_s": tau2,
            "r_total_ohm": r0 + r1 + r2,
            "rmse_v": rmse,
            "mae_v": mae,
        })

        tr = pd.DataFrame({
            "pulse_id": pid,
            "time_s": ds["time_s"],
            "current_a": ds["current_a"],
            "voltage_measured_v": ds["voltage_v"],
            "voltage_fitted_v": v_fit,
            "error_v": err,
        })
        traces.append(tr)

    group_meta = {
        "soc_group_id": int(pulse_group_df["soc_group_id"].iloc[0]),
        "n_pulses": int(len(pulse_group_df)),
        "soc_mean": float(pulse_group_df["soc_test"].mean()),
        "r0_ohm": r0,
        "r1_ohm": r1,
        "c1_f": c1,
        "r2_ohm": r2,
        "c2_f": c2,
        "tau1_s": tau1,
        "tau2_s": tau2,
        "r_total_ohm": r0 + r1 + r2,
        "fit_success": bool(result.success),
        "message": str(result.message),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
    }

    return result, group_meta, pd.DataFrame(per_pulse), traces


# ============================================================
# Plot/save helpers
# ============================================================
def save_group_plots(output_dir, group_id, traces):
    group_dir = output_dir / f"soc_group_{group_id:02d}"
    group_dir.mkdir(parents=True, exist_ok=True)

    for tr in traces:
        pulse_id = int(tr["pulse_id"].iloc[0])

        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax[0].plot(tr["time_s"], tr["voltage_measured_v"], label="Measured")
        ax[0].plot(tr["time_s"], tr["voltage_fitted_v"], "--", label="Fitted")
        ax[0].set_ylabel("Voltage [V]")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        ax[1].plot(tr["time_s"], tr["error_v"])
        ax[1].set_xlabel("Time [s]")
        ax[1].set_ylabel("Error [V]")
        ax[1].grid(True, alpha=0.3)

        fig.suptitle(f"SOC group {group_id:02d} - Pulse {pulse_id}")
        fig.tight_layout()
        fig.savefig(group_dir / f"pulse_{pulse_id:03d}_fit.png", dpi=300)
        plt.close(fig)

        tr.to_csv(group_dir / f"pulse_{pulse_id:03d}_trace.csv", index=False)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Improved groupwise 2RC identification from HPPC data."
    )
    parser.add_argument("csv_file", type=str, help="HPPC CSV file path")
    parser.add_argument("--output-dir", type=str, default="results_2rc_groupwise_improved")
    parser.add_argument("--nominal-capacity-ah", type=float, default=2.9)
    parser.add_argument("--current-threshold-a", type=float, default=0.5)
    parser.add_argument("--min-pulse-duration-s", type=float, default=2.0)
    parser.add_argument("--pre-rest-s", type=float, default=8.0)
    parser.add_argument("--post-rest-s", type=float, default=25.0)
    parser.add_argument("--pulses-per-group", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_hppc_csv(args.csv_file)
    nominal_capacity_ah = args.nominal_capacity_ah or estimate_nominal_capacity_ah(df)
    df = add_soc(df, nominal_capacity_ah)

    pulse_df = detect_discharge_pulses(
        df,
        current_threshold_a=args.current_threshold_a,
        min_pulse_duration_s=args.min_pulse_duration_s,
        pre_rest_s=args.pre_rest_s,
        post_rest_s=args.post_rest_s,
    )

    if pulse_df.empty:
        raise RuntimeError("No discharge pulses were detected.")

    pulse_df = assign_plateau_groups_by_sequence(
        pulse_df,
        pulses_per_group=args.pulses_per_group
    )
    pulse_df.to_csv(output_dir / "detected_pulses_with_groups.csv", index=False)

    group_summaries = []
    per_pulse_all = []
    fit_reports = {}

    grouped = pulse_df.groupby("soc_group_id", sort=True)

    for gid, gdf in grouped:
        result, group_meta, per_pulse_df, traces = fit_one_soc_group(df, gdf)

        group_summaries.append(group_meta)
        per_pulse_all.append(per_pulse_df)
        fit_reports[f"soc_group_{gid:02d}"] = fit_report(result)

        save_group_plots(output_dir, gid, traces)

    group_summary_df = pd.DataFrame(group_summaries).sort_values("soc_group_id")
    per_pulse_df = pd.concat(per_pulse_all, ignore_index=True).sort_values("pulse_id")

    group_summary_df.to_csv(output_dir / "2rc_group_fit_summary.csv", index=False)
    per_pulse_df.to_csv(output_dir / "2rc_per_pulse_from_group_fit.csv", index=False)

    with open(output_dir / "fit_reports.json", "w", encoding="utf-8") as f:
        json.dump(fit_reports, f, indent=2)

    print(f"Done. Results saved to: {output_dir.resolve()}")
    print("Generated files:")
    print("  - detected_pulses_with_groups.csv")
    print("  - 2rc_group_fit_summary.csv")
    print("  - 2rc_per_pulse_from_group_fit.csv")
    print("  - fit_reports.json")
    print("  - soc_group_XX/pulse_XXX_fit.png")
    print("  - soc_group_XX/pulse_XXX_trace.csv")


if __name__ == "__main__":
    main()