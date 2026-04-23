#!/usr/bin/env python3
"""
2RC ECM parameter identification framework for HPPC data.

Target use case
---------------
- CSV exported from battery cycler with at least: Time, Voltage, Current
- HPPC-style pulse + rest segments
- Identification of per-pulse 2RC parameters:
    R0, R1, C1, R2, C2
- Optional SOC tagging based on integrated Ah and nominal capacity

Key features
------------
1. Robust CSV loading with column auto-detection.
2. Current-step based pulse/rest segmentation.
3. Instantaneous resistance estimation for R0.
4. lmfit-based bounded least-squares fit for 2RC parameters.
5. Exact discrete-time state update for RC branch voltages under piecewise-constant current.
6. Batch fitting for all detected HPPC pulses.
7. Diagnostic plots and CSV export.

Notes
-----
- Sign convention is configurable. By default, positive current is treated as discharge.
- The framework is intentionally transparent and modifiable, suitable for research workflows.
- A 2RC Thevenin model is used here:
      Vt = OCV(SOC) - I*R0 - V1 - V2
      dV1/dt = -(1/(R1*C1))*V1 + I/C1
      dV2/dt = -(1/(R2*C2))*V2 + I/C2
  under the positive-discharge sign convention.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Parameters, minimize, fit_report
from scipy.signal import medfilt


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass
class PulseWindow:
    pulse_id: int
    start_idx: int
    end_idx: int
    rest_before_start_idx: int
    rest_before_end_idx: int
    rest_after_start_idx: int
    rest_after_end_idx: int
    current_level: float
    duration_s: float
    soc_est: float


@dataclass
class FitSummary:
    pulse_id: int
    soc_est: float
    current_level: float
    r0_ohm: float
    r1_ohm: float
    c1_f: float
    r2_ohm: float
    c2_f: float
    tau1_s: float
    tau2_s: float
    rmse_v: float
    success: bool
    message: str
    nfev: int


# -----------------------------------------------------------------------------
# Column handling and preprocessing
# -----------------------------------------------------------------------------

COLUMN_ALIASES = {
    "time": ["Time", "Test_Time(s)", "Step_Time(s)", "time", "time_s"],
    "voltage": ["Voltage", "V", "voltage", "Voltage(V)"],
    "current": ["Current", "I", "current", "Current(A)"],
    "ah": ["Ah", "Amp_Hours", "Capacity_Ah", "ah"],
    "temperature": ["Battery_Temp_degC", "Cell_Temp_degC", "Temperature", "Temp_degC"],
}


def infer_column(columns: Sequence[str], key: str) -> str:
    aliases = COLUMN_ALIASES[key]
    for alias in aliases:
        if alias in columns:
            return alias
    raise KeyError(f"Could not infer required column for '{key}'. Available columns: {list(columns)}")



def load_hppc_csv(csv_path: str | Path, current_sign: str = "auto") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = df.columns

    time_col = infer_column(cols, "time")
    voltage_col = infer_column(cols, "voltage")
    current_col = infer_column(cols, "current")

    out = pd.DataFrame(
        {
            "time_s": pd.to_numeric(df[time_col], errors="coerce"),
            "voltage_v": pd.to_numeric(df[voltage_col], errors="coerce"),
            "current_a_raw": pd.to_numeric(df[current_col], errors="coerce"),
        }
    )

    if "ah" in COLUMN_ALIASES:
        try:
            ah_col = infer_column(cols, "ah")
            out["ah"] = pd.to_numeric(df[ah_col], errors="coerce")
        except KeyError:
            pass

    try:
        temp_col = infer_column(cols, "temperature")
        out["temperature_c"] = pd.to_numeric(df[temp_col], errors="coerce")
    except KeyError:
        pass

    out = out.dropna(subset=["time_s", "voltage_v", "current_a_raw"]).copy()
    out = out.sort_values("time_s").reset_index(drop=True)

    # Shift time to start at 0 for convenience
    out["time_s"] = out["time_s"] - out["time_s"].iloc[0]

    # Determine sign convention
    i_raw = out["current_a_raw"].to_numpy()
    if current_sign == "discharge_positive":
        current = i_raw.copy()
    elif current_sign == "discharge_negative":
        current = -i_raw
    else:
        # Heuristic: for typical HPPC discharge pulse, terminal voltage drops when current is applied.
        # We inspect the largest current step and compare with voltage change.
        di = np.diff(i_raw, prepend=i_raw[0])
        step_idx = int(np.argmax(np.abs(di))) if len(di) else 0
        idx2 = min(step_idx + 3, len(out) - 1)
        dv = out["voltage_v"].iloc[idx2] - out["voltage_v"].iloc[max(step_idx - 1, 0)]
        if i_raw[step_idx] > 0 and dv < 0:
            current = i_raw.copy()  # positive likely means discharge
        elif i_raw[step_idx] < 0 and dv < 0:
            current = -i_raw        # negative likely means discharge
        else:
            current = i_raw.copy()

    out["current_a"] = current
    out["dt_s"] = out["time_s"].diff().fillna(0.0).clip(lower=0.0)
    return out



def smooth_current(current_a: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size == 1:
        return current_a.copy()
    return medfilt(current_a, kernel_size=kernel_size)



def estimate_soc(df: pd.DataFrame, nominal_capacity_ah: float, initial_soc: float = 1.0) -> pd.Series:
    if "ah" in df.columns and df["ah"].notna().any():
        ah = df["ah"].to_numpy()
        ah0 = np.nanmin(ah)
        used_ah = ah - ah0
        soc = initial_soc - used_ah / nominal_capacity_ah
    else:
        time_h = df["time_s"].to_numpy() / 3600.0
        charge_throughput_ah = np.cumsum(np.r_[0.0, 0.5 * (df["current_a"].to_numpy()[1:] + df["current_a"].to_numpy()[:-1]) * np.diff(time_h)])
        soc = initial_soc - charge_throughput_ah / nominal_capacity_ah
    soc = np.clip(soc, 0.0, 1.0)
    return pd.Series(soc, index=df.index, name="soc_est")


# -----------------------------------------------------------------------------
# Pulse detection
# -----------------------------------------------------------------------------


def detect_hppc_pulses(
    df: pd.DataFrame,
    current_threshold_a: float = 0.2,
    rest_threshold_a: float = 0.05,
    min_pulse_duration_s: float = 2.0,
    min_rest_duration_s: float = 10.0,
    smooth_kernel: int = 5,
) -> List[PulseWindow]:
    """
    Detect pulse windows with a rest segment before and after the pulse.
    """
    t = df["time_s"].to_numpy()
    i = smooth_current(df["current_a"].to_numpy(), kernel_size=smooth_kernel)
    soc = df["soc_est"].to_numpy() if "soc_est" in df.columns else np.full(len(df), np.nan)

    is_pulse = np.abs(i) >= current_threshold_a
    is_rest = np.abs(i) <= rest_threshold_a

    # Build contiguous segments
    segs: List[Tuple[int, int, bool]] = []
    start = 0
    state = bool(is_pulse[0]) if len(is_pulse) else False
    for k in range(1, len(is_pulse)):
        st = bool(is_pulse[k])
        if st != state:
            segs.append((start, k - 1, state))
            start = k
            state = st
    if len(is_pulse):
        segs.append((start, len(is_pulse) - 1, state))

    pulses: List[PulseWindow] = []
    pulse_id = 0
    for idx, (s, e, pulse_state) in enumerate(segs):
        if not pulse_state:
            continue
        duration = t[e] - t[s]
        if duration < min_pulse_duration_s:
            continue

        # Need adjacent rest segments on both sides
        if idx == 0 or idx == len(segs) - 1:
            continue
        rs, re, rst_before_state = segs[idx - 1]
        as_, ae, rst_after_state = segs[idx + 1]
        if rst_before_state or rst_after_state:
            continue

        rest_before_duration = t[re] - t[rs]
        rest_after_duration = t[ae] - t[as_]
        if rest_before_duration < min_rest_duration_s or rest_after_duration < min_rest_duration_s:
            continue

        # Refine rest segments using is_rest mask
        if np.mean(is_rest[rs : re + 1]) < 0.8 or np.mean(is_rest[as_ : ae + 1]) < 0.8:
            continue

        pulse_id += 1
        pulses.append(
            PulseWindow(
                pulse_id=pulse_id,
                start_idx=s,
                end_idx=e,
                rest_before_start_idx=rs,
                rest_before_end_idx=re,
                rest_after_start_idx=as_,
                rest_after_end_idx=ae,
                current_level=float(np.median(i[s : e + 1])),
                duration_s=float(duration),
                soc_est=float(np.nanmedian(soc[s : e + 1])),
            )
        )
    return pulses


# -----------------------------------------------------------------------------
# ECM model and fitting
# -----------------------------------------------------------------------------


def exact_rc_update(v_prev: float, current_a: float, r_ohm: float, c_f: float, dt_s: float) -> float:
    """Exact discrete update for one RC branch under piecewise-constant current."""
    tau = max(r_ohm * c_f, 1e-12)
    alpha = np.exp(-dt_s / tau)
    return alpha * v_prev + r_ohm * (1.0 - alpha) * current_a



def simulate_2rc_voltage(
    time_s: np.ndarray,
    current_a: np.ndarray,
    ocv_v: float,
    r0_ohm: float,
    r1_ohm: float,
    c1_f: float,
    r2_ohm: float,
    c2_f: float,
    v1_init: float = 0.0,
    v2_init: float = 0.0,
) -> np.ndarray:
    v1 = float(v1_init)
    v2 = float(v2_init)
    vt = np.zeros_like(time_s, dtype=float)
    vt[0] = ocv_v - current_a[0] * r0_ohm - v1 - v2
    for k in range(1, len(time_s)):
        dt = max(float(time_s[k] - time_s[k - 1]), 0.0)
        ik = float(current_a[k])
        v1 = exact_rc_update(v1, ik, r1_ohm, c1_f, dt)
        v2 = exact_rc_update(v2, ik, r2_ohm, c2_f, dt)
        vt[k] = ocv_v - ik * r0_ohm - v1 - v2
    return vt



def estimate_r0_from_step(
    df: pd.DataFrame,
    pulse: PulseWindow,
    pre_samples: int = 5,
    post_samples: int = 5,
) -> float:
    s = pulse.start_idx
    pre0 = max(pulse.rest_before_end_idx - pre_samples + 1, pulse.rest_before_start_idx)
    post1 = min(s + post_samples, pulse.end_idx + 1)

    v_pre = df.loc[pre0:pulse.rest_before_end_idx, "voltage_v"].median()
    i_pre = df.loc[pre0:pulse.rest_before_end_idx, "current_a"].median()
    v_post = df.loc[s:post1, "voltage_v"].median()
    i_post = df.loc[s:post1, "current_a"].median()

    di = i_post - i_pre
    dv = v_pre - v_post  # positive if voltage drops after discharge current is applied
    if np.isclose(di, 0.0):
        return 0.01
    r0 = dv / di
    return float(np.clip(r0, 1e-5, 0.5))



def build_fit_window(df: pd.DataFrame, pulse: PulseWindow, include_pre_rest_s: float = 10.0, include_post_rest_s: float = 120.0) -> pd.DataFrame:
    t = df["time_s"].to_numpy()
    t0 = t[pulse.start_idx]
    t1 = t[pulse.end_idx]

    start_time = max(t[pulse.rest_before_start_idx], t0 - include_pre_rest_s)
    end_time = min(t[pulse.rest_after_end_idx], t1 + include_post_rest_s)

    mask = (t >= start_time) & (t <= end_time)
    w = df.loc[mask, ["time_s", "voltage_v", "current_a"]].copy().reset_index(drop=True)
    w["time_s"] = w["time_s"] - w["time_s"].iloc[0]
    return w



def residual_2rc(params: Parameters, time_s: np.ndarray, current_a: np.ndarray, voltage_v: np.ndarray) -> np.ndarray:
    ocv_v = params["ocv_v"].value
    r0 = params["r0_ohm"].value
    r1 = params["r1_ohm"].value
    c1 = params["c1_f"].value
    r2 = params["r2_ohm"].value
    c2 = params["c2_f"].value
    v1_init = params["v1_init_v"].value
    v2_init = params["v2_init_v"].value

    v_model = simulate_2rc_voltage(time_s, current_a, ocv_v, r0, r1, c1, r2, c2, v1_init, v2_init)

    # Mild weighting: emphasize pulse and early recovery slightly
    pulse_like = np.abs(current_a) > 0.1 * max(1e-9, np.max(np.abs(current_a)))
    weights = np.ones_like(voltage_v)
    weights[pulse_like] = 1.5
    if len(weights) > 5:
        weights[: min(20, len(weights))] *= 1.2
    return (v_model - voltage_v) * weights



def make_initial_parameters(window_df: pd.DataFrame, r0_guess: float) -> Parameters:
    p = Parameters()
    ocv_guess = float(window_df["voltage_v"].iloc[0])

    p.add("ocv_v", value=ocv_guess, min=ocv_guess - 0.2, max=ocv_guess + 0.2)
    p.add("r0_ohm", value=max(r0_guess, 1e-4), min=1e-5, max=0.05)

    # Broad but physically reasonable defaults for 18650 cell HPPC fitting.
    p.add("r1_ohm", value=max(2.0 * r0_guess, 5e-4), min=1e-5, max=0.2)
    p.add("c1_f", value=2.0e3, min=10.0, max=5.0e5)
    p.add("r2_ohm", value=max(4.0 * r0_guess, 1e-3), min=1e-5, max=0.5)
    p.add("c2_f", value=2.0e4, min=10.0, max=1.0e6)

    # Initial polarization states
    p.add("v1_init_v", value=0.0, min=-0.5, max=0.5)
    p.add("v2_init_v", value=0.0, min=-0.5, max=0.5)
    return p



def fit_single_pulse_2rc(
    df: pd.DataFrame,
    pulse: PulseWindow,
    include_pre_rest_s: float = 10.0,
    include_post_rest_s: float = 120.0,
    method: str = "least_squares",
) -> Tuple[FitSummary, pd.DataFrame, np.ndarray, str]:
    window_df = build_fit_window(df, pulse, include_pre_rest_s, include_post_rest_s)
    r0_guess = estimate_r0_from_step(df, pulse)
    params = make_initial_parameters(window_df, r0_guess)

    t = window_df["time_s"].to_numpy()
    i = window_df["current_a"].to_numpy()
    v = window_df["voltage_v"].to_numpy()

    result = minimize(
        residual_2rc,
        params,
        args=(t, i, v),
        method=method,
        nan_policy="omit",
    )

    best = result.params
    v_fit = simulate_2rc_voltage(
        t,
        i,
        best["ocv_v"].value,
        best["r0_ohm"].value,
        best["r1_ohm"].value,
        best["c1_f"].value,
        best["r2_ohm"].value,
        best["c2_f"].value,
        best["v1_init_v"].value,
        best["v2_init_v"].value,
    )

    rmse = float(np.sqrt(np.mean((v_fit - v) ** 2)))
    summary = FitSummary(
        pulse_id=pulse.pulse_id,
        soc_est=pulse.soc_est,
        current_level=pulse.current_level,
        r0_ohm=float(best["r0_ohm"].value),
        r1_ohm=float(best["r1_ohm"].value),
        c1_f=float(best["c1_f"].value),
        r2_ohm=float(best["r2_ohm"].value),
        c2_f=float(best["c2_f"].value),
        tau1_s=float(best["r1_ohm"].value * best["c1_f"].value),
        tau2_s=float(best["r2_ohm"].value * best["c2_f"].value),
        rmse_v=rmse,
        success=bool(result.success),
        message=str(result.message),
        nfev=int(getattr(result, "nfev", -1)),
    )
    report = fit_report(result)
    return summary, window_df, v_fit, report


# -----------------------------------------------------------------------------
# Plotting and export
# -----------------------------------------------------------------------------


def plot_fit(window_df: pd.DataFrame, v_fit: np.ndarray, title: str, save_path: Optional[str | Path] = None) -> None:
    t = window_df["time_s"].to_numpy()
    v = window_df["voltage_v"].to_numpy()
    i = window_df["current_a"].to_numpy()

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(t, v, label="Measured Voltage")
    ax1.plot(t, v_fit, "--", label="2RC Fit")
    ax1.set_ylabel("Voltage [V]")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(t, i, label="Current")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Current [A]")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# -----------------------------------------------------------------------------
# End-to-end pipeline
# -----------------------------------------------------------------------------


def run_batch_identification(
    csv_path: str | Path,
    output_dir: str | Path,
    nominal_capacity_ah: float = 2.9,
    initial_soc: float = 1.0,
    current_sign: str = "auto",
    current_threshold_a: float = 0.2,
    rest_threshold_a: float = 0.05,
    min_pulse_duration_s: float = 2.0,
    min_rest_duration_s: float = 10.0,
    include_pre_rest_s: float = 10.0,
    include_post_rest_s: float = 120.0,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_hppc_csv(csv_path, current_sign=current_sign)
    df["soc_est"] = estimate_soc(df, nominal_capacity_ah=nominal_capacity_ah, initial_soc=initial_soc)

    pulses = detect_hppc_pulses(
        df,
        current_threshold_a=current_threshold_a,
        rest_threshold_a=rest_threshold_a,
        min_pulse_duration_s=min_pulse_duration_s,
        min_rest_duration_s=min_rest_duration_s,
    )
    if not pulses:
        raise RuntimeError("No valid HPPC pulses were detected. Check current thresholds and rest durations.")

    summaries: List[FitSummary] = []
    reports: Dict[str, str] = {}

    for pulse in pulses:
        summary, window_df, v_fit, report = fit_single_pulse_2rc(
            df,
            pulse,
            include_pre_rest_s=include_pre_rest_s,
            include_post_rest_s=include_post_rest_s,
        )
        summaries.append(summary)
        reports[f"pulse_{pulse.pulse_id:03d}"] = report

        fit_csv = output_dir / f"pulse_{pulse.pulse_id:03d}_fit_trace.csv"
        window_out = window_df.copy()
        window_out["voltage_fit_v"] = v_fit
        window_out.to_csv(fit_csv, index=False)

        fig_path = output_dir / f"pulse_{pulse.pulse_id:03d}_fit.png"
        plot_fit(window_df, v_fit, f"Pulse {pulse.pulse_id} | SOC≈{pulse.soc_est:.3f}", fig_path)

    summary_df = pd.DataFrame([asdict(s) for s in summaries])
    summary_df = summary_df.sort_values("soc_est", ascending=False).reset_index(drop=True)
    summary_df.to_csv(output_dir / "2rc_fit_summary.csv", index=False)

    with open(output_dir / "fit_reports.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)

    # Save preprocessed data and detected pulse index table for inspection
    df.to_csv(output_dir / "preprocessed_hppc.csv", index=False)
    pulse_table = pd.DataFrame([asdict(p) for p in pulses])
    pulse_table.to_csv(output_dir / "detected_pulses.csv", index=False)
    return summary_df


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="2RC parameter identification from HPPC CSV data.")
    p.add_argument("csv_path", type=str, help="Path to HPPC CSV file")
    p.add_argument("--output-dir", type=str, default="hppc_2rc_results", help="Output directory")
    p.add_argument("--nominal-capacity-ah", type=float, default=2.9, help="Nominal cell capacity in Ah")
    p.add_argument("--initial-soc", type=float, default=1.0, help="Initial SOC at start of dataset")
    p.add_argument(
        "--current-sign",
        type=str,
        default="auto",
        choices=["auto", "discharge_positive", "discharge_negative"],
        help="Current sign convention in the source CSV",
    )
    p.add_argument("--current-threshold-a", type=float, default=0.2, help="|I| threshold for pulse detection [A]")
    p.add_argument("--rest-threshold-a", type=float, default=0.05, help="|I| threshold for rest detection [A]")
    p.add_argument("--min-pulse-duration-s", type=float, default=2.0, help="Minimum pulse duration [s]")
    p.add_argument("--min-rest-duration-s", type=float, default=10.0, help="Minimum rest duration [s]")
    p.add_argument("--include-pre-rest-s", type=float, default=10.0, help="Rest before pulse used in fit window [s]")
    p.add_argument("--include-post-rest-s", type=float, default=120.0, help="Rest after pulse used in fit window [s]")
    return p



def main() -> None:
    args = build_argparser().parse_args()
    summary_df = run_batch_identification(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        nominal_capacity_ah=args.nominal_capacity_ah,
        initial_soc=args.initial_soc,
        current_sign=args.current_sign,
        current_threshold_a=args.current_threshold_a,
        rest_threshold_a=args.rest_threshold_a,
        min_pulse_duration_s=args.min_pulse_duration_s,
        min_rest_duration_s=args.min_rest_duration_s,
        include_pre_rest_s=args.include_pre_rest_s,
        include_post_rest_s=args.include_post_rest_s,
    )
    print("\n=== 2RC Fit Summary ===")
    print(summary_df.to_string(index=False))
    print("\nSaved results to output directory.")


if __name__ == "__main__":
    main()
