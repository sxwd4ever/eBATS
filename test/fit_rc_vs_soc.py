import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ========= User settings =========
CSV_FILE = "results_2rc_groupwise/2rc_group_fit_summary.csv"
OUTPUT_DIR = "soc_fit_results_v2"

# Prefer group summary after groupwise fitting:
#   "2rc_group_fit_summary.csv"   -> use SOC_COL = "soc_mean"
#   "2rc_per_pulse_from_group_fit.csv" -> use SOC_COL = "soc_test"
SOC_COL = "soc_mean"

PARAM_COLS = ["r0_ohm", "r1_ohm", "c1_f", "r2_ohm", "c2_f"]

POLY_DEGREE = 4
FIT_MODE = "poly"   # "log_poly" or "poly"

# Optional filtering
USE_ERROR_FILTER = False
ERROR_COL = "rmse_v"
MAX_ERROR = 0.01
# =================================


def fit_poly(soc: np.ndarray, y: np.ndarray, degree: int):
    """
    Fit in y-SOC space:
        y = a_n*soc^n + ... + a_0
    """
    mask = np.isfinite(soc) & np.isfinite(y)
    x = soc[mask]
    z = y[mask]

    if len(x) < degree + 2:
        raise ValueError(f"Not enough valid points for degree-{degree} fitting.")

    coeffs = np.polyfit(x, z, degree)
    y_fit = np.polyval(coeffs, x)

    err = y_fit - z
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((z - np.mean(z)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))

    metrics = {
        "n_points": int(len(x)),
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
    }
    return x, z, coeffs, y_fit, metrics


def fit_log_poly(soc: np.ndarray, y: np.ndarray, degree: int):
    """
    Fit in log(y)-SOC space:
        log(y) = a_n*soc^n + ... + a_0
    =>  y = exp(poly(soc))
    This keeps predicted R/C values positive.
    """
    mask = np.isfinite(soc) & np.isfinite(y) & (y > 0)
    x = soc[mask]
    z = y[mask]

    if len(x) < degree + 2:
        raise ValueError(f"Not enough valid positive points for degree-{degree} fitting.")

    coeffs = np.polyfit(x, np.log(z), degree)
    y_fit = np.exp(np.polyval(coeffs, x))

    err = y_fit - z
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((z - np.mean(z)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))

    metrics = {
        "n_points": int(len(x)),
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
    }
    return x, z, coeffs, y_fit, metrics


def coeffs_to_expression(coeffs: np.ndarray, var: str = "soc", use_exp: bool = True) -> str:
    """
    Convert polyfit coefficients to expression string.

    If use_exp=True:
        exp(a_n*soc^n + ... + a_0)
    else:
        a_n*soc^n + ... + a_0
    """
    degree = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        power = degree - i
        if power == 0:
            terms.append(f"({c:.12e})")
        elif power == 1:
            terms.append(f"({c:.12e})*{var}")
        else:
            terms.append(f"({c:.12e})*{var}**{power}")
    poly_expr = " + ".join(terms)
    return f"np.exp({poly_expr})" if use_exp else f"({poly_expr})"


def make_python_function_text(func_name: str, coeffs: np.ndarray, use_exp: bool) -> str:
    expr = coeffs_to_expression(coeffs, var="soc", use_exp=use_exp)
    return (
        f"def {func_name}(soc):\n"
        f"    soc = np.asarray(soc, dtype=float)\n"
        f"    soc = np.clip(soc, 0.0, 1.0)\n"
        f"    return {expr}\n"
    )


def auto_prepare_dataframe(df: pd.DataFrame, soc_col: str):
    """
    Keep only required columns and merge repeated SOC values if needed.
    For groupwise output, SOC should already be unique.
    For per-pulse output, repeated SOCs are averaged.
    """
    keep_cols = [soc_col] + PARAM_COLS + ([ERROR_COL] if ERROR_COL in df.columns else [])
    missing = [c for c in [soc_col] + PARAM_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = df[keep_cols].copy()

    if USE_ERROR_FILTER:
        if ERROR_COL not in work.columns:
            raise KeyError(f"Filtering enabled but '{ERROR_COL}' is missing.")
        work = work[work[ERROR_COL] <= MAX_ERROR].copy()

    # If there are repeated SOC values, average them
    agg_dict = {p: "mean" for p in PARAM_COLS}
    if ERROR_COL in work.columns:
        agg_dict[ERROR_COL] = "mean"

    work = work.groupby(soc_col, as_index=False).agg(agg_dict)
    work = work.sort_values(soc_col).reset_index(drop=True)
    return work


def main():
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_FILE)
    df = auto_prepare_dataframe(df, SOC_COL)

    summary_rows = []
    coeff_dict = {}
    function_blocks = ["import numpy as np\n\n"]

    use_exp = (FIT_MODE == "log_poly")

    for param in PARAM_COLS:
        soc = df[SOC_COL].to_numpy(dtype=float)
        y = df[param].to_numpy(dtype=float)

        if FIT_MODE == "log_poly":
            x, z, coeffs, y_fit, metrics = fit_log_poly(soc, y, POLY_DEGREE)
            y_plot_func = lambda xx: np.exp(np.polyval(coeffs, xx))
        elif FIT_MODE == "poly":
            x, z, coeffs, y_fit, metrics = fit_poly(soc, y, POLY_DEGREE)
            y_plot_func = lambda xx: np.polyval(coeffs, xx)
        else:
            raise ValueError("FIT_MODE must be 'log_poly' or 'poly'.")

        coeff_dict[param] = coeffs.tolist()

        expr = coeffs_to_expression(coeffs, var="soc", use_exp=use_exp)
        summary_rows.append({
            "parameter": param,
            "degree": POLY_DEGREE,
            "fit_mode": FIT_MODE,
            "n_points": metrics["n_points"],
            "r2": metrics["r2"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "expression": expr,
        })

        function_name = param.replace("_ohm", "").replace("_f", "")
        function_blocks.append(make_python_function_text(function_name, coeffs, use_exp=use_exp))
        function_blocks.append("\n")

        x_plot = np.linspace(np.min(x), np.max(x), 400)
        y_plot = y_plot_func(x_plot)

        plt.figure(figsize=(6, 4))
        plt.scatter(x, z, s=25, label="Identified data")
        plt.plot(x_plot, y_plot, label=f"{FIT_MODE}, degree={POLY_DEGREE}")
        plt.xlabel("SOC")
        plt.ylabel(param)
        plt.title(f"{param} vs SOC")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"{param}_vs_soc.png", dpi=300)
        plt.close()

        fit_points_df = pd.DataFrame({
            "soc": x,
            f"{param}_raw": z,
            f"{param}_fit": y_fit,
            f"{param}_rel_error_pct": (y_fit - z) / z * 100.0,
        })
        fit_points_df.to_csv(outdir / f"{param}_fit_points.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "soc_fit_summary.csv", index=False)

    with open(outdir / "soc_fit_coefficients.json", "w", encoding="utf-8") as f:
        json.dump({
            "fit_mode": FIT_MODE,
            "degree": POLY_DEGREE,
            "soc_col": SOC_COL,
            "source_csv": CSV_FILE,
            "coefficients": coeff_dict,
        }, f, indent=2)

    with open(outdir / "rc_soc_functions.py", "w", encoding="utf-8") as f:
        f.writelines(function_blocks)

    print(f"Done. Results saved to: {outdir.resolve()}")
    print("Generated files:")
    print("  - soc_fit_summary.csv")
    print("  - soc_fit_coefficients.json")
    print("  - rc_soc_functions.py")
    for p in PARAM_COLS:
        print(f"  - {p}_vs_soc.png")
        print(f"  - {p}_fit_points.csv")


if __name__ == "__main__":
    main()