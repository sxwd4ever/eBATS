import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.interpolate import CubicSpline


# ============================================================
# User settings
# ============================================================
MAT_FILE = "data/05-08-17_13.26 C20 OCV Test_C20_25dC.mat"
OUTPUT_DIR = "ocv_fit_results"

Q_NOM_AH = 2.90          # nominal capacity of Panasonic 18650PF
POLY_DEGREE = 4          # polynomial degree for optional polynomial fit
CURRENT_EPS = 0.02       # threshold [A] to classify rest region
USE_SPLINE_AS_MAIN = True  # True: spline output as main curve; False: polynomial as main curve


# ============================================================
# MAT loader
# ============================================================
def load_mat_file(mat_path):
    data = sio.loadmat(mat_path, squeeze_me=False, struct_as_record=False)
    return data


def find_main_struct(data):
    """
    Try to find the main MATLAB struct containing the measurement fields.
    Typical candidates: 'meas', 'data', etc.
    """
    candidate_keys = []
    for k, v in data.items():
        if k.startswith("__"):
            continue
        candidate_keys.append(k)

    if not candidate_keys:
        raise KeyError("No user data found in the .mat file.")

    # Prefer common names
    for name in ["meas", "data", "dataset", "test"]:
        if name in data:
            return name, data[name]

    # Otherwise use the first non-meta key
    return candidate_keys[0], data[candidate_keys[0]]


def unwrap_mat_struct(obj):
    """
    Robustly unwrap MATLAB structs loaded by scipy.io.loadmat.
    """
    while isinstance(obj, np.ndarray) and obj.size == 1:
        obj = obj[0, 0] if obj.ndim == 2 else obj.flat[0]
    return obj


def extract_field(struct_obj, field_name_candidates):
    """
    Extract a field from a MATLAB struct-like object using candidate names.
    """
    obj = unwrap_mat_struct(struct_obj)

    # Case 1: matlab struct object with attributes
    for name in field_name_candidates:
        if hasattr(obj, name):
            val = getattr(obj, name)
            return np.asarray(val).squeeze()

    # Case 2: numpy structured array with dtype names
    if hasattr(obj, "dtype") and obj.dtype.names is not None:
        lower_map = {n.lower(): n for n in obj.dtype.names}
        for name in field_name_candidates:
            if name.lower() in lower_map:
                val = obj[lower_map[name.lower()]]
                return np.asarray(val).squeeze()

    # Case 3: dict-like
    if isinstance(obj, dict):
        lower_map = {k.lower(): k for k in obj.keys()}
        for name in field_name_candidates:
            if name.lower() in lower_map:
                val = obj[lower_map[name.lower()]]
                return np.asarray(val).squeeze()

    raise KeyError(f"Cannot find any field among {field_name_candidates}")


def load_measurements_from_mat(mat_path):
    data = load_mat_file(mat_path)
    root_name, root_obj = find_main_struct(data)

    time = extract_field(root_obj, ["Time", "time", "Test_Time"])
    voltage = extract_field(root_obj, ["Voltage", "voltage", "Volt"])
    current = extract_field(root_obj, ["Current", "current", "Curr"])
    ah = extract_field(root_obj, ["Ah", "ah", "AmpHour"])

    df = pd.DataFrame({
        "time_s": np.asarray(time, dtype=float).flatten(),
        "voltage_v": np.asarray(voltage, dtype=float).flatten(),
        "current_a": np.asarray(current, dtype=float).flatten(),
        "ah": np.asarray(ah, dtype=float).flatten(),
    })

    df = df.dropna().reset_index(drop=True)
    df["time_s"] = df["time_s"] - df["time_s"].iloc[0]
    return df, root_name


# ============================================================
# OCV extraction
# ============================================================
def estimate_soc(df, q_nom_ah):
    """
    Estimate SOC from Ah.
    Assumes Ah increases with discharged capacity during discharge portions.
    For a full C/20 test containing both charge and discharge, we normalize by Ah range.
    """
    out = df.copy()

    ah = out["ah"].to_numpy(dtype=float)
    ah_min = np.min(ah)
    ah_max = np.max(ah)

    if ah_max - ah_min < 1e-6:
        raise ValueError("Ah range is too small to estimate SOC.")

    # Normalize using Ah span
    soc = 1.0 - (ah - ah_min) / (ah_max - ah_min)
    soc = np.clip(soc, 0.0, 1.0)
    out["soc"] = soc
    return out


def split_charge_discharge(df, current_eps=0.02):
    """
    Positive/negative current sign may vary by dataset.
    We identify the two active branches by sign and ignore near-zero current.
    """
    out = df.copy()

    active = np.abs(out["current_a"]) > current_eps
    work = out[active].copy()

    if work.empty:
        raise ValueError("No active current region found. CURRENT_EPS may be too large.")

    # Determine dominant positive and negative branches
    pos = work[work["current_a"] > 0].copy()
    neg = work[work["current_a"] < 0].copy()

    # We do not assume which one is charge/discharge physically.
    # Instead, later we sort each branch by SOC and average by SOC bins.
    return pos, neg


def smooth_branch_to_bins(branch_df, n_bins=101):
    """
    Bin data onto a common SOC grid and average voltage.
    """
    if branch_df.empty:
        return pd.DataFrame(columns=["soc", "voltage_v"])

    soc_grid = np.linspace(0.0, 1.0, n_bins)
    soc = branch_df["soc"].to_numpy(dtype=float)
    v = branch_df["voltage_v"].to_numpy(dtype=float)

    # Sort by SOC
    order = np.argsort(soc)
    soc = soc[order]
    v = v[order]

    rows = []
    for s in soc_grid:
        mask = np.abs(soc - s) <= 0.01
        if np.any(mask):
            rows.append({"soc": s, "voltage_v": np.mean(v[mask])})

    return pd.DataFrame(rows)


def build_ocv_curve(df, current_eps=0.02):
    """
    Build an averaged OCV curve from the two low-rate branches.
    """
    pos, neg = split_charge_discharge(df, current_eps=current_eps)

    pos_b = smooth_branch_to_bins(pos)
    neg_b = smooth_branch_to_bins(neg)

    merged = pd.merge(
        pos_b, neg_b, on="soc", how="outer", suffixes=("_pos", "_neg")
    ).sort_values("soc").reset_index(drop=True)

    # Average available branches
    merged["ocv_avg_v"] = merged[["voltage_v_pos", "voltage_v_neg"]].mean(axis=1, skipna=True)
    merged = merged.dropna(subset=["ocv_avg_v"]).reset_index(drop=True)

    if len(merged) < 6:
        raise ValueError("Too few valid OCV points after branch merging.")

    return merged, pos, neg


# ============================================================
# Fitting
# ============================================================
def fit_ocv_poly(soc, ocv, degree=5):
    coeffs = np.polyfit(soc, ocv, degree)
    ocv_fit = np.polyval(coeffs, soc)

    err = ocv_fit - ocv
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((ocv - np.mean(ocv)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return coeffs, ocv_fit, {"rmse": rmse, "mae": mae, "r2": r2}


def fit_ocv_spline(soc, ocv):
    order = np.argsort(soc)
    soc = soc[order]
    ocv = ocv[order]

    # Remove duplicate SOC points by averaging
    temp = pd.DataFrame({"soc": soc, "ocv": ocv}).groupby("soc", as_index=False).mean()
    spline = CubicSpline(temp["soc"].to_numpy(), temp["ocv"].to_numpy(), bc_type="natural")
    ocv_fit = spline(temp["soc"].to_numpy())

    err = ocv_fit - temp["ocv"].to_numpy()
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((temp["ocv"] - np.mean(temp["ocv"])) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return spline, temp, {"rmse": rmse, "mae": mae, "r2": r2}


def poly_coeffs_to_expression(coeffs, var="soc"):
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
    return " + ".join(terms)


# ============================================================
# Export
# ============================================================
def export_python_function(outdir, poly_coeffs=None, spline_table=None, use_spline=True):
    lines = ["import numpy as np\n\n"]

    if poly_coeffs is not None:
        expr = poly_coeffs_to_expression(poly_coeffs, var="soc")
        lines.append("def ocv_poly(soc):\n")
        lines.append("    soc = np.asarray(soc, dtype=float)\n")
        lines.append("    soc = np.clip(soc, 0.0, 1.0)\n")
        lines.append(f"    return {expr}\n\n")

    if spline_table is not None:
        soc_list = spline_table["soc"].to_list()
        ocv_list = spline_table["ocv"].to_list()

        lines.append(f"_soc_table = np.array({soc_list}, dtype=float)\n")
        lines.append(f"_ocv_table = np.array({ocv_list}, dtype=float)\n\n")
        lines.append("def ocv_spline_like(soc):\n")
        lines.append("    soc = np.asarray(soc, dtype=float)\n")
        lines.append("    soc = np.clip(soc, 0.0, 1.0)\n")
        lines.append("    return np.interp(soc, _soc_table, _ocv_table)\n\n")

    if use_spline and spline_table is not None:
        lines.append("def ocv(soc):\n")
        lines.append("    return ocv_spline_like(soc)\n")
    elif poly_coeffs is not None:
        lines.append("def ocv(soc):\n")
        lines.append("    return ocv_poly(soc)\n")

    with open(outdir / "ocv_functions.py", "w", encoding="utf-8") as f:
        f.writelines(lines)


# ============================================================
# Main
# ============================================================
def main():
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    df, root_name = load_measurements_from_mat(MAT_FILE)
    df = estimate_soc(df, Q_NOM_AH)

    ocv_df, pos_branch, neg_branch = build_ocv_curve(df, current_eps=CURRENT_EPS)

    soc = ocv_df["soc"].to_numpy(dtype=float)
    ocv = ocv_df["ocv_avg_v"].to_numpy(dtype=float)

    poly_coeffs, ocv_poly_fit, poly_metrics = fit_ocv_poly(soc, ocv, degree=POLY_DEGREE)
    spline, spline_table, spline_metrics = fit_ocv_spline(soc, ocv)

    # Save raw/processed data
    df.to_csv(outdir / "mat_extracted_data.csv", index=False)
    pos_branch.to_csv(outdir / "branch_positive.csv", index=False)
    neg_branch.to_csv(outdir / "branch_negative.csv", index=False)
    ocv_df.to_csv(outdir / "ocv_points.csv", index=False)

    # Save metrics and coefficients
    export_data = {
        "source_mat_file": MAT_FILE,
        "root_struct_name": root_name,
        "poly_degree": POLY_DEGREE,
        "poly_coefficients": poly_coeffs.tolist(),
        "poly_metrics": poly_metrics,
        "spline_metrics": spline_metrics,
    }
    with open(outdir / "ocv_fit_summary.json", "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)

    export_python_function(
        outdir,
        poly_coeffs=poly_coeffs,
        spline_table=spline_table,
        use_spline=USE_SPLINE_AS_MAIN
    )

    # Plot 1: extracted branches
    plt.figure(figsize=(7, 5))
    if not pos_branch.empty:
        plt.scatter(pos_branch["soc"], pos_branch["voltage_v"], s=8, label="Branch A")
    if not neg_branch.empty:
        plt.scatter(neg_branch["soc"], neg_branch["voltage_v"], s=8, label="Branch B")
    plt.xlabel("SOC")
    plt.ylabel("Voltage [V]")
    plt.title("C/20 charge-discharge branches")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "ocv_branches.png", dpi=300)
    plt.close()

    # Plot 2: averaged OCV + fits
    x_plot = np.linspace(0.0, 1.0, 500)
    y_poly = np.polyval(poly_coeffs, x_plot)
    y_spline = spline(x_plot)

    plt.figure(figsize=(7, 5))
    plt.scatter(soc, ocv, s=20, label="Averaged OCV points")
    plt.plot(x_plot, y_poly, label=f"Polynomial fit (deg={POLY_DEGREE})")
    plt.plot(x_plot, y_spline, label="Cubic spline fit")
    plt.xlabel("SOC")
    plt.ylabel("OCV [V]")
    plt.title("OCV-SOC fitting")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "ocv_soc_fit.png", dpi=300)
    plt.close()

    # Plot 3: polynomial fit error
    plt.figure(figsize=(7, 4))
    plt.plot(soc, ocv_poly_fit - ocv, "o-", label="Polynomial fit error")
    plt.xlabel("SOC")
    plt.ylabel("Error [V]")
    plt.title("Polynomial OCV fit error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "ocv_poly_fit_error.png", dpi=300)
    plt.close()

    print(f"Done. Results saved to: {outdir.resolve()}")
    print("Generated files:")
    print("  - mat_extracted_data.csv")
    print("  - branch_positive.csv")
    print("  - branch_negative.csv")
    print("  - ocv_points.csv")
    print("  - ocv_fit_summary.json")
    print("  - ocv_functions.py")
    print("  - ocv_branches.png")
    print("  - ocv_soc_fit.png")
    print("  - ocv_poly_fit_error.png")


if __name__ == "__main__":
    main()