import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ========= 用户可改参数 =========
CSV_FILE = "results_2rc/2rc_fit_summary.csv"
OUTPUT_DIR = "soc_fit_results"
POLY_DEGREE = 5

SOC_COL = "soc_est"
PARAM_COLS = ["r0_ohm", "r1_ohm", "c1_f", "r2_ohm", "c2_f"]

# 可选：按拟合误差筛选脉冲
USE_RMSE_FILTER = False
RMSE_COL = "rmse_v"
MAX_RMSE = 0.005
# ==============================


def fit_log_poly(soc: np.ndarray, y: np.ndarray, degree: int):
    """
    在 log(y)-SOC 空间做多项式拟合:
        log(y) = a_n*soc^n + ... + a_0
    返回:
        coeffs: np.ndarray, 按 np.polyval 顺序排列
        y_fit:  拟合值
        metrics: dict
    """
    mask = np.isfinite(soc) & np.isfinite(y) & (y > 0)
    x = soc[mask]
    z = y[mask]

    if len(x) < degree + 2:
        raise ValueError(f"有效数据点不足，无法进行 {degree} 阶拟合。")

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


def coeffs_to_expression(coeffs: np.ndarray, var: str = "soc") -> str:
    """
    将 np.polyfit 系数转换为字符串表达式:
        exp(a5*soc**5 + a4*soc**4 + ... + a0)
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
    return f"np.exp({poly_expr})"


def make_python_function_text(func_name: str, coeffs: np.ndarray) -> str:
    expr = coeffs_to_expression(coeffs, var="soc")
    return (
        f"def {func_name}(soc):\n"
        f"    soc = np.asarray(soc)\n"
        f"    return {expr}\n"
    )


def main():
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_FILE)

    required_cols = [SOC_COL] + PARAM_COLS
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"CSV 中缺少列: {col}")

    if USE_RMSE_FILTER:
        if RMSE_COL not in df.columns:
            raise KeyError(f"启用了 RMSE 筛选，但 CSV 中缺少列: {RMSE_COL}")
        df = df[df[RMSE_COL] <= MAX_RMSE].copy()

    df = df.sort_values(SOC_COL).reset_index(drop=True)

    summary_rows = []
    coeff_dict = {}
    function_blocks = ["import numpy as np\n\n"]

    for param in PARAM_COLS:
        soc = df[SOC_COL].to_numpy(dtype=float)
        y = df[param].to_numpy(dtype=float)

        x, z, coeffs, y_fit, metrics = fit_log_poly(soc, y, POLY_DEGREE)
        coeff_dict[param] = coeffs.tolist()

        expr = coeffs_to_expression(coeffs, var="soc")
        summary_rows.append({
            "parameter": param,
            "degree": POLY_DEGREE,
            "n_points": metrics["n_points"],
            "r2": metrics["r2"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "expression": expr,
        })

        function_name = param.replace("_ohm", "").replace("_f", "")
        function_blocks.append(make_python_function_text(function_name, coeffs))
        function_blocks.append("\n")

        x_plot = np.linspace(np.min(x), np.max(x), 400)
        y_plot = np.exp(np.polyval(coeffs, x_plot))

        plt.figure(figsize=(6, 4))
        plt.scatter(x, z, s=20, label="Identified data")
        plt.plot(x_plot, y_plot, label=f"Log-poly fit (deg={POLY_DEGREE})")
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
        json.dump(coeff_dict, f, indent=2)

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