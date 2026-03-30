import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 用户参数
# ============================================================
COEFF_JSON = "soc_fit_results/soc_fit_coefficients.json"

# 电芯参数（可按你的 NPF 电芯修正）
Q_NOM_AH = 2.90           # 额定容量 [Ah]
C_RATE = 3.0              # 放电倍率
SOC_INIT = 1.00           # 初始 SOC
SOC_END = 0.05            # 终止 SOC
DT = 1.0                  # 时间步长 [s]

# 热参数：集中参数模型
T_INIT_C = 25.0           # 初始温度 [°C]
T_AMB_C = 25.0            # 环境温度 [°C]
M_CELL = 0.045            # 电芯质量 [kg]，18650 常见约 45 g
CP_CELL = 1000.0          # 比热容 [J/(kg·K)]，经验值 900~1100
H_A = 0.12                # 总散热系数 hA [W/K]，自然对流下先给经验值

OUTPUT_DIR = "sim_3C_heat_results"


# ============================================================
# 工具函数
# ============================================================
def build_param_func(coeffs):
    """
    根据 log-polynomial 系数构造参数函数：
        p(soc) = exp(polyval(coeffs, soc))
    """
    coeffs = np.asarray(coeffs, dtype=float)

    def f(soc):
        soc = np.asarray(soc)
        soc = np.clip(soc, 0.0, 1.0)
        return np.exp(np.polyval(coeffs, soc))

    return f


def load_param_functions(coeff_json_path):
    with open(coeff_json_path, "r", encoding="utf-8") as f:
        coeff_data = json.load(f)

    required = ["r0_ohm", "r1_ohm", "c1_f", "r2_ohm", "c2_f"]
    for key in required:
        if key not in coeff_data:
            raise KeyError(f"缺少参数: {key}")

    funcs = {k: build_param_func(v) for k, v in coeff_data.items()}
    return funcs


# ============================================================
# 主仿真：3C 恒流放电 + 2RC 热生热模型
# ============================================================
def simulate_3c_heat(
    funcs,
    q_nom_ah=2.9,
    c_rate=3.0,
    soc_init=1.0,
    soc_end=0.05,
    dt=1.0,
    t_init_c=25.0,
    t_amb_c=25.0,
    m_cell=0.045,
    cp_cell=1000.0,
    hA=0.12,
):
    """
    常电流放电：
        I = 3C * Q_nom

    2RC 电压状态：
        dV1/dt = -V1/(R1*C1) + I/C1
        dV2/dt = -V2/(R2*C2) + I/C2

    不可逆热：
        q_gen = I^2*R0 + I*V1 + I*V2

    热模型：
        m*cp*dT/dt = q_gen - hA*(T - Tamb)
    """

    I = c_rate * q_nom_ah   # [A]，这里按放电电流正值处理

    # 估算总时长
    discharge_hours = (soc_init - soc_end) / c_rate
    t_end = discharge_hours * 3600.0
    n_steps = int(np.floor(t_end / dt)) + 1

    # 状态初始化
    t_arr = np.zeros(n_steps)
    soc_arr = np.zeros(n_steps)
    T_arr = np.zeros(n_steps)
    V1_arr = np.zeros(n_steps)
    V2_arr = np.zeros(n_steps)
    R0_arr = np.zeros(n_steps)
    R1_arr = np.zeros(n_steps)
    C1_arr = np.zeros(n_steps)
    R2_arr = np.zeros(n_steps)
    C2_arr = np.zeros(n_steps)
    q_ohmic_arr = np.zeros(n_steps)
    q_pol1_arr = np.zeros(n_steps)
    q_pol2_arr = np.zeros(n_steps)
    q_gen_arr = np.zeros(n_steps)

    soc = soc_init
    T = t_init_c
    V1 = 0.0
    V2 = 0.0

    for k in range(n_steps):
        t = k * dt

        # 记录当前状态
        t_arr[k] = t
        soc_arr[k] = soc
        T_arr[k] = T
        V1_arr[k] = V1
        V2_arr[k] = V2

        # 参数随 SOC 变化
        R0 = float(funcs["r0_ohm"](soc))
        R1 = float(funcs["r1_ohm"](soc))
        C1 = float(funcs["c1_f"](soc))
        R2 = float(funcs["r2_ohm"](soc))
        C2 = float(funcs["c2_f"](soc))

        R0_arr[k] = R0
        R1_arr[k] = R1
        C1_arr[k] = C1
        R2_arr[k] = R2
        C2_arr[k] = C2

        # 生热功率
        q_ohmic = I**2 * R0
        q_pol1 = I * V1
        q_pol2 = I * V2
        q_gen = q_ohmic + q_pol1 + q_pol2

        q_ohmic_arr[k] = q_ohmic
        q_pol1_arr[k] = q_pol1
        q_pol2_arr[k] = q_pol2
        q_gen_arr[k] = q_gen

        # 最后一个点不再更新
        if k == n_steps - 1:
            break

        # ===== 2RC 状态更新 =====
        tau1 = max(R1 * C1, 1e-12)
        tau2 = max(R2 * C2, 1e-12)

        # 解析离散
        a1 = np.exp(-dt / tau1)
        a2 = np.exp(-dt / tau2)

        V1 = a1 * V1 + R1 * (1.0 - a1) * I
        V2 = a2 * V2 + R2 * (1.0 - a2) * I

        # ===== 温度更新 =====
        dTdt = (q_gen - hA * (T - t_amb_c)) / (m_cell * cp_cell)
        T = T + dTdt * dt

        # ===== SOC 更新 =====
        soc = soc - I * dt / (q_nom_ah * 3600.0)
        soc = max(soc, soc_end)

    df = pd.DataFrame({
        "time_s": t_arr,
        "time_min": t_arr / 60.0,
        "soc": soc_arr,
        "temp_c": T_arr,
        "r0_ohm": R0_arr,
        "r1_ohm": R1_arr,
        "c1_f": C1_arr,
        "r2_ohm": R2_arr,
        "c2_f": C2_arr,
        "v1_v": V1_arr,
        "v2_v": V2_arr,
        "q_ohmic_w": q_ohmic_arr,
        "q_pol1_w": q_pol1_arr,
        "q_pol2_w": q_pol2_arr,
        "q_gen_w": q_gen_arr,
    })

    return df


# ============================================================
# 输出与绘图
# ============================================================
def save_and_plot(df, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df.to_csv(outdir / "sim_3C_heat.csv", index=False)

    summary = {
        "max_heat_w": float(df["q_gen_w"].max()),
        "mean_heat_w": float(df["q_gen_w"].mean()),
        "max_temp_c": float(df["temp_c"].max()),
        "final_temp_c": float(df["temp_c"].iloc[-1]),
        "final_soc": float(df["soc"].iloc[-1]),
        "sim_time_s": float(df["time_s"].iloc[-1]),
    }

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 图1：生热功率
    plt.figure(figsize=(8, 4.5))
    plt.plot(df["time_min"], df["q_gen_w"], label="Total heat generation")
    plt.plot(df["time_min"], df["q_ohmic_w"], "--", label="Ohmic heat")
    plt.plot(df["time_min"], df["q_pol1_w"], "--", label="Polarization heat 1")
    plt.plot(df["time_min"], df["q_pol2_w"], "--", label="Polarization heat 2")
    plt.xlabel("Time [min]")
    plt.ylabel("Heat generation [W]")
    plt.title("3C discharge heat generation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "heat_generation_vs_time.png", dpi=300)
    plt.close()

    # 图2：温度
    plt.figure(figsize=(8, 4.5))
    plt.plot(df["time_min"], df["temp_c"])
    plt.xlabel("Time [min]")
    plt.ylabel("Temperature [°C]")
    plt.title("Cell temperature during 3C discharge")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "temperature_vs_time.png", dpi=300)
    plt.close()

    # 图3：生热功率-SOC
    plt.figure(figsize=(8, 4.5))
    plt.plot(df["soc"], df["q_gen_w"])
    plt.xlabel("SOC [-]")
    plt.ylabel("Heat generation [W]")
    plt.title("Heat generation vs SOC")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "heat_generation_vs_soc.png", dpi=300)
    plt.close()

    print(f"Results saved to: {outdir.resolve()}")
    print(json.dumps(summary, indent=2))


# ============================================================
# main
# ============================================================
def main():
    funcs = load_param_functions(COEFF_JSON)

    df = simulate_3c_heat(
        funcs=funcs,
        q_nom_ah=Q_NOM_AH,
        c_rate=C_RATE,
        soc_init=SOC_INIT,
        soc_end=SOC_END,
        dt=DT,
        t_init_c=T_INIT_C,
        t_amb_c=T_AMB_C,
        m_cell=M_CELL,
        cp_cell=CP_CELL,
        hA=H_A,
    )

    save_and_plot(df, OUTPUT_DIR)


if __name__ == "__main__":
    main()