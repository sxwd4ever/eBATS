import math
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# 失速速度计算相关
def calculate_stall_speed(W, rho, S):
    """
    此函数用于计算失速速度
    :param W: 飞机重量
    :param rho: 空气密度，会随飞行高度变化
    :param S: 机翼总面积
    :return: 失速速度
    """
    # 标准机翼形状的最大升力系数
    C_L_max = 1.3
    # 失速速度公式
    V_stall = math.sqrt((2 * W) / (C_L_max * rho * S))
    return V_stall


# 大气参数计算相关
# 定义常量
L = -6.5 * 10 ** (-3)  # 温度递减率 (Km^-1)
T0 = 288.15  # 海平面地面温度 (K)
P0 = 101325  # 海平面标准大气压力 (Pa)
M = 0.0289652  # 干空气的摩尔质量 (kg/mol)
R = 8.31446  # 气体常数 (J/(mol·K))
g = 9.80665  # 重力加速度 (m/s^2)


def calculate_temperature(z):
    """
    此函数用于计算指定高度 z 处的温度
    :param z: 高度，单位为米 (m)
    :return: 高度 z 处的温度，单位为开尔文 (K)
    """
    return T0 - L * z / 1000


def calculate_density(z):
    """
    此函数用于计算指定高度 z 处的空气密度
    :param z: 高度，单位为米 (m)
    :return: 高度 z 处的空气密度，单位为千克每立方米 (kg/m^3)
    """
    return (P0 * M) / (R * T0) * ((1 - (L * z / T0)) ** ((g * M) / (R * L) - 1))


# 计算爬升功率
def calculate_climb_power(W, V_x, V_y, L_D_climb, eta_mech, eta_prop):
    """
    计算爬升功率
    :param W: 飞机重量
    :param V_x: 水平速度
    :param V_y: 垂直速度
    :param L_D_climb: 爬升阶段的升阻比
    :param eta_mech: 机械效率
    :param eta_prop: 推进效率
    :return: 爬升功率
    """
    return (W * V_y + (W * V_x) / L_D_climb) / (eta_mech * eta_prop)


# 计算下降功率
def calculate_descent_power(W, V_x, V_y, L_D_descent, eta_mech, eta_prop):
    """
    计算下降功率
    :param W: 飞机重量
    :param V_x: 水平速度
    :param V_y: 垂直速度
    :param L_D_descent: 下降阶段的升阻比
    :param eta_mech: 机械效率
    :param eta_prop: 推进效率
    :return: 下降功率
    """
    return (W * V_y + (W * V_x) / L_D_descent) / (eta_mech * eta_prop)


# 示例参数
W = 5343.058  # 飞机重量（假设值，单位：kg）
S = 31.648  # 机翼总面积（假设值，单位：m²）
R_c = 1.3  # 水平速度系数
R_d = 1.0  # 垂直速度系数
L_D_climb = 4.559  # 爬升阶段的升阻比
L_D_descent = 4.559  # 下降阶段的升阻比
eta_mech = 0.9114  # 机械效率
eta_prop = 0.8576  # 推进效率
ROC_climb = 3.792  # 爬升率 (m/s)
ROC_descent = 3.048  # 下降率 (m/s)

# 高度范围从 0 到 457.2 米
height_range = list(range(0, 458))
num_segments = 5
segment_size = len(height_range) // num_segments

segment_heights = []
segment_densities = []
segment_stall_speeds = []
segment_climb_powers = []
segment_descent_powers = []

for i in range(num_segments):
    start = i * segment_size
    end = (i + 1) * segment_size if i < num_segments - 1 else len(height_range)
    segment = height_range[start:end]

    segment_temp_densities = []
    segment_temp_stall_speeds = []
    for height in segment:
        density = calculate_density(height)
        stall_speed = calculate_stall_speed(W, density, S)
        segment_temp_densities.append(density)
        segment_temp_stall_speeds.append(stall_speed)

    avg_height = sum(segment) / len(segment)
    avg_density = sum(segment_temp_densities) / len(segment_temp_densities)
    avg_stall_speed = sum(segment_temp_stall_speeds) / len(segment_temp_stall_speeds)

    # 计算水平和垂直速度
    V_x = R_c * avg_stall_speed
    V_y_climb = R_d * ROC_climb
    V_y_descent = -R_d * ROC_descent  # 下降时垂直速度为负

    # 计算爬升功率
    climb_power = calculate_climb_power(W, V_x, V_y_climb, L_D_climb, eta_mech, eta_prop)

    # 计算下降功率
    descent_power = calculate_descent_power(W, V_x, V_y_descent, L_D_descent, eta_mech, eta_prop)

    segment_heights.append(avg_height)
    segment_densities.append(avg_density)
    segment_stall_speeds.append(avg_stall_speed)
    segment_climb_powers.append(climb_power)
    segment_descent_powers.append(descent_power)

    print(f"第 {i + 1} 段:")
    print(f"  空气密度: {avg_density:.4f} kg/m³")
    print(f"  失速速度: {avg_stall_speed:.4f} m/s")
    print(f"  爬升功率: {climb_power:.4f} W")
    print(f"  下降功率: {descent_power:.4f} W")

# 计算5段的平均爬升功率
average_climb_power = sum(segment_climb_powers) / len(segment_climb_powers)
# 计算5段的平均下降功率
average_descent_power = sum(segment_descent_powers) / len(segment_descent_powers)

print(f"总平均爬升功率: {average_climb_power:.4f} W")
print(f"总平均下降功率: {average_descent_power:.4f} W")

# 扩展数据到完整高度范围
full_heights = height_range
full_densities = []
full_stall_speeds = []
full_climb_powers = []
full_descent_powers = []
for height in full_heights:
    for i in range(num_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < num_segments - 1 else len(height_range)
        if start <= height < end:
            full_densities.append(segment_densities[i])
            full_stall_speeds.append(segment_stall_speeds[i])
            full_climb_powers.append(segment_climb_powers[i])
            full_descent_powers.append(segment_descent_powers[i])
            break

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建一个包含四个子图的图形
plt.figure(figsize=(14, 11))

# 绘制平均空气密度随高度变化的阶梯图
plt.subplot(2, 2, 1)
plt.step(full_heights, full_densities, where='mid', color='b')
plt.xlabel('高度 (m)')
plt.xticks(rotation=45)
plt.ylabel('空气密度 (kg/m³)')
plt.title('空气密度随高度变化')
plt.xlim(0, max(height_range))
plt.xticks(range(0, max(height_range) + 1, 50))

# 绘制平均失速速度随高度变化的阶梯图
plt.subplot(2, 2, 2)
plt.step(full_heights, full_stall_speeds, where='mid', color='r')
plt.xlabel('高度 (m)')
plt.xticks(rotation=45)
plt.ylabel('失速速度 (m/s)')
plt.title('失速速度随高度变化')
plt.xlim(0, max(height_range))
plt.xticks(range(0, max(height_range) + 1, 50))

# 绘制爬升功率随高度变化的阶梯图
plt.subplot(2, 2, 3)
plt.step(full_heights, full_climb_powers, where='mid', color='g')
plt.xlabel('高度 (米)')
plt.xticks(rotation=45)
plt.ylabel('爬升功率 (m)')
plt.title('爬升功率随高度变化')
plt.xlim(0, max(height_range))
plt.xticks(range(0, max(height_range) + 1, 50))

# 绘制下降功率随高度变化的阶梯图
plt.subplot(2, 2, 4)
plt.step(full_heights, full_descent_powers, where='mid', color='m')
plt.xlabel('高度 (m)')
plt.xticks(rotation=45)
plt.ylabel('下降功率 (W)')
plt.title('下降功率随高度变化')
plt.xlim(0, max(height_range))
plt.xticks(range(0, max(height_range) + 1, 50))

plt.tight_layout()
plt.show()