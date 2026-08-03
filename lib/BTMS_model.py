# A library for BTMS modelling. This includes the 0D and 1D thermal solvers for battery temperature prediction under different cooling conditions.
# The BTMS could be air-cooled, liquid-cooled, or phase change material (PCM) cooled. 

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import copy
import CoolProp.CoolProp as CP

# Žukauskas Nusellet correlation for natural convection from a horizontal cylinder, which uses air as working fluid. 
# eps is the correction factor for the Nusselt number, which can be selected based on the battery arrangement (in terms of number of columns) and Reynolds number using the select_correction_factor function defined below.
def zukauskas_nusellet(Re, Pr, *args):
    '''
    Calculate the Nusselt number for natural convection from a horizontal cylinder using the Žukauskas Nusellet correlation.
    Found eq. 19 in [[parkpark2025RefinedAircooled]].
    Parameters:
    Re: Reynolds number
    Pr: Prandtl number
    args: additional arguments for the correlation (e.g., correction factor)
    Returns:
    Nu: Nusselt number

    '''
    eps = args[0]  # correction factor for the Nusselt number
    Pr_s = args[1]  # Prandtl number at the surface temperature

    if Re > 0 and Re < 500:
       Nu = eps * 1.04 * (Re ** 0.4) * (Pr ** 0.36) * (Pr / Pr_s) ** 0.25
    elif Re >= 500 and Re < 1000:
        Nu = eps * 0.71 * (Re ** 0.5) * (Pr ** 0.36) * (Pr / Pr_s) ** 0.25
    elif Re >= 1000 and Re < 2e5:
        Nu = eps * 0.193 * (Re ** 0.62) * (Pr ** 0.36) * (Pr / Pr_s) ** 0.25
    else:
        Nu = eps * 0.027 * (Re ** 0.8) * (Pr ** 0.36) * (Pr / Pr_s) ** 0.25
        
    return Nu

# select the correction factor based on battery arrangement (in terms of number of columns) and Reynolds number
def select_correction_factor(no_col, Re):
    """
    Return correction factor eps for tube bank/cell-column arrangements.

    Data source (from your referenced table):
    - 1e2 < Re < 1e3
    - Re > 1e3
    with columns 1..9.

    Parameters
    ----------
    no_col : int
        Number of columns in the battery arrangement (1 to 9).
    Re : float
        Reynolds number based on battery diameter.

    Returns
    -------
    float
        Correction factor eps.
    """
    if no_col < 1:
        raise ValueError("no_col must be >= 1")

    # Clamp to available tabulated range [1, 9].
    col = int(round(no_col))
    col = max(1, min(9, col))
    idx = col - 1

    eps_1e2_to_1e3 = [0.832, 0.874, 0.914, 0.939, 0.955, 0.963, 0.970, 0.976, 0.980]
    eps_gt_1e3 = [0.619, 0.758, 0.840, 0.897, 0.923, 0.942, 0.954, 0.965, 0.971]

    # Piecewise selection with linear blending across the transition region.
    if Re <= 1e2:
        eps = eps_1e2_to_1e3[idx]
    elif Re < 1e3:
        eps = eps_1e2_to_1e3[idx]
    elif Re > 1e3:
        eps = eps_gt_1e3[idx]
    else:  # Re == 1e3
        eps = 0.5 * (eps_1e2_to_1e3[idx] + eps_gt_1e3[idx])

    return eps

def liquid_nusselt_number(Re, Pr, heating=True):
    """
    Nusselt number for internal liquid flow in a circular channel.

    Model used here:
    - Laminar fully-developed constant-wall-temperature pipe flow: Nu = 3.66 for Re < 2300
    - Turbulent Dittus-Boelter: Nu = 0.023 Re^0.8 Pr^n for Re >= 4000
      n = 0.4 when the coolant is heated by the wall/battery; n = 0.3 when the coolant is cooled.
    - Transitional region 2300 <= Re < 4000: linear interpolation between the two values.
    """
    Re = float(Re)
    Pr = float(Pr)

    if Re <= 0 or Pr <= 0:
        raise ValueError("Re and Pr must be positive for the liquid Nusselt calculation.")

    Nu_laminar = 4.36
    n = 0.4 if heating else 0.3
    
    # # use DB correlation 
    # Nu_DB = 0.023 * (Re ** 0.8) * (Pr ** n)
    
    # return Nu_DB

    def Nu_DB(Re_value):
        return 0.023 * (Re_value ** 0.8) * (Pr ** n)

    if Re < 2300:
        return Nu_laminar
    if Re >= 4000:
        return Nu_DB(Re)

    # Linear interpolation in the transition region.
    Nu_2300 = Nu_laminar
    Nu_4000 = Nu_DB(4000.0)
    weight = (Re - 2300.0) / (4000.0 - 2300.0)
    return Nu_2300 + weight * (Nu_4000 - Nu_2300)


def liquid_nusselt_number_rect_channel(W_channel, H_channel):
    """
    Nusselt number for fully developed laminar flow
    in a rectangular liquid-cooling channel with
    three heated walls and one adiabatic wall.

    The bottom wall and two side walls participate
    in heat transfer, while the top wall is adiabatic.

    The aspect ratio is defined as:

        beta = min(W_channel, H_channel)
               / max(W_channel, H_channel)

    Parameters
    ----------
    W_channel : float
        Rectangular-channel width, m.
    H_channel : float
        Rectangular-channel height, m.

    Returns
    -------
    float
        Nusselt number for three-sided heat transfer.
    """

    W_channel = float(W_channel)
    H_channel = float(H_channel)

    if W_channel <= 0 or H_channel <= 0:
        raise ValueError(
            "W_channel and H_channel must be positive."
        )

    beta = (
        min(W_channel, H_channel)
        / max(W_channel, H_channel)
    )

    Nu = 8.235 * (
        1.0
        - 1.883 * beta
        + 3.767 * beta**2
        - 5.814 * beta**3
        + 5.361 * beta**4
        - 2.0 * beta**5
    )

    return Nu

# solve coolant temperture distribution using a 1D finite difference approach along the flow direction

def cal_energy_balance(T_dist, args):  
    
    num_seg_bat = args.get("num_seg_bat")
    num_seg_cool = len(T_dist) - num_seg_bat
    dt = args.get("dt")
        
    T_cool_pre = args.get("T_cool_pre")
    T_bat_pre = args.get("T_bat_pre")
    u_cool_in = args.get("u_cool_in")
    p_cool = args.get("p_cool")
    fluid_cool = args.get("fluid_cool")
    A_HT_seg = args.get("A_HT_seg") # heat transfer area for each segment
    A_cool_cs = args.get("A_cool_cs")
    m_bat = args.get("m_bat")
    cp_bat = args.get("cp_bat")
    D_bat = args.get("D_bat")
    T_cool_in = args.get("T_cool_in")
    is_cool = args.get("is_cool")
    htc_cool = args.get("htc_cool") # use constant heat transfer coefficient for simplicity; adjust as needed
    cp_cool = args.get("cp_cool") # use constant specific heat capacity for simplicity; adjust as needed
    rho_cool = args.get("rho_cool") # use constant density for simplicity; adjust as needed
    Q_gen = args.get("Q_gen")
    cell_to_cool_map = args.get("cell_to_cool_map")
    num_cell_seg = args.get("num_cell_seg", 1.0)   
    debug = args.get("debug", False)
    upwind_scheme = args.get("upwind_scheme", False)
    
    # The T_dist variable contains the temperature for both coolant and battery for each segment along the flow path
    
    T_cool = T_dist[0:num_seg_cool]  # the first half: coolant temperature distribution along the flow path, this is the variable we want to solve for in the optimization solver
    T_bat = T_dist[num_seg_cool:] # the second half   
         
    
    Q_cool_HT = np.zeros(num_seg_bat)  # the convective heat transfer from the battery to the coolant for each segment
    Q_cool_change = np.zeros(num_seg_bat)  # the change of heat capacity of the coolant 
    Q_bat = np.zeros(num_seg_bat)  # battery internal energy change 

    for i in range(num_seg_bat): 

        Qdot_cool_HT_seg = 0 # heat transfer for this segment
        Qdot_cool_change_seg = 0 # change of heat capacity for this segment of the coolant
        if cell_to_cool_map is None:
            i_cool_seg = (i,)

        else:
            i_cool_seg = cell_to_cool_map[i]

        T_seg = []  # store the coolant temperatures for this battery segment
        A_HT_seg_all = [] 
        
        for seg in i_cool_seg:
            
            if not seg: pass # skip if no corresponding coolant segment for this battery segment
            
            T_cool_seg_in = T_cool_in if seg == 0 else T_cool[seg-1]  # inlet coolant temperature for this segment
            T_cool_seg_out = T_cool[seg]  # outlet coolant temperature for this segment
            
            if upwind_scheme:
                T_cool_bar = T_cool_seg_in 
                T_seg.append(T_cool_seg_in)
            else:
                T_cool_bar = (T_cool_seg_in + T_cool_seg_out) / 2
                T_seg.append(T_cool_seg_in)
                T_seg.append(T_cool_seg_out)
            
            A_HT_seg_all.append(A_HT_seg)     
                  
            Qdot_cool_change_seg += u_cool_in * A_cool_cs * rho_cool * cp_cool * (T_cool_seg_out - T_cool_seg_in) # heat added to or removed from the coolant for this segment
  
        # convective heat transfer on the cooling side
        if not is_cool:
            Qdot_cool_HT_seg = 0
        else:    
            if debug and T_bat[i] < T_cool_bar:
                print(f"Segment {i}: T_bat = {T_bat[i]}, T_cool_bar = {T_cool_bar}, cooling skipped because T_bat <= T_cool_bar")
            # Q_cool_HT_seg = 0
            Qdot_cool_HT_seg = htc_cool[i] * sum(A_HT_seg_all) * (T_bat[i] - np.mean(T_seg))  # use the average temperature between the current and inlet coolant temperature for heat transfer calculation
        
        
        if(debug):
            print(f"u_cool_in = {u_cool_in}, A_cool_cs = {A_cool_cs}, rho_cool = {rho_cool}, cp_cool = {cp_cool}, T_cool_seg_out = {T_cool_seg_out}, T_cool_seg_in = {T_cool_seg_in}")
            print(f"Qdot_cool_HT_seg = {Qdot_cool_HT_seg}, Qdot_cool_change_seg = {Qdot_cool_change_seg}")
            
        dT_bat = T_bat[i] - T_bat_pre[i]  # change in battery temperature 
        
        if debug:
            print(f"Segment {i}: Q_gen = {Q_gen}, Q_cool_HT = {Q_cool_HT}, dT_bat = {dT_bat}; T_bat_cur = {T_bat}, T_cool_bar = {T_cool_bar}")  # debug print for power generation, cooling power, and battery temperature change for this segment

        Q_cool_HT[i] = Qdot_cool_HT_seg * dt  # store the calculated convective heat transfer for this segment
        Q_cool_change[i] = Qdot_cool_change_seg * dt  # 
        Q_bat[i] = m_bat * num_cell_seg * cp_bat * dT_bat
        
    return Q_cool_HT, Q_cool_change, Q_bat

def cal_power_residual(T_dist, args):
    
    Q_cool_HT, Q_cool_change, Q_bat = cal_energy_balance(T_dist, args)

    num_seg = args.get("num_seg")
    num_cell_seg = args.get("num_cell_seg", 1.0)
    

    Q_gen = args.get("Q_gen") * num_cell_seg * np.ones(num_seg) # assuming constant power generation for simplicity; adjust as needed

    
    res_bat = Q_gen - Q_bat - Q_cool_HT

    # Coolant energy balance residual:
    # Q_cool_HT = Q_cool_change
    res_cool = Q_cool_HT - Q_cool_change
    
    # residual = np.zeros(num_seg)
    # residual = res_bat**2 + res_cool**2 
    # The solver is too stiff. It is better to replace it with the one below
    # return residual
    
    
    # Return raw residuals. least_squares will square them internally.
    return np.concatenate((res_bat, res_cool))

def solve_coolant_temperature_distribution(args, tol=1e-6, maxiter=1000, debug=False):
    
    T_bat_pre = args.get("T_bat_pre")
    num_seg = len(T_bat_pre)    
    T_cool_in = args.get("T_cool_in")
    is_cool = args.get("is_cool")
    Q_gen = args.get("Q_gen")
    dt = args.get("dt")
    m_bat = args.get("m_bat")
    cp_bat = args.get("cp_bat") 
    args["debug"] = debug

    # Non-cooling period:
    # Do not solve coolant temperature distribution with least_squares.
    # Coolant temperature remains equal to inlet air temperature.
    # Battery temperature is updated only by internal heat generation.
    if not is_cool:
        T_cool = np.ones(num_seg) * T_cool_in
        T_bat = T_bat_pre + Q_gen * dt / (m_bat * cp_bat)
        return np.concatenate((T_cool, T_bat))
    
    T_cool_out = args.get("T_cool_out") # expected outlet coolant temperature
    T_cool_pre = args.get("T_cool_pre")

    # Use previous time-step coolant temperature as the initial guess.
    # If it is unavailable or invalid, fall back to the original linear guess.
    if T_cool_pre is not None and np.all(np.isfinite(T_cool_pre)):
        T_cool = copy.deepcopy(T_cool_pre)
    else:
        T_cool = np.linspace(T_cool_in, T_cool_out, num_seg + 1)[1:]

    # Keep the initial guess within the lower bound.
    T_cool = np.maximum(T_cool, T_cool_in + 1e-8)

    # Use previous battery temperature as the initial guess.
    T_bat = copy.deepcopy(T_bat_pre)

    T_dist = np.concatenate((T_cool, T_bat))
    
    if debug:
        print(f"Initial guess for {len(T_cool)} coolant temperatures along the flow path: {T_cool}")

    num_solutions = len(T_dist)

    upper_bound = np.ones(num_solutions) * (T_cool_in + 300)
    lower_bound = np.ones(num_solutions) * T_cool_in
    
    res = opt.least_squares(        
        cal_power_residual, 
        T_dist, 
        args=([args]), 
        method ='trf', 
        verbose=2 if debug else 0,
        ftol=tol, 
        max_nfev=maxiter,
        bounds=(lower_bound, upper_bound)
    )
    
    # output the battery temperature and coolant temperature distribution for debugging

    if debug:
        print("Optimization success:", res.success)
        print("Final residual:", res.fun)        
        print("Optimized coolant temperatures:", res.x)
    
    return res.x




# =============================================================================
# Auxiliary power and BTMS mass estimation
# =============================================================================
# Note:
# For liquid cooling, the auxiliary power and BTMS mass are calculated
# for the complete liquid-cooling system.
#
# The hydraulic pressure drop is evaluated using the flow conditions
# in one representative parallel channel, while pump power is calculated
# using the total coolant flow rate through all parallel channels.
def cal_friction_factor(Re):
    Re = float(Re)

    if Re <= 0:
        raise ValueError("Re must be positive for friction-factor calculation.")

    if Re < 2300.0:
        return 64.0 / Re

    if Re >= 4000.0:
        return 0.3164 * Re ** (-0.25)

    f_laminar = 64.0 / 2300.0
    f_turbulent = 0.3164 * 4000.0 ** (-0.25)
    weight = (Re - 2300.0) / (4000.0 - 2300.0)

    return f_laminar + weight * (f_turbulent - f_laminar)

def cal_friction_factor_rect_channel(Re,W_channel,H_channel,):
    """
    Darcy friction factor for fully developed laminar flow
    in a rectangular channel.

    The aspect ratio is defined as:

        beta = min(W_channel, H_channel)
               / max(W_channel, H_channel)

    The Darcy friction factor is:

        f = 96 / Re * (
            1
            - 1.3553 * beta
            + 1.9467 * beta**2
            - 1.7012 * beta**3
            + 0.9564 * beta**4
            - 0.2537 * beta**5
        )
  
    """

    Re = float(Re)
    W_channel = float(W_channel)
    H_channel = float(H_channel)



    beta = (
        min(W_channel, H_channel)
        / max(W_channel, H_channel)
    )

    f = 96.0 / Re * (
        1.0
        - 1.3553 * beta
        + 1.9467 * beta**2
        - 1.7012 * beta**3
        + 0.9564 * beta**4
        - 0.2537 * beta**5
    )

    return f



def cal_btms_aux_power(args, friction_factor=None):
    """
    Calculate BTMS auxiliary power and energy consumption.

    Liquid cooling:
        The pressure drop is calculated using one representative
        parallel channel, while the pump power is calculated using
        the total coolant flow rate through all parallel channels.

    Air cooling:
        The pressure drop is calculated using one representative
        airflow path, while the fan power is calculated using the
        total airflow rate through all parallel airflow domains.
    """

    fluid_cool = str(args["fluid_cool"]).lower()
    operation_time = float(args.get("operation_time", 0.0))
    K_minor = float(args.get("K_minor", 0.0))

    # ==========================================================
    # Liquid cooling
    # ==========================================================
    if fluid_cool == "water":

        rho = float(args["rho_cool"])
        mu = float(args["mu_cool"])
        A_cs = float(args["A_cool_cs"])
        Dh = float(args["D_channel"])
        L = float(args["L_channel"])
        m_dot_total = float(args["m_dot_total"])
        num_channel = int(args.get("num_channel", args.get("num_channels")))
        efficiency = float(args.get("pump_efficiency", 0.35))

        m_dot_channel = m_dot_total / num_channel
        u = m_dot_channel / (rho * A_cs)

        V_dot_channel = m_dot_channel / rho
        V_dot_total = m_dot_total / rho

    # ==========================================================
    # Air cooling
    # ==========================================================
    elif fluid_cool == "air":

        rho = float(args["rho_cool"])
        mu = float(args["mu_cool"])
        u = float(args["u_cool_in"])
        A_cs = float(args["A_cool_cs"])
        Dh = float(args["D_channel"])
        L = float(args["L_channel"])
        num_channel = int(args.get("num_channel", args.get("num_channels", 1)))
        efficiency = float(args.get("fan_efficiency", 0.35))

        V_dot_channel = u * A_cs
        V_dot_total = num_channel * V_dot_channel

        m_dot_channel = rho * V_dot_channel
        m_dot_total = rho * V_dot_total

    # ==========================================================
    # Reynolds number
    # ==========================================================
    Re = rho * u * Dh / mu

    # ==========================================================
    # Friction factor
    # ==========================================================

    if friction_factor is None:
        raise ValueError(
             "friction_factor must be provided for auxiliary-power calculation."
        )

    f = float(friction_factor)

    if f <= 0:
        raise ValueError(
            "friction_factor must be positive."
        )


    # ==========================================================
    # Pressure drop
    # ==========================================================
    dynamic_pressure = 0.5 * rho * u**2

    delta_p_major = f * (L / Dh) * dynamic_pressure
    delta_p_minor = K_minor * dynamic_pressure
    delta_p = delta_p_major + delta_p_minor

    # ==========================================================
    # Auxiliary power and energy
    # ==========================================================
    P_aux_W = delta_p * V_dot_total / efficiency
    E_aux_J = P_aux_W * operation_time

    return {
        "Re": Re,
        "friction_factor": f,
        "u_cool_in_m_s": u,
        "delta_p_major_Pa": delta_p_major,
        "delta_p_minor_Pa": delta_p_minor,
        "delta_p_Pa": delta_p,
        "m_dot_total_kg_s": m_dot_total,
        "m_dot_channel_kg_s": m_dot_channel,
        "V_dot_total_m3_s": V_dot_total,
        "V_dot_channel_m3_s": V_dot_channel,
        "num_channel": num_channel,
        "P_aux_W": P_aux_W,
        "E_aux_J": E_aux_J,
    }

# Calculate the total mass of the liquid-cooling BTMS, including the cold plate, coolant, pump, and pipes.
# N_c is used to scale the coolant volume to the full module.
# Calculate the total mass of BTMS according to cooling type.
def cal_btms_mass(args):

    fluid_cool = str(args["fluid_cool"]).lower()
    return_components = bool(args.get("return_components", False))

    # ==========================================================
    # Air cooling BTMS
    # ==========================================================
    if fluid_cool == "air":

        m_fan = float(args.get("m_fan", 0.0))
        m_BTMS_kg = m_fan

        if return_components:
            return {
                "m_BTMS_kg": m_BTMS_kg,
                "m_fan_kg": m_fan,
            }

        return m_BTMS_kg

    # ==========================================================
    # Liquid cooling BTMS
    # ==========================================================
    elif fluid_cool == "water":

        rho_cool = float(args["rho_cool"])
        rho_plate = float(args["rho_plate"])

        L_channel = float(args["L_channel"])
        L_plate = float(args["L_plate"])

        A_cool_cs = float(args["A_cool_cs"])
        H_channel = float(args["H_channel"])
        H_bottom = float(args.get("H_bottom", 2.0e-3))

        num_channel = int(args.get("num_channel", args.get("num_channels")))

        m_pump = float(args.get("m_pump", 0.0))
        m_pipe = float(args.get("m_pipe", 0.0))

        # The cover-plate thickness is neglected.
        H_plate = H_channel + H_bottom

        # Total external volume of the cold plate.
        V_plate_original = L_plate * L_channel * H_plate

        # Total coolant volume inside all parallel channels.
        V_coolant_total = num_channel * A_cool_cs * L_channel

        # Remaining aluminium volume after subtracting the channels.
        V_aluminium = V_plate_original - V_coolant_total

        if V_aluminium < 0:
            raise ValueError("The coolant-channel volume exceeds the cold-plate volume.")

        m_plate = rho_plate * V_aluminium
        m_coolant = rho_cool * V_coolant_total
        m_BTMS_kg = m_plate + m_coolant + m_pump + m_pipe

        if return_components:
            return {
                "m_BTMS_kg": m_BTMS_kg,
                "m_plate_kg": m_plate,
                "m_coolant_kg": m_coolant,
                "m_pump_kg": m_pump,
                "m_pipe_kg": m_pipe,
                "L_channel_m": L_channel,
                "L_plate_m": L_plate,
                "H_channel_m": H_channel,
                "H_bottom_m": H_bottom,
                "H_plate_m": H_plate,
                "V_plate_original_m3": V_plate_original,
                "V_coolant_total_m3": V_coolant_total,
                "V_aluminium_m3": V_aluminium,
                "num_channel": num_channel,
            }

        return m_BTMS_kg

    else:

        raise ValueError(
            f"Unsupported cooling type: {fluid_cool}. "
            "Only 'air' and 'water' are supported."
        )