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


# solve coolant temperture distribution using a 1D finite difference approach along the flow direction

def cal_energy_balance(T_dist, args):  
    
    num_seg = args.get("num_seg")
    dt = args.get("dt")
        
    T_cool_pre = args.get("T_cool_pre")
    T_bat_pre = args.get("T_bat_pre")
    u_cool_in = args.get("u_cool_in")
    p_cool = args.get("p_cool")
    fluid_cool = args.get("fluid_cool")
    A_HT_seg = args.get("A_HT_seg") # heat transfer area for each segment
    A_cool = args.get("A_cool")
    m_bat = args.get("m_bat")
    cp_bat = args.get("cp_bat")
    D_bat = args.get("D_bat")
    T_cool_in = args.get("T_cool_in")
    is_cool = args.get("is_cool")
    htc_cool = args.get("htc_cool") # use constant heat transfer coefficient for simplicity; adjust as needed
    cp_cool = args.get("cp_cool") # use constant specific heat capacity for simplicity; adjust as needed
    rho_cool = args.get("rho_cool") # use constant density for simplicity; adjust as needed
    Q_gen = args.get("Q_gen")
    debug = args.get("debug", False)
    
    # The T_dist variable contains the temperature for both coolant and battery for each segment along the flow path
    
    T_cool = T_dist[0:num_seg]  # the first half: coolant temperature distribution along the flow path, this is the variable we want to solve for in the optimization solver
    T_bat = T_dist[num_seg:] # the second half   
         
    
    Q_cool_HT = np.zeros(num_seg)  # the convective heat transfer from the battery to the coolant for each segment
    Q_cool_change = np.zeros(num_seg)  # the change of heat capacity of the coolant 
    Q_bat = np.zeros(num_seg)  # battery internal energy change 

    for i in range(num_seg): 

        T_cool_seg_in = T_cool_in if i == 0 else T_cool[i-1]  # inlet coolant temperature for this segment
        T_cool_seg_out = T_cool[i]  # outlet coolant temperature for this segment
        T_cool_bar = (T_cool_seg_in + T_cool_seg_out) / 2  # average coolant temperature for this segment        
        # T_cool_bar = T_cool_seg_in 
        # convective heat transfer on the cooling side
        if not is_cool:
            Q_cool_HT[i] = 0
          
        else:    
            Q_cool_HT[i] = htc_cool[i] * A_HT_seg * (T_bat[i] - T_cool_bar)  # use the average temperature between the current and inlet coolant temperature for heat transfer calculation

        dT_bat = T_bat[i] - T_bat_pre[i]  # change in battery temperature 
        
        if debug:
            print(f"Segment {i}: Q_gen = {Q_gen}, Q_cool_HT = {Q_cool_HT}, dT_bat = {dT_bat}; T_bat_cur = {T_bat}, T_cool_bar = {T_cool_bar}")  # debug print for power generation, cooling power, and battery temperature change for this segment

        Q_bat[i] = dT_bat * m_bat * cp_bat / dt  # heat input rate related to temperature change        
          
        Q_cool_change[i] = u_cool_in * A_cool * rho_cool * cp_cool * (T_cool[i] - T_cool_seg_in)  # calculate the power transfer to the coolant based on the current guess of the coolant temperature distribution for this segment, using a constant specific heat capacity and temperature change for the coolant for simplicity; adjust as needed
        
        # # detemrine the power transfer based on the change in enthalpy of the coolant across this segment; This calculation is more accurate but more computationally expensive than using a constant specific heat capacity and temperature change for the coolant; adjust as needed
        # h_cool_in = CP.PropsSI('H', 'T', T_cool_seg_in+273.15, 'P', p_cool, fluid_cool)  # enthalpy of the coolant at the inlet temperature for this segment
        # h_cool_out = CP.PropsSI('H', 'T', T_cool[i]+273.15, 'P', p_cool, fluid_cool)  # enthalpy of the coolant at the outlet temperature for this segment
        # Q_cool_change = u_cool_in * A_cool * rho_cool * (h_cool_out - h_cool_in)  # calculate the power transfer to the coolant based on the change in enthalpy of the coolant across this segment
        
                
        # calculate the residual between the input power and the power received by the coolant    
  
    return Q_cool_HT, Q_cool_change, Q_bat

def cal_power_residual(T_dist, args):
    
    Q_cool_HT, Q_cool_change, Q_bat = cal_energy_balance(T_dist, args)
    num_seg = args.get("num_seg")
    Q_gen = args.get("Q_gen") * np.ones(num_seg) # assuming constant power generation for simplicity; adjust as needed
    
    res_bat = Q_gen - Q_bat - Q_cool_HT

    # Coolant energy balance residual:
    # Q_cool_HT = Q_cool_change
    res_cool = Q_cool_HT - Q_cool_change
    
    # residual = np.zeros(num_seg)
    # residual = res_bat**2 + res_cool**2

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