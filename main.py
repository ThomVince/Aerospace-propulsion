#%% imports & parameters
import numpy as np
import matplotlib.pyplot as plt

from functions import *

# ===========================================
# ============ design parameters ============
# ===========================================

B = 3           # number of blades
R = 1.7/2       # propeller tip radius [m]
D = 2*R         # propeller diameter [m]
D_hub = 0.3     # hub diameter [m]
xi0 = D_hub / D # nondimensional hub radius [-]

T = 500     # required thrust [N]
V = 45      # freestream takeoff velocity [m/s]
rho = 1.225 # air density [kg/m^3]
n = 2100/60 # rotational speed [rev/s]

nu = 1.5e-5 # kinematic viscosity [m^2/s]

N = 100    # number of blade sections

# ===========================================
# =========== derived parameters ============
# ===========================================

Omega = 2*np.pi*n               # propeller angular velocity [rad/s]
lambd = V/(Omega*R)             # speed ratio [-]
Tc = 2*T/(rho*V**2*np.pi*R**2)  # thrust coefficient [-]

#%% Part 1 (Design procedure)
if __name__ == "__main__":
    
# ===========================================
# ================ Part 1 ===================
# ===========================================
    
    xi_ = np.linspace(xi0, 1, N)
    zeta_init = 0.0
    zeta, cl_, epsilon_, alpha_, Wc_, a_, a_prime_, W_, c_, beta_, I1, I2, J1, J2, Pc, eta, sigma = design_loop(xi_, xi0, zeta_init, lambd, Tc, B, V, R, nu, Omega)

    plot_results(xi_, c_*100, 'Chord [cm]')
    plot_results(xi_, beta_*180/np.pi, 'Pitch angle [deg]')

    plt.show()

    print("=== STEP 1 ===")
    print(f"Displacement velocity ratio zeta : {zeta:.4f}")
    print(f"Propulsive efficiency eta : {eta*100:.2f}%")
    print(f"Chord distribution from hub to tip: {c_[0]*100:.2f} --> {c_[-1]*100:.2f} cm")
    print(f"Maximum chord length: {np.max(c_)*100:.2f} cm at xi = {xi_[np.argmax(c_)]:.2f}")
    print(f"Pitch distribution: {beta_[0]*180/np.pi:.2f} deg --> {beta_[-1]*180/np.pi:.2f} deg")

#%% Part 2 (BEMT)
# ===========================================
# ================ Part 2 ===================
# ===========================================

    dT_0, dC_0, dP_0 = bemt(xi_, beta_, B, V, Omega, R, nu, c_, V, 0, rho)

    T_0 = np.trapezoid(dT_0) # No other arguments given as seen in the syllabus
    P_0 = np.trapezoid(dP_0)
    
    eta_P_0 = T_0*V/P_0
    
    eta_P_0_ = (dT_0[abs(dP_0) >= 0.0001]*V/dP_0[abs(dP_0) >= 0.0001])
        
    plot_results(xi_,dT_0,"Thrust distribution [N]")
    plot_results(xi_,dP_0,"Power distribution [W]")
    plot_results(xi_[abs(dP_0) >= 0.0001],eta_P_0_,"Propulsive efficiency [-]")
    
    plt.show()
    
    # Pitch angle +10°
    dT_10, dC_10, dP_10 = bemt(xi_, beta_ + 10 / 180 * np.pi, B, V, Omega, R, nu, c_, V, 0, rho)
    
    T_10 = np.trapezoid(dT_10)
    P_10 = np.trapezoid(dP_10)
    eta_P_10 = T_10*V/P_10
    
    eta_P_10_ = (dT_10[abs(dP_10) >= 0.0001]*V/dP_10[abs(dP_10) >= 0.0001])
        
    plot_results(xi_,dT_10,"Thrust distribution [N]")
    plot_results(xi_,dP_10,"Power distribution [W]")
    plot_results(xi_[abs(dP_10) >= 0.0001],eta_P_10_,"Propulsive efficiency [-]")
    
    plt.show()

    # Comparison of results :
    compare_results(xi_,[dT_0,dT_10], "Thrust [N]",["Base","Collective pitch of 10°"])
    compare_results(xi_,[dP_0,dP_10], "Power [W]",["Base","Collective pitch of 10°"])
    compare_results(xi_[0:N-1],[eta_P_0_,eta_P_10_], "Efficiency [-]", ["Base","Collective pitch of 10°"])
    
    plt.show()

    print("\n=== STEP 2 ===")
    print(f"Collective pitch = 0 deg")
    print(f"Thrust     = {T_0:.2f} N")
    print(f"Power      = {P_0:.2f} W  ({P_0/1000:.2f} kW)")
    print(f"Efficiency = {eta_P_0:.4f}  ({eta_P_0*100:.2f}%)")

    print(f"\nCollective pitch = 10 deg")
    print(f"Thrust     = {T_10:.2f} N")
    print(f"Power      = {P_10:.2f} W  ({P_10/1000:.2f} kW)")
    print(f"Efficiency = {eta_P_10:.4f}  ({eta_P_10*100:.2f}%)")

#%% Part 3 (Collective pitch effect)
# ===========================================
# ================ Part 3 ===================
# ===========================================
    
    J = np.linspace(5e-2,5,1000)
    
    J_0, CT_0_, CP_0_, eta_P_0_      = coefs_wrt_adv_ratio(xi_,beta_                   , B, Omega, R, nu, c_, J, rho)
    J_10, CT_10_, CP_10_, eta_P_10_  = coefs_wrt_adv_ratio(xi_,beta_ + 10 * np.pi / 180, B, Omega, R, nu, c_, J, rho)
    J_20, CT_20_, CP_20_, eta_P_20_  = coefs_wrt_adv_ratio(xi_,beta_ + 20 * np.pi / 180, B, Omega, R, nu, c_, J, rho)
    J_30, CT_30_, CP_30_, eta_P_30_  = coefs_wrt_adv_ratio(xi_,beta_ + 30 * np.pi / 180, B, Omega, R, nu, c_, J, rho)
    
    xaxis = [J_0,J_10,J_20,J_30]

    yaxis_T = [CT_0_,CT_10_,CT_20_,CT_30_]
    yaxis_P = [CP_0_,CP_10_,CP_20_,CP_30_]
    yaxis_eta = [eta_P_0_,eta_P_10_,eta_P_20_,eta_P_30_]
    labels = ["Pitch = 0°","Pitch = 10°","Pitch = 20°","Pitch = 30°"]
    
    plot_multiple(xaxis,yaxis_T,"Advance ratio [-]","Thrust coefficient [-]",labels)
    plot_multiple(xaxis,yaxis_P,"Advance ratio [-]","Power coefficient [-]",labels)
    plot_multiple(xaxis,yaxis_eta,"Advance ratio [-]","Propulsive efficiency [-]",labels)
    
    plt.show()
    
    print("\n=== STEP 3 ===")
    print("=== MAXIMUM EFFICIENCY CONDITIONS ===\n")

    pitch_labels = [0, 10, 20, 30]
    J_list   = [J_0,    J_10,    J_20,    J_30]
    eta_list = [eta_P_0_, eta_P_10_, eta_P_20_, eta_P_30_]

    for pitch_deg, J_arr, eta_arr in zip(pitch_labels, J_list, eta_list):
        idx_max   = np.argmax(eta_arr)
        J_etamax  = J_arr[idx_max]
        eta_max   = eta_arr[idx_max]
        V_etamax  = J_etamax * n * D
        print(f"Collective pitch = {pitch_deg} deg")
        print(f"J at eta_max     = {J_etamax:.3f}")
        print(f"V at eta_max     = {V_etamax:.2f} m/s")
        print(f"eta_max          = {eta_max:.4f} ({eta_max*100:.2f}%)\n")
    
#%% Part 4 (Cruise condition)
# ===========================================
# ================ Part 4 ===================
# ===========================================

    V_cruise = 90  # cruise velocity [m/s]

    # Advance ratio in cruise
    J_cruise = V_cruise / (n * D)

    pitch_angles_deg = np.linspace(0, 45, 451)

    T_cruise = np.zeros_like(pitch_angles_deg)
    P_cruise = np.zeros_like(pitch_angles_deg)
    eta_cruise = np.zeros_like(pitch_angles_deg)

    for i, pitch_deg in enumerate(pitch_angles_deg):
        pitch_rad = pitch_deg * np.pi / 180

        beta_cruise = beta_ + pitch_rad

        dT_cruise, dC_cruise, dP_cruise = bemt(xi_, beta_cruise, B, V_cruise, Omega, R, nu, c_, V_cruise, 0, rho)

        T = np.trapezoid(dT_cruise)
        P = np.trapezoid(dP_cruise)

        T_cruise[i] = T
        P_cruise[i] = P

        if P > 0:
            eta_cruise[i] = T * V_cruise / P
        else:
            eta_cruise[i] = np.nan

    # To remove non-physical results
    valid = np.isfinite(eta_cruise) & (eta_cruise > 0) & (eta_cruise < 1.5)

    pitch_angles_valid = pitch_angles_deg[valid]
    T_valid = T_cruise[valid]
    P_valid = P_cruise[valid]
    eta_valid = eta_cruise[valid]

    # Best collective pitch
    idx_best = np.argmax(eta_valid)

    best_pitch = pitch_angles_valid[idx_best]
    best_T = T_valid[idx_best]
    best_P = P_valid[idx_best]
    best_eta = eta_valid[idx_best]

    # PLOTS
    plot_part4(pitch_angles_valid, eta_valid, "Collective pitch angle [deg]", r"Propulsive efficiency $\eta$ [-]", best_pitch)    # Efficiency versus collective pitch
    plot_part4(pitch_angles_valid, T_valid, "Collective pitch angle [deg]", "Thrust [N]", best_pitch)                             # Thrust versus collective pitch
    plot_part4(pitch_angles_valid, P_valid, "Collective pitch angle [deg]", "Power [W]", best_pitch)                              # Power versus collective pitch

    plt.show()

    # PRINTS
    print("\n=== STEP 4 ===")
    print(f"Cruise advance ratio J: {J_cruise:.4f}")

    print(f"Best collective pitch: {best_pitch:.1f} deg")
    print(f"Thrust at best pitch: {best_T:.4f} N")
    print(f"Power at best pitch: {best_P:.4f} W")
    print(f"Maximum cruise efficiency = {best_eta:.4f} ({best_eta*100:.2f}%)")

# %%
