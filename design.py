#%% imports & definitions
import numpy as np
import matplotlib.pyplot as plt

from design_func import *

#from design_func import F_phi, Wc_Re, epsilon_alpha, a_a_prime_W, I_prime_J_prime

# ===========================================
# ============ design parameters ============
# ===========================================

B = 3           # number of blades
R = 1.7/2       # propeller tip radius [m]
xi0 = 0.3/(2*R) # nondimensional hub radius [-]

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

# ===========================================
# ============ design procedure =============
# ===========================================

def design_loop(xi_, zeta_init, lambd, Tc, B, V, R, nu, max_iter=100):
    """
    Iterative design loop to find the optimal displacement velocity ratio zeta.

    Inputs:
        xi_       : nondimensional blade section radii [-]
        zeta_init : initial guess for the displacement velocity ratio [-]
        lambd     : speed ratio [-]
        Tc        : thrust coefficient [-]
        B         : number of blades [-]
        V         : freestream velocity [m/s]
        R         : tip radius [m]
        nu        : kinematic viscosity [m^2/s]
        max_iter  : maximum number of iterations [-]

    Outputs:
        zeta, cl_, epsilon_, alpha_, Wc_, a_, a_prime_, W_, c_, beta_,
        I1, I2, J1, J2, Pc, eta, sigma
    """

    N = len(xi_)
    zeta = zeta_init
    cl_guess = 1.0    # initial guess for the lift coefficient of each section

    phi_    = np.zeros(N)               # flow angle at each blade section [rad]
    xi_     = np.linspace(xi0, 1, N)    # nondimensional blade section radii [-]
    cl_     = np.zeros(N)               # lift coefficient at each blade section [-]
    epsilon_= np.zeros(N)               # drag-to-lift ratio at each blade section [-]
    alpha_  = np.zeros(N)               # angle of attack at each blade section [rad]
    Wc_     = np.zeros(N)               # relative velocity at each blade section [m/s]
    
    a_      = np.zeros(N)
    a_prime_= np.zeros(N)
    W_      = np.zeros(N)
    c_      = np.zeros(N)
    beta_   = np.zeros(N)

    I1_prime = np.zeros(N)
    I2_prime = np.zeros(N)
    J1_prime = np.zeros(N)
    J2_prime = np.zeros(N)

    for i in range(max_iter):
        F_, phi_ = F_phi(xi_, zeta, lambd, B)
        x = Omega * xi_ * R / V
        G_ = F_ * x * np.cos(phi_) * np.sin(phi_)

        for i in range(N):
            cl = cl_guess
            for j in range(20):
                Wc, Re = Wc_Re(lambd, cl, G_[i], V, R, zeta, B, nu)
                epsilon, alpha, cl_opt = epsilon_alpha(Re)

                if abs((cl_opt - cl)/cl) < 1e-3:
                    break
                
                cl = cl_opt
            else:
                print(f"Warning: convergence not reached for section {i} after 20 iterations.")

            cl_[i]      = cl
            epsilon_[i] = epsilon
            alpha_[i]   = alpha
            Wc_[i]      = Wc

        for i in range(N):
            a_[i], a_prime_[i], W_[i] = a_a_prime_W(zeta, phi_[i], epsilon_[i], x[i], V)
            c_[i] = Wc_[i] / W_[i]
            beta_[i] =  alpha_[i] + phi_[i]

            I1_prime[i], I2_prime[i], J1_prime[i], J2_prime[i] = I_prime_J_prime(xi_[i], G_[i], phi_[i], epsilon_[i], lambd)

        I1 = np.trapezoid(I1_prime, xi_)
        I2 = np.trapezoid(I2_prime, xi_)

        zeta_new = (I1/(2*I2)) - ((I1/(2*I2))**2 - Tc/I2)**0.5

        if abs(zeta_new - zeta) < 1e-3:
            J1 = np.trapezoid(J1_prime, xi_)
            J2 = np.trapezoid(J2_prime, xi_)
            Pc = J1*zeta_new + J2*zeta_new**2
            eta = Tc / Pc
            sigma = (B*c_)/(2*np.pi*xi_*R)
            
            return zeta_new, cl_, epsilon_, alpha_, Wc_, a_, a_prime_, W_, c_, beta_, I1, I2, J1, J2, Pc, eta, sigma
        
        zeta = zeta_new
        
    raise RuntimeError(f"Design loop did not converge after {max_iter} iterations.")


def bemt(xi_, beta_, B, V, Omega, R, nu, c_, v_a3_init, v_u2p_init , max_iter=100):
    """
    Iterative implementation of the BEMT algorithm to find the thrust and torque distribution.

    Inputs:
        xi_        : nondimensional blade section radii [-]
        beta_      : twist angle of the wing section [rad]
        B          : number of propeller blades [-]
        V          : freestream velocity [m/s]
        R          : tip radius [m]
        nu         : kinematic viscosity [m^2/s]
        v_a3_init  : initial guess for the axial downstream velocity [m/s]
        v_u2p_init : initial guess for the radial velocity immediatly downstream of the blades [m/s]
        max_iter   : maximum number of iterations [-]

    Outputs:
        dT         : Thrust distribution on the wing
        dC         : Torque distribution on the wing
        dP         : Power required to move the motor
    """
    v_a3 = v_a3_init
    v_u2p = v_u2p_init
    dr = R*(1-xi_[0])/N
    
    dT = np.zeros_like(xi_)
    dC = np.zeros_like(xi_)
    
    diff_iter_1 = 1
    diff_iter_2 = 1
    
    iter_nb = 0
    
    while diff_iter_1 > 0.001 or diff_iter_2 > 0.001:
        
        iter_nb += 1
        if iter_nb == max_iter:
            raise RuntimeError(f"Design loop did not converge after {max_iter} iterations.")
        
        v_a2 = (V + v_a3)/2
        w_a2 = v_a2
        v_u2 = v_u2p/2
        w_u2 = v_u2 - Omega * xi_ * R
        
        dm_dot = 2 * np.pi * xi_ * R * dr * rho * v_a2
        
        w_2 = (w_a2**2 + w_u2**2)**(1/2)
        phi_2 = np.arctan2(w_u2,w_a2)
        
        aoa = beta_ - (np.pi/2 + phi_2)
        Re = rho * w_2 * c_ / nu
        
        dL, dD = partial_lift_drag(aoa,Re,c_,rho,w_2,dr)
        
        cos_phi = w_a2/w_2
        sin_phi = w_u2/w_2
        
        dT = -B * (dL * sin_phi + dD * cos_phi)
        dC = xi_ * R * B * (dL * cos_phi - dD * sin_phi)
        
        v_a3_old = v_a3
        v_u2p_old = v_u2p
        
        v_a3 = V + dT/dm_dot
        v_u2p = dC/(dm_dot*xi_*R)
        
        diff_iter_1 = max(v_a3 - v_a3_old)
        diff_iter_2 = max(v_u2p - v_u2p_old)
        
    dP = dC*Omega
        
    return dT, dC, dP

def coefs_wrt_adv_ratio(xi_,beta_,B,Omega,R,nu,c_,J):
    """
    Function to find the thrust and power coefficient
    as well as the propulsive efficiency wrt. the advance ratio

    Inputs:
        xi_       : nondimensional blade section radii [-]
        beta_     : twist angle of the blade [rad]
        B         : number of blades [-]
        R         : tip radius [m]
        nu        : kinematic viscosity [m^2/s]
        c_        : chord length [m]
        J         : given advance ratio [-]

    Outputs:
        zeta, cl_, epsilon_, alpha_, Wc_, a_, a_prime_, W_, c_, beta_,
        I1, I2, J1, J2, Pc, eta, sigma
    """
    
    n = Omega/(2*np.pi)
    D = 2*R
    
    CT_ = np.zeros(len(J))
    CP_ = np.zeros(len(J))
    eta_ = np.zeros(len(J))
    
    for i in range(len(J)):
        
        V_inf = J[i]*n*D
        dT, dC, dP    = bemt(xi_,beta_,B,V_inf,Omega,R,nu,c_,V_inf,0)
        
        
        T = np.trapezoid(dT)
        P = np.trapezoid(dP)
        
        CT = 4*T/(D**4 * rho * n**2)
        CP = 4*P/(rho * n**3 * D**5)
        
        CT_[i] = CT
        CP_[i] = CP
        eta_[i] = T*V_inf/P
        
    idx = np.argmin(eta_)
    
    eta_ = eta_[0:idx]
    cond = eta_ > 0
    
    #Only take the "useful" values
    eta_ = eta_[cond]
    J_ = J[0:idx]
    J_ = J_[cond]
    CT_ = CT_[0:idx]
    CT_ = CT_[cond]
    CP_ = CP_[0:idx]
    CP_ = CP_[cond]
    
    return J_, CT_, CP_, eta_

def plot_results(xi_, y, ylabel):
    
    plt.figure(figsize=(10, 6))
    plt.plot(xi_, y)
    plt.xlabel(r'Nondimensional blade section radius $\xi$ [-]', fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.grid()
    
def compare_results(xi_,y_,ylabel , labels,xlabel=r'Nondimensional blade section radius $\xi$ [-]'):
    
    plt.figure(figsize=(10, 6))
    
    if len(y_) != len(labels):
        raise RuntimeError("The number of labels is not the same as the number of different results to compare.")
    
    for i in range(len(labels)):
        plt.plot(xi_,y_[i],label=labels[i])
        
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.legend()
    plt.grid()
    
def plot_multiple(x_, y_, xlabel, ylabel,labels):
    
    plt.figure(figsize=(10, 6))
    
    if len(y_) != len(labels) or len(x_) != len(y_):
        raise RuntimeError("The number of labels is not the same as the number of different results to compare.")
    
    for i in range(len(labels)):
        plt.plot(x_[i],y_[i],label=labels[i])
        
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.legend()
    plt.grid()
#%% Part 1 (Wing parameters)
if __name__ == "__main__":
    
# ===========================================
# ================ Part 1 ===================
# ===========================================
    
    xi_ = np.linspace(xi0, 1, N)
    zeta_init = 0.0
    zeta, cl_, epsilon_, alpha_, Wc_, a_, a_prime_, W_, c_, beta_, I1, I2, J1, J2, Pc, eta, sigma = design_loop(xi_, zeta_init, lambd, Tc, B, V, R, nu)

    print(f"Displacement velocity ratio zeta : {zeta:.4f}")

    plot_results(xi_, c_*100, 'Chord [cm]')
    plot_results(xi_, beta_*180/np.pi, 'Pitch angle [deg]')
    plot_results(xi_,alpha_*180/np.pi,'Angle of attack [deg]')

    plt.show()
#%% Part 2 (BEMT)
# ===========================================
# ================ Part 2 ===================
# ===========================================

    dT_0, dC_0, dP_0 = bemt(xi_,beta_,B,V,Omega,R,nu,c_,V,0)

    T_0 = np.trapezoid(dT_0) #No other arguments given as seen in the syllabus
    P_0 = np.trapezoid(dP_0)
    
    eta_P_0 = T_0*V/P_0
    
    eta_P_0_ = (dT_0[abs(dP_0) >= 0.0001]*V/dP_0[abs(dP_0) >= 0.0001])
    
    print(f"The thrust produced is {T_0:.4f} N")
    print(f"The power consumed is {P_0:.4f} W")
    print(f"The propulsive efficiency is {eta_P_0:.4f}")
    
    plot_results(xi_,dT_0,"Thrust distribution [N]")
    plot_results(xi_,dP_0,"Power distribution [W]")
    plot_results(xi_[abs(dP_0) >= 0.0001],eta_P_0_,"Propulsive efficiency [-]")
    
    plt.show()
    
    # Pitch angle +10°
    
    dT_10, dC_10, dP_10 = bemt(xi_,beta_ + 10 / 180 * np.pi ,B,V,Omega,R,nu,c_,V,0)
    
    T_10 = np.trapezoid(dT_10)
    P_10 = np.trapezoid(dP_10)
    eta_P_10 = T_10*V/P_10
    
    eta_P_10_ = (dT_10[abs(dP_10) >= 0.0001]*V/dP_10[abs(dP_10) >= 0.0001])
    
    print(f"The thrust produced with a collective pitch of +10° is {T_10:.4f} N")
    print(f"The power consumed with a collective pitch of +10° is {P_10:.4f} W")
    print(f"The propulsive efficiency with a collective pitch of +10° is {eta_P_10:.4f}")
    
    plot_results(xi_,dT_10,"Thrust distribution [N]")
    plot_results(xi_,dP_10,"Power distribution [W]")
    plot_results(xi_[abs(dP_10) >= 0.0001],eta_P_10_,"Propulsive efficiency [-]")
    
    plt.show()

    # Comparison of results :
    
    compare_results(xi_,[dT_0,dT_10],"Thrust [N]",["Base","Collective pitch of 10°"])
    compare_results(xi_,[dP_0,dP_10],"Power [W]",["Base","Collective pitch of 10°"])
    compare_results(xi_[0:N-1],[eta_P_0_,eta_P_10_],"Efficiency [-]",["Base","Collective pitch of 10°"])
    
    plt.show()
#%% Part 3 (Collective pitch effect)
# ===========================================
# ================ Part 3 ===================
# ===========================================
    
    J = np.linspace(5e-3,5,1000)
    
    J_0, CT_0_, CP_0_, eta_P_0_      = coefs_wrt_adv_ratio(xi_,beta_                     ,B,Omega,R,nu,c_,J)
    J_10, CT_10_, CP_10_, eta_P_10_  = coefs_wrt_adv_ratio(xi_,beta_ + 10 * np.pi / 180  ,B,Omega,R,nu,c_,J)
    J_20, CT_20_, CP_20_, eta_P_20_  = coefs_wrt_adv_ratio(xi_,beta_ + 20 * np.pi / 180  ,B,Omega,R,nu,c_,J)
    J_30, CT_30_, CP_30_, eta_P_30_  = coefs_wrt_adv_ratio(xi_,beta_ + 30 * np.pi / 180  ,B,Omega,R,nu,c_,J)
    
    xaxis = [J_0,J_10,J_20,J_30]

    yaxis_T = [CT_0_,CT_10_,CT_20_,CT_30_]
    yaxis_P = [CP_0_,CP_10_,CP_20_,CP_30_]
    yaxis_eta = [eta_P_0_,eta_P_10_,eta_P_20_,eta_P_30_]
    labels = ["Pitch = 0°","Pitch = 10°","Pitch = 20°","Pitch = 30°"]
    
    plot_multiple(xaxis,yaxis_T,"Advance ratio [-]","Thrust coefficient [-]",labels)
    plot_multiple(xaxis,yaxis_P,"Advance ratio [-]","Power coefficient [-]",labels)
    plot_multiple(xaxis,yaxis_eta,"Advance ratio [-]","Propulsive efficiency [-]",labels)
    
    plt.show()
    
    
#%% Part 4 (Cruise condition)

    V = 90 #[m/s]
