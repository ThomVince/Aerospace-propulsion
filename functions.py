import numpy as np
import matplotlib.pyplot as plt

from clarkypolarsRe import clarkypolarsRe

# ===========================================
# =========== Part 1 functions ==============
# ===========================================

def F_phi(xi, zeta, lambd, B):
    """
    Prandtl's momentum loss factor F and flow angle phi for a given blade section

    Inputs:
        xi    : nondimensional blade section radii [-]
        zeta  : displacement velocity ratio [-]
        lambd : speed ratio [-]
        B     : number of blades [-]

    Outputs:
        F   : Prandtl loss factor at each section [-]
        phi : flow angle at each section [rad]
    """
    phi_t = np.arctan(lambd * (1 + zeta/2)) # tip flow angle [rad]
    f = (B/2)*(1-xi)/np.sin(phi_t)
    F = (2/np.pi)*np.arccos(np.exp(-f))
    phi =  np.arctan2(np.tan(phi_t),xi)
    return F, phi

def Wc_Re(lambd, cl_fixed, G, V, R, zeta, B, nu):
    """
    Relative velocity times chord (Wc) and chord Reynolds number for a blade section.

    Inputs:
        lambd   : speed ratio [-]
        cl_fixed: lift coefficient (current estimate) [-]
        G       : circulation function at the section [-]
        V       : freestream velocity [m/s]
        R       : tip radius [m]
        zeta    : displacement velocity ratio [-]
        B       : number of blades [-]
        nu      : kinematic viscosity [m^2/s]

    Outputs:
        Wc : local total velocity times the chord [m²/s]
        Re : chord Reynolds number [-]
    """
    Wc = 4*np.pi*lambd*G*V*R*zeta/(cl_fixed*B)
    Re = Wc/nu
    return Wc, Re

def epsilon_alpha(Re):
    """
    Minimum drag-to-lift ratio epsilon and corresponding angle of attack and cl at a given Reynolds number

    Input:
        Re : chord Reynolds number [-]

    Outputs:
        epsilon : minimum drag-to-lift ratio [-]
        alpha   : corresponding angle of attack [rad]
        cl_opt  : corresponding lift coefficient [-]
    """
    aoa_ = np.linspace(-np.pi/2 , np.pi/2, 180)
    cl_, cd_ = clarkypolarsRe(aoa_,Re)
    
    inv_epsilon_ = cl_ / cd_
    
    idx = np.argmax(inv_epsilon_)
    
    aoa_ = np.linspace(aoa_[idx]-np.pi/180,aoa_[idx]+np.pi/180,120)
    
    cl_, cd_ = clarkypolarsRe(aoa_,Re)
    
    inv_epsilon_ = cl_ / cd_
    
    idx = np.argmax(inv_epsilon_)
    
    epsilon = 1/inv_epsilon_[idx]
    alpha = aoa_[idx]
    cl_opt = cl_[idx]
    
    return epsilon, alpha, cl_opt

def a_a_prime_W(zeta, phi, epsilon, x, V):
    """
    Axial interference factor a, rotational interference factor a_prime, and local total velocity W

    Inputs:
        zeta    : displacement velocity ratio [-]
        phi     : flow angle [rad]
        epsilon : drag-to-lift ratio [-]
        x       : nondimesional distance (Omega*r/V) [-]
        V       : freestream velocity [m/s]

    Outputs:
        a       : axial interference factor [-]
        a_prime : rotational interference factor [-]
        W       : local total velocity [m/s]
    """
    a = (zeta/2) * np.cos(phi)**2 * (1 - epsilon * np.tan(phi))
    a_prime = (zeta/(2*x)) * np.cos(phi) * np.sin(phi) * (1 + epsilon / np.tan(phi))
    W = V * (1 + a) / np.sin(phi)
    return a, a_prime, W

def I_prime_J_prime(xi, G, phi, epsilon, lambd):
    """
    The four derivatives I1', I2', J1', J2' for thrust and power coefficient integration

    Inputs:
        xi      : nondimensional blade section radius [-]
        G       : circulation function at the section [-]
        phi     : flow angle [rad]
        epsilon : drag-to-lift ratio [-]
        lambd   : speed ratio [-]

    Outputs:
        I1_prime, I2_prime, J1_prime, J2_prime : the four derivatives
    """
    I1_prime = 4 * xi * G * (1 - epsilon*np.tan(phi))
    I2_prime = lambd * (I1_prime / (2*xi)) * (1 + epsilon/np.tan(phi)) * np.sin(phi) * np.cos(phi)
    J1_prime = 4*xi*G*(1 + epsilon/np.tan(phi))
    J2_prime = (J1_prime/2) * (1 - epsilon*np.tan(phi)) * np.cos(phi)**2
    
    return I1_prime, I2_prime, J1_prime, J2_prime

def design_loop(xi_, xi0, zeta_init, lambd, Tc, B, V, R, nu, Omega, max_iter=100):
    """
    Iterative design loop to find the optimal displacement velocity ratio zeta.

    Inputs:
        xi_       : nondimensional blade section radii [-]
        xi0       : nondimensional hub radius [-]
        zeta_init : initial guess for the displacement velocity ratio [-]
        lambd     : speed ratio [-]
        Tc        : thrust coefficient [-]
        B         : number of blades [-]
        V         : freestream velocity [m/s]
        R         : tip radius [m]
        nu        : kinematic viscosity [m^2/s]
        Omega     : propeller angular velocity [rad/s]
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

        I1 = float(np.trapezoid(I1_prime, xi_))
        I2 = float(np.trapezoid(I2_prime, xi_))

        zeta_new = (I1/(2*I2)) - ((I1/(2*I2))**2 - Tc/I2)**0.5

        if abs(zeta_new - zeta) < 1e-3:
            J1 = float(np.trapezoid(J1_prime, xi_))
            J2 = float(np.trapezoid(J2_prime, xi_))
            Pc = J1*zeta_new + J2*zeta_new**2
            eta = Tc / Pc
            sigma = (B*c_)/(2*np.pi*xi_*R)
            
            return zeta_new, cl_, epsilon_, alpha_, Wc_, a_, a_prime_, W_, c_, beta_, I1, I2, J1, J2, Pc, eta, sigma
        
        zeta = zeta_new
        
    raise RuntimeError(f"Design loop did not converge after {max_iter} iterations.")

# ===========================================
# =========== Part 2 functions ==============
# ===========================================

def partial_lift_drag(AoA, Re, chord, rho, w, dr):
    """
    The lift and drag values at a given blade section

    Inputs:
        AoA     : Angle of attack [rad]
        Re      : Reynolds number at the section [-]
        chord   : chord length of section [m]
        rho     : density of air [kg/m³]
        w       : local velocity [m/s]
        dr      : blade section length [m]

    Outputs:
        dL, dD
    """
    
    cl, cd = clarkypolarsRe(AoA, Re)
    dL = cl * chord * rho * w**2 * dr / 2
    dD = cd * chord * rho * w**2 * dr / 2
    return dL, dD

def bemt(xi_, beta_, B, V, Omega, R, nu, c_, v_a3_init, v_u2p_init, rho, max_iter=100):
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
        rho        : air density [kg/m^3]
        max_iter   : maximum number of iterations [-]

    Outputs:
        dT         : Thrust distribution on the wing
        dC         : Torque distribution on the wing
        dP         : Power required to move the motor
    """
    N = len(beta_)
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
        
        diff_iter_1 = max(abs(v_a3 - v_a3_old))
        diff_iter_2 = max(abs(v_u2p - v_u2p_old))
        
    dP = dC*Omega
    
    return dT, dC, dP

def coefs_wrt_adv_ratio(xi_,beta_,B,Omega,R,nu,c_,J, rho):
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
        rho        : air density [kg/m^3]

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
        dT, dC, dP    = bemt(xi_, beta_, B, V_inf, Omega, R, nu, c_, V_inf, 0, rho, max_iter=1000)
        
        
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
    
    #Only take the "useful" ( > 0) values
    eta_ = eta_[cond]
    J_ = J[0:idx]
    J_ = J_[cond]
    CT_ = CT_[0:idx]
    CT_ = CT_[cond]
    CP_ = CP_[0:idx]
    CP_ = CP_[cond]
    
    return J_, CT_, CP_, eta_

# ===========================================
# ============ Plot functions ===============
# ===========================================

def plot_results(xi_, y, ylabel, pdf=None):
    
    plt.figure(figsize=(10, 6))
    plt.plot(xi_, y, lw=2.2)
    plt.xlabel(r'Non-dimensional blade section radius $\xi$ [-]', fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid()
    plt.tight_layout()

    if pdf is not None:
        plt.savefig(pdf + ".pdf", format='pdf', dpi=300, bbox_inches='tight')
    
def compare_results(xi_,y_,ylabel , labels,xlabel=r'Non-dimensional blade section radius $\xi$ [-]', pdf=None):
    
    plt.figure(figsize=(10, 6))
    
    if len(y_) != len(labels):
        raise RuntimeError("The number of labels is not the same as the number of different results to compare.")
    
    for i in range(len(labels)):
        plt.plot(xi_,y_[i],label=labels[i], lw=2.2)
        
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(fontsize=20)
    plt.grid()
    plt.tight_layout()

    if pdf is not None:
        plt.savefig(pdf + ".pdf", format='pdf', dpi=300, bbox_inches='tight')
    
def plot_multiple(x_, y_, xlabel, ylabel,labels, pdf=None):
    
    plt.figure(figsize=(10, 6))
    
    if len(y_) != len(labels) or len(x_) != len(y_):
        raise RuntimeError("The number of labels is not the same as the number of different results to compare.")
    
    for i in range(len(labels)):
        plt.plot(x_[i],y_[i],label=labels[i], lw=2.2)
        
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(fontsize=20)
    plt.grid()
    plt.tight_layout()

    if pdf is not None:
        plt.savefig(pdf + ".pdf", format='pdf', dpi=300, bbox_inches='tight')

def plot_part4(x_, y_, xlabel, ylabel, best_pitch, pdf=None):
    plt.figure(figsize=(10, 6))
    plt.plot(x_, y_, lw=2.2, zorder=10)
    plt.axvline(best_pitch, color='k', linestyle="--", label=f"Best pitch = {best_pitch:.1f}°")
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.grid()
    plt.legend(fontsize=20)

    if pdf is not None:
        plt.savefig(pdf + ".pdf", format='pdf', dpi=300, bbox_inches='tight')
