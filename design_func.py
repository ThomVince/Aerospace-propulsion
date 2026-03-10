import numpy as np
from Clarky_polars.clarkypolarsRe import clarkypolarsRe

# ===========================================
# =============== functions =================
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
    phi =  np.arctan(np.tan(phi_t)/xi)
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
    aoa_ = np.linspace(-np.pi/2, np.pi/2, 1000) # angle of attack range [rad]
    cl_, cd_ = clarkypolarsRe(aoa_, Re)

    mask = cl_ > 0.0
    epsilon_ = cd_[mask] / cl_[mask]

    idx = np.argmin(epsilon_)

    epsilon = epsilon_[idx]
    alpha = aoa_[mask][idx]
    cl_opt = cl_[mask][idx]
    
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
    a_prime = (zeta/2 * x) * np.cos(phi) * np.sin(phi) * (1 + epsilon / np.tan(phi))
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
