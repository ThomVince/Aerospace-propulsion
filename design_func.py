import numpy as np
from clarkypolarsRe import clarkypolarsRe

# ===========================================
# =============== functions =================
# ===========================================

def F_phi(xi, zeta, lambd, B):
    """ Prandtl's momentum loss factor F and flow angle phi for a given blade section """
    phi_t = np.arctan(lambd * (1 + zeta/2)) # tip flow angle [rad]
    f = (B/2)*(1-xi)/np.sin(phi_t)
    F = (2/np.pi)*np.arccos(np.exp(-f))
    phi =  np.arctan(np.tan(phi_t)/xi)
    return F, phi

def Wc_Re(lambd, Cl_fixed, G, V, R, zeta, B, nu):
    """ Relative velocity at the blade section """
    Wc = 4*np.pi*lambd*G*V*R*zeta/(Cl_fixed*B)
    Re = Wc/nu
    return Wc, Re

def epsilon_alpha(Re):
    """ minimum drag-to-lift ratio epsilon and corresponding angle of attack alpha and cl """
    aoa_ = np.linspace(-np.pi/2, np.pi/2, 1000) # angle of attack range [rad]
    cl_, cd_ = clarkypolarsRe(aoa_, Re)

    mask = cl_ > 0.05
    epsilon_ = cd_[mask] / cl_[mask]

    idx = np.argmin(epsilon_)

    epsilon = epsilon_[idx]
    alpha = aoa_[mask][idx]
    cl_opt = cl_[mask][idx]
    
    return epsilon, alpha, cl_opt

def a_a_prime_W(zeta, phi, epsilon, x, V):
    a = (zeta/2) * np.cos(phi)**2 * (1 - epsilon * np.tan(phi))
    a_prime = (zeta/2 * x) * np.cos(phi) * np.sin(phi) * (1 + epsilon / np.tan(phi))
    W = V * (1 + a) / np.sin(phi)
    return a, a_prime, W

def I_prime_J_prime(xi, G, phi, epsilon, lambd):
    I1_prime = 4 * xi * G * (1 - epsilon*np.tan(phi))
    I2_prime = lambd * (I1_prime / (2*xi)) * (1 + epsilon/np.tan(phi)) * np.sin(phi) * np.cos(phi)
    J1_prime = 4*xi*G*(1 + epsilon/np.tan(phi))
    J2_prime = (J1_prime/2) * (1 - epsilon*np.tan(phi)) * np.cos(phi)**2
    return I1_prime, I2_prime, J1_prime, J2_prime