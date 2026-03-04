import numpy as np

from design_func import *

# ===========================================
# ============ design parameters ============
# ===========================================

B = 3       # number of blades
R = 1.7/2   # propeller tip radius [m]
xi0 = 0.3/R # nondimensional hub radius [-]

T_req = 500 # required thrust [N]
V = 45      # freestream takeoff velocity [m/s]
rho = 1.225 # air density [kg/m^3]
n = 2100/60 # rotational speed [rev/s]

nu = 1.5e-5 # kinematic viscosity [m^2/s]

N = 50      # number of blade sections

# ===========================================
# =========== derived parameters ============
# ===========================================

Omega = 2*np.pi*n   # propeller angular velocity [rad/s]
lambd = V/(Omega*R) # speed ratio [-]

# ===========================================
# ============ design procedure =============
# ===========================================

zeta = 0.5 # initial guess for the displacement velocity ratio

phi_ = np.zeros(N)   # flow angle at each blade section [rad]
xi_ = np.linspace(xi0, 1, N) # nondimensional blade section radii [-]

F_, phi_ = F_phi(xi_, zeta, lambd, B)

cl_ = np.zeros(N) # lift coefficient at each blade section [-]
epsilon_ = np.zeros(N) # drag-to-lift ratio at each blade section [-]
alpha_ = np.zeros(N) # angle of attack at each blade section [rad]
Wc_ = np.zeros(N) # relative velocity at each blade section [m/s]

x = Omega * xi_ * R / V

G_ = np.array([F_[i] * x[i] * np.cos(phi_[i]) * np.sin(phi_[i]) for i in range(N)])

cl = 1.0 # initial guess for the lift coefficient of the first section

for i in range(N):
    for j in range(10):
        Wc, Re = Wc_Re(lambd, cl, G_[i], V, R, zeta, B, nu)

        epsilon, alpha, cl_opt = epsilon_alpha(Re)

        if abs(cl_opt - cl) < 1e-3:
            break
        
        cl = cl_opt

    cl_[i] = cl
    epsilon_[i] = epsilon
    alpha_[i] = alpha
    Wc_[i] = Wc

a_ = np.zeros(N)
a_prime_ = np.zeros(N)
W_ = np.zeros(N)
c_ = np.zeros(N)
beta_ = np.zeros(N)

for i in range(N):
    a_[i], a_prime_[i], W_[i] = a_a_prime_W(zeta, phi_[i], epsilon_[i], x[i], V)
    c_[i] = Wc_[i] / W_[i]
    beta_[i] =  alpha_[i] + phi_[i]

    I1_prime, I2_prime, J1_prime, J2_prime = I_prime_J_prime(xi_[i], G_[i], phi_[i], epsilon_[i], lambd)
