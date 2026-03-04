import numpy as np
import matplotlib.pyplot as plt

from design_func import *

# ===========================================
# ============ design parameters ============
# ===========================================

B = 3       # number of blades
R = 1.7/2   # propeller tip radius [m]
xi0 = 0.3/R # nondimensional hub radius [-]

T = 500 # required thrust [N]
V = 45      # freestream takeoff velocity [m/s]
rho = 1.225 # air density [kg/m^3]
n = 2100/60 # rotational speed [rev/s]

nu = 1.5e-5 # kinematic viscosity [m^2/s]

N = 30      # number of blade sections

# ===========================================
# =========== derived parameters ============
# ===========================================

Omega = 2*np.pi*n   # propeller angular velocity [rad/s]
lambd = V/(Omega*R) # speed ratio [-]
Tc = 2*T/(rho*V**2*np.pi*R**2) # thrust coefficient [-]

# ===========================================
# ============ design procedure =============
# ===========================================

zeta = 0.0 # initial guess for the displacement velocity ratio
cl = 1.0 # initial guess for the lift coefficient of the first section

phi_ = np.zeros(N)   # flow angle at each blade section [rad]
xi_ = np.linspace(xi0, 1, N) # nondimensional blade section radii [-]
cl_ = np.zeros(N) # lift coefficient at each blade section [-]
epsilon_ = np.zeros(N) # drag-to-lift ratio at each blade section [-]
alpha_ = np.zeros(N) # angle of attack at each blade section [rad]
Wc_ = np.zeros(N) # relative velocity at each blade section [m/s]

a_ = np.zeros(N)
a_prime_ = np.zeros(N)
W_ = np.zeros(N)
c_ = np.zeros(N)
beta_ = np.zeros(N)

I1_prime = np.zeros(N)
I2_prime = np.zeros(N)
J1_prime = np.zeros(N)
J2_prime = np.zeros(N)

while True:
    F_, phi_ = F_phi(xi_, zeta, lambd, B)

    x = Omega * xi_ * R / V

    G_ = np.array([F_[i] * x[i] * np.cos(phi_[i]) * np.sin(phi_[i]) for i in range(N)])

    for i in range(N):
        for j in range(10):
            Wc, Re = Wc_Re(lambd, cl, G_[i], V, R, zeta, B, nu)

            epsilon, alpha, cl_opt = epsilon_alpha(Re)

            if abs((cl_opt - cl)/cl) < 1e-3:
                break
            
            cl = cl_opt

        cl_[i] = cl
        epsilon_[i] = epsilon
        alpha_[i] = alpha
        Wc_[i] = Wc

    for i in range(N):
        a_[i], a_prime_[i], W_[i] = a_a_prime_W(zeta, phi_[i], epsilon_[i], x[i], V)
        c_[i] = Wc_[i] / W_[i]
        beta_[i] =  alpha_[i] + phi_[i]

        I1_prime[i], I2_prime[i], J1_prime[i], J2_prime[i] = I_prime_J_prime(xi_[i], G_[i], phi_[i], epsilon_[i], lambd)

    I1 = np.trapezoid(I1_prime, xi_)
    I2 = np.trapezoid(I2_prime, xi_)

    zeta_new = (I1/(2*I2)) - ((I1/(2*I2))**2 - Tc/I2)**0.5

    if abs(zeta_new - zeta) < 1e-3: # relative or absolute tolerance ?
        J1 = np.trapezoid(J1_prime, xi_)
        J2 = np.trapezoid(J2_prime, xi_)
        Pc = J1*zeta_new + J2*zeta_new**2
        eta = Tc / Pc
        sigma = (B*c_)/(2*np.pi*xi_*R)
        break
    
    zeta = zeta_new

plt.figure(figsize=(10, 6))
plt.plot(xi_, np.degrees(beta_))
plt.xlabel(r'Nondimensional blade section radius $\xi$ [-]', fontsize=15)
plt.ylabel(r'Angle of attack $\beta$ [deg]', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()

plt.show()
