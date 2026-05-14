import numpy as np
import matplotlib.pyplot as plt

from functions import design_loop

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


Omega = 2*np.pi*n               # propeller angular velocity [rad/s]
lambd = V/(Omega*R)             # speed ratio [-]
Tc = 2*T/(rho*V**2*np.pi*R**2)  # thrust coefficient [-]

# ===========================================
# ============ convergence study ============
# ===========================================

zeta_init = 0.0

N_values = [5, 10, 20, 50, 100, 150, 200]

zeta_vals  = []
eta_vals   = []

print(f"{'N':>5}  {'zeta':>8}  {'eta [%]':>10}")
print("-" * 30)

for N in N_values:
    xi_ = np.linspace(xi0, 1, N)
    zeta, cl_, epsilon_, alpha_, Wc_, a_, a_prime_, W_, c_, beta_, I1, I2, J1, J2, Pc, eta, sigma = design_loop(xi_, xi0, zeta_init, lambd, Tc, B, V, R, nu, Omega)
    zeta_vals.append(zeta)
    eta_vals.append(eta)
    print(f"{N:>6}  {zeta:>8.4f}  {eta*100:>10.2f}")

# ===========================================
# =================== plots =================
# ===========================================

plt.figure(figsize=(8, 5))
plt.plot(N_values, zeta_vals, 'o-', color='steelblue')
plt.axvline(100, linestyle='--', color='gray', lw=2.2, label='$N = 100$')
plt.xlabel('Number of blade sections $N$ [-]', fontsize=20)
plt.ylabel(r'Displacement velocity ratio $\zeta$ [-]', fontsize=20)
plt.legend(fontsize=20)
plt.grid()
plt.tick_params(labelsize=20)
plt.tight_layout()

# plt.savefig('convergence_zeta.pdf', format='pdf', dpi=300, bbox_inches='tight')

plt.figure(figsize=(8, 5))
plt.plot(N_values, eta_vals, 'o-', color='darkorange')
plt.axvline(100, linestyle='--', color='gray', lw=2.2, label='$N = 100$')
plt.xlabel('Number of blade sections $N$ [-]', fontsize=20)
plt.ylabel(r'Propulsive efficiency $\eta$ [%]', fontsize=20)
plt.legend(fontsize=20)
plt.grid()
plt.tick_params(labelsize=20)
plt.tight_layout()

# plt.savefig('convergence_eta.pdf', format='pdf', dpi=300, bbox_inches='tight')

plt.show()
