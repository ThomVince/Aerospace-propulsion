import numpy as np
from Part1_shortcut_fcts import *
import matplotlib.pyplot as plt

#Constants
rho = 1.225

#Propeller characteristics
nb_blades = 3
tip_diam = 1.7
hub_diam = 0.3

#Part 1 attributes
thrust = 500
rot_speed_RPM = 2100
v_freestream = 45

#Code

nb_sections = 30

zeta_init = 0

R = (tip_diam)/2
Omega =  rot_speed_RPM * 2 * np.pi / 60
lmbda = v_freestream/(Omega*R)
Tc = 2*thrust/(np.pi*rho*(v_freestream**2)*(R**2))

xis = np.linspace(hub_diam,tip_diam,nb_sections)/(2*R)

chords = np.zeros_like(xis)
twists = np.zeros_like(xis)

zetas = [-1,0]

Pc = 0

while abs((zetas[1]-zetas[0])) > 0.001*abs(zetas[0]):

    zeta = zetas[1]

    psi_t = flow_angle_at_tip(zeta,lmbda)

    I1_dervs = np.zeros_like(xis)
    I2_dervs = np.zeros_like(xis)
    J1_dervs = np.zeros_like(xis)
    J2_dervs = np.zeros_like(xis)



    for i in range(nb_sections):
        
        xi = xis[i]
        
        #2
        F = prandtl_mom_loss_factor(xi,psi_t,nb_blades)
        psi = flow_angle(zeta,xi,lmbda)
        
        #3, 4, 5
        G = circulation_fct(F,psi,xi,lmbda)
        aoa, eps, Re = aoa_eps_Re_max_lift_to_drag(lmbda,G,v_freestream,zeta,R,nb_blades,1)
        W_times_c = Wc(lmbda,G,v_freestream,zeta,R,nb_blades,aoa,Re)
        
        #6
        a = ax_interf_factor(zeta,psi,eps)
        a_prime = rot_interf_factor(zeta,psi,eps,xi/lmbda)
        W = rel_velocity(a,v_freestream,psi)
        
        #7
        chords[i] = W_times_c/W
        twists[i] = aoa + psi
        
        I1_dervs[i], I2_dervs[i],J1_dervs[i],J2_dervs[i] = derivatives(xi,G,eps,psi,lmbda)
        
    I1 = np.trapezoid(I1_dervs,xis)
    I2 = np.trapezoid(I2_dervs,xis)
    J1 = np.trapezoid(J1_dervs,xis)
    J2 = np.trapezoid(J2_dervs,xis)

    zetas[0] = zeta
    zetas[1] = I1/(2*I2) - ((I1/(2*I2))**2 - Tc/I2)**(1/2)
    
    Pc = J1*zetas[1] + J2*zetas[1]**2

print("The final value for zeta is : ",zetas[1])

plt.plot(xis,chords)
plt.grid()
plt.xlabel("Non-dimensionnal blade section radius [-]")
plt.ylabel("Chord length [m]")
plt.show()
    
    