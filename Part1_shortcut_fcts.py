import numpy as np
from clarkypolarsRe import clarkypolarsRe
"""from clarkypolars import clarkypolars"""

nu = 1.5e-5

def flow_angle_at_tip(zeta , lmbda):
    return np.atan(lmbda*(1+zeta/2))

def prandtl_mom_loss_factor(xi, psi_tip, nb_blades):
    f = nb_blades/2 * (1-xi)/np.sin(psi_tip)
    return 2/np.pi * np.acos(np.exp(-f))

def flow_angle(zeta, xi, lmbda):
    return lmbda*(1+zeta/2)/xi

def circulation_fct(F,psi,xi,lmbda):
    return F*np.cos(psi)*np.sin(psi)*xi/lmbda

def Wc(lmbda,G,V,zeta,R,B,aoa,Re):
    return 4*np.pi*lmbda*G*V*R*zeta/(B*clarkypolarsRe(aoa,Re)[0])

def aoa_min_eps(Re):
    aoas = np.linspace(-np.pi,np.pi,1000)
    cl_tab, cd_tab = clarkypolarsRe(aoas,Re)
    inv_epsilons = cl_tab/cd_tab
    aoa = aoas[np.argmax(inv_epsilons)]
    eps = 1/max(inv_epsilons)
    return aoa, eps

def aoa_eps_Re_max_lift_to_drag(lmbda,G,V,zeta,R,B,tolerance_Re):
    
    cl_guess = 1
    
    Wc_init = 4*np.pi*lmbda*G*V*R*zeta/(B*cl_guess)
    diff_iter = tolerance_Re+1
    Res = [0,Wc_init/nu]
    
    nb_ite_tot = 0
    
    while tolerance_Re < diff_iter and nb_ite_tot < 1000:

        Re = Res[1]
        
        aoa, eps = aoa_min_eps(Re)
        
        W_c = Wc(lmbda,G,V,zeta,R,B,aoa,Re)
        
        Res[0] = Re
        Res[1] = W_c/nu
        
        diff_iter = abs(Res[1]-Res[0])
        
        nb_ite_tot += 1
    
    if nb_ite_tot >= 1000:
        print("Convergence issue. Re(999) = ",Res[0]," ; Re(1000) = ",Res[1])
        return
    else:
        aoa, eps = aoa_min_eps(Res[1])
        return aoa, eps, Res[1]
    
def ax_interf_factor(zeta,psi,eps):
    return (1 - eps * np.tan(psi)) * zeta/2 * (np.cos(psi))**2

def rot_interf_factor(zeta,psi,eps,x):
    return (1 + eps/np.tan(psi)) * zeta/(2*x) * np.cos(psi) * np.sin(psi)

def rel_velocity(a,v,psi):
    return v*(1+a)/np.sin(psi)

def derivatives(xi,G,eps,psi,lmbda):
    
    I1_der = 4 * xi * G * (1 - eps * np.tan(psi))
    I2_der = lmbda * I1_der/(2*xi) * (1+eps/np.tan(psi)) * np.sin(psi) * np.cos(psi)
    J1_der = 4*xi*G*(1+eps/np.tan(psi))
    J2_der = J1_der/2*(1-eps*np.tan(psi))*np.cos(psi)**2
    
    return I1_der,I2_der,J1_der,J2_der


