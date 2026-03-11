import numpy as np
from clarkypolarsRe import clarkypolarsRe
"""from clarkypolars import clarkypolars"""

nu = 1.5e-5

def flow_angle_at_tip(zeta , lmbda):
    psi_t = np.arctan(lmbda*(1+zeta/2))
    return psi_t

def prandtl_mom_loss_factor_flow_angle(xi, psi_tip, B):
    f = B/2 * (1-xi)/np.sin(psi_tip)
    F = 2/np.pi * np.arccos(np.exp(-f))
    psi = np.arctan(np.tan(psi_tip)/xi)
    return F,psi

def circulation_fct(F,psi, x):
    return F*np.cos(psi)*np.sin(psi)*x

def Re_AoA_Eps_Cls_opt(lmbd,G,V,zeta,R,B):
    
    Res = np.zeros_like(G)
    AoAs = np.zeros_like(G)
    Epses = np.zeros_like(G)
    
    for i in range(len(G)):
        cls = [-1,1]
        nb_iter = 0
        eps = -1
        Re = -1
        aoa = -1
        
        while nb_iter < 10:
            if abs((cls[1]-cls[0])/cls[0]) < 0.001:
                break
            cls[0] = cls[1]
            Re = 4*np.pi*lmbd*G[i]*V*R*zeta/(cls[1]*B)/nu
            aoa_tab = np.linspace(-np.pi/2,np.pi/2,500)
            cl_tab,cd_tab = clarkypolarsRe(aoa_tab,Re)
            inv_eps = cl_tab/cd_tab
            index = np.argmax(inv_eps)
            eps = 1/inv_eps[index]
            aoa = aoa_tab[index]
            cls[1] = cl_tab[index]
            nb_iter += 1
            
        if nb_iter >=10:
            print("Convergence error.")
            return
        
        Res[i] = Re
        AoAs[i] = aoa
        Epses[i] = eps
    return Res,AoAs,Epses
            
            
            
        
    
def ax_interf_factor(zeta,psi,eps):
    return (1 - eps * np.tan(psi)) * zeta/2 * (np.cos(psi))**2

def rot_interf_factor(zeta,psi,eps,x):
    return (1 + eps/np.tan(psi)) * zeta/(2*x) * np.cos(psi) * np.sin(psi)

def rel_velocity(a,v,psi):
    return v*(1+a)/np.sin(psi)

def I_derivatives(xi,G,eps,psi,lmbda):
    
    I1_der = 4 * xi * G * (1 - eps * np.tan(psi))
    I2_der = lmbda * I1_der/(2*xi) * (1+eps/np.tan(psi)) * np.sin(psi) * np.cos(psi)
    
    return I1_der,I2_der
def J_derivatives(xi,G,eps,psi,lmbda):
    J1_der = 4 * xi * G * (1 + eps/np.tan(psi))
    J2_der = J1_der/2 * (1 - eps*np.tan(psi)) * np.cos(psi)**2
    
    return J1_der,J2_der

