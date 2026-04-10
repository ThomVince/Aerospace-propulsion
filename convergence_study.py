import numpy as np
import matplotlib.pyplot as plt

from design import design_loop, xi0,lambd,Tc,B,V,R,nu

N_ = [50,100,150,200,300,400]
error_lim = 1e-4
zetas = np.zeros(len(N_))
errs = np.zeros(len(N_)-1)
for i in range(len(N_)) :
    xi_     = np.linspace(xi0, 1, N_[i])
    zeta, cl_, epsilon_, alpha_, Wc_, a_, a_prime_, W_, c_, beta_, I1, I2, J1, J2, Pc, eta, sigma = design_loop(xi_, 0, lambd, Tc, B, V, R, nu)
    zetas[i] = zeta
    print("Run with ",N_[i]," sections done. zeta =",zeta)
    if i > 0:
        errs[i-1] = abs((zetas[i]-zetas[i-1])/zetas[i])

plt.plot(N_[1:len(N_)],errs)
plt.hlines(error_lim,N_[0],N_[-1])
plt.xlabel("Number of sections [-]")
plt.ylabel("Relative error [%]")
plt.grid()
plt.show()

