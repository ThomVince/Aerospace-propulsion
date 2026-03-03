"""
This python function is adapted from the MATLAB clarkypolarsRe(aoa, re) function written by 
Thomas Lambert in 2023.

CLARKYPOLARSRE Returns the lift and drag coefficient for a Clark-Y airfoil corresponding to 
one combination of angle of attack (in radians) and Reynolds number for a Clark-Y airfoil.

This script works by interpolating hard-coded values for the polars. The original values were 
obtained using XFoil for angles of attack between -20 and 20 degrees and Reynolds between 1e4 
and 1e7 for a Mach number of 0. The polars were then extended over the entire range of angles 
of attacks using empirical formulas.

IMPORTANT:
The function works by loading the clarkypolars.mat file, which contains a polar structure 
whose fields are
    - reynolds (:,1): the reynolds for each polar
    - aoa (:,length(reynolds)): the angles of attack for each polar
    - cl (size(aoa,1),length(reynolds)): the cl  for each polar
    - cd (size(aoa,1),length(reynolds)): the cd for each polar

Note:
The extension of the polar was realised following the "Viterna" method. See Viterna & 
Corrigan. 1982. "Fixed Pitch Rotor Performance of Large Horizontal Axis Wind Turbines".
-----
Syntax:
cl, cd = clarkypolarsRe(aoa, re) returns the values for cl and cd of a Clark-Y airfoil for 
the angle of attack aoa and reynolds number re.

Inputs:
aoa: angle of attack in RADIANS. This is to be passed as a np.array.
re : Reynolds number. This is to be passed as a np.array.

Outputs:
cl: lift coefficient for the angle of attack passed as input.
cd: drag coefficient for the angle of attack passed as input.

----------------------------------------------------------------------------------------------
This script was written for the 2024 and 2026 BEMT project for the course of AERO0014 (Aerospace
Propulsion) at the University of Liege. It is a gross over simplification. Do not use it in
any practical application.
----------------------------------------------------------------------------------------------
(c) Copyright 2024 University of Liege
Author: Thomas Lambert, Maxime Borbouse <maxime.borbouse@uliege.be>
ULiege - Aeroelasticity and Experimental Aerodynamics - Design of Turbomachines
MIT License
"""

from mat4py import loadmat
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import matplotlib.pyplot as plt

def clarkypolarsRe(aoa, re):

    POLAR_DATA_FILE = 'Clarky_polars/clarkypolars.mat'

    # Load hard-coded polarData
    tmp = loadmat(POLAR_DATA_FILE)
    polarData = tmp['polarData']

    aoa = np.arctan2(np.sin(aoa), np.cos(aoa))

    # 2D interpolation
    cl_interpolator = RegularGridInterpolator((np.array(polarData['aoa'])[:, 0], np.array(polarData['reynolds'])), np.array(polarData['cl']), bounds_error=False, fill_value=None)
    cd_interpolator = RegularGridInterpolator((np.array(polarData['aoa'])[:, 0], np.array(polarData['reynolds'])), np.array(polarData['cd']), bounds_error=False, fill_value=None)

    cl = cl_interpolator((aoa, re))
    cd = cd_interpolator((aoa, re))

    return cl, cd

