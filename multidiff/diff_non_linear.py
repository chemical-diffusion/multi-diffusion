import os
import numpy as np
from scipy import integrate, interpolate, optimize
from math import exp

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

def diffusion_equation(Y, t):
    y, yp = Y
    return [yp, -2 * t * yp / y]


def g_coeff(delta_gamma):
    path = os.path.join(dir_path, 'g_coeffs.npy')
    coeffs = np.load(path)
    return np.polyval(coeffs, delta_gamma)


def compute_profile_crank(beta, D_pre, x_prof):
    C1 = -1
    C2 = 1
    D_a = D_pre * exp(-beta)
    delta_gamma = beta * (C2- C1)
    x_max = np.abs(x_prof).max()
    y_max = x_max / (2. * np.sqrt(D_a))

    spatial_coor = np.linspace(0, y_max, 2000)
    g = g_coeff(delta_gamma)

    u_fwd, _ = integrate.odeint(diffusion_equation, [1, g], spatial_coor).T
    gam_fwd = np.log(u_fwd)

    u_bk, _ = integrate.odeint(diffusion_equation, [1, g], -spatial_coor).T
    gam_bk = np.log(u_bk)

    gam_0 = 0.5 * ( gam_fwd[-1] + gam_bk[-1])
    D_0 = D_a * exp(-gam_0)

    coor = np.concatenate((-spatial_coor[::-1], spatial_coor))
    prof_gamma = np.concatenate((gam_bk[::-1], gam_fwd))
    x = 2 * np.sqrt(D_0) * coor
    psi = 2 * (prof_gamma - gam_0) / delta_gamma
    interp = interpolate.interp1d(x, psi, bounds_error=False,
                                          fill_value='extrapolate')
    return 0.5 * (1 - interp(x_prof))


def _error_function_crank(coeffs, x, exp_profile):
    beta, D_pre = coeffs
    psi = compute_profile_crank(beta, D_pre, x)
    return psi - exp_profile


def fit_crank(x, exp_profile, coeffs_init):
    """
    Here we impose bounds to help the fit converge. For the beta value the
    upper value is 10 because is already too stiff to converge.
    """
    res = optimize.least_squares(_error_function_crank, coeffs_init,
                    args=(x, exp_profile), bounds=([0, 0], [10, np.inf]))

    return res['x']
