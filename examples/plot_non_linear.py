"""
Fitting non-linear profiles
====================================================

"""

import numpy as np
from scipy import optimize, integrate
import matplotlib.pyplot as plt
from multidiff import fit_crank, compute_profile_crank
from derivative import dxdt

beta = 2.5
D = .1

l = 400
t = np.linspace(-1, 1, l)

x = compute_profile_crank(beta, D, t)
x += 0.02 * np.random.randn(l)
x_interval = np.linspace(0, 1, 100)

params = fit_crank(t, x, (1, 1))
print(params)
x_fitted = compute_profile_crank(*params, t)

# Matano
#x_smooth = dxdt(x, t, kind="kalman", alpha=.5)
smooth_derivative = dxdt(x, t, kind="savitzky_golay", left=.1, right=.1, order=4)

# Matano method
integrate_signal = integrate.cumtrapz(t[::-1], x[::-1])[::-1]
D_matano = -1/2 * integrate_signal / smooth_derivative[:-1]

mask = np.logical_and(x > 0.25, x < 0.75)[:-1]

def exp_fit(x, bet, A):
    return A * np.exp(-2 * bet * x)

params_exp = optimize.curve_fit(exp_fit, x[:-1][mask], 
               D_matano[mask])[0]

print(params_exp)

fig, ax = plt.subplots(nrows=2)
ax[0].plot(t, x, 'o')
ax[0].plot(t, x_fitted)
ax[1].semilogy(x[:-1], D_matano, 'o', label='Matano')
ax[1].semilogy(x[:-1][mask], D_matano[mask], 'o')
ax[1].semilogy(x_interval, D * np.exp(-2 * beta * x_interval), label='ground truth')
ax[1].semilogy(x_interval, params[1] * np.exp(-2 * params[0] * x_interval), label='fit')
plt.show()
