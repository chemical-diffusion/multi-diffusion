"""
Fitting non-linear profiles
===========================

"""

import numpy as np
from scipy import optimize, integrate
import matplotlib.pyplot as plt
from multidiff import fit_crank, compute_profile_crank
from derivative import dxdt


# Define a diffusion profile from a diffusion couple,
# where the diffusion coefficient depends exponentially on concentration
D = .1 # diffusion coefficient in the most mobile phase
beta = 2.5 # coefficient of the exponential dependence
# The diffusivity in the least mobile phase is D exp(-2 beta)

l = 400
x = np.linspace(-1, 1, l)

# Concentration profile
c = compute_profile_crank(beta, D, x)
c += 0.03 * np.random.randn(l)
c_interval = np.linspace(0, 1, 100)

# %%
# There exists a closed-form of the diffusion equation for the specific case of
#
# - a diffusion-couple geometry (two semi-infinite media of constant concentration put
#   in contact at t=0)
# - an exponential dependence of the diffusivity with concentration
# Therefore we can directly fit the diffusion profile for the ``beta`` and ``D`` parameters,
# by using the function ``fit_crank``.
params = fit_crank(x, c, (1, 1))
print(params)
c_fitted = compute_profile_crank(*params, x)


# %%
# For the general case of a non-constant diffusivity, the Boltzmann-Matano methods
# estimates locally the diffusivity. This methods is explained for example in 
# https://en.wikipedia.org/wiki/Boltzmann%E2%80%93Matano_analysis 
# The local diffusion coefficient is given by
#
# .. math::
#    D(c^*) = \frac{-1}{2t} \frac{\int_{c^*}^{c_L} x \mathrm{d}c}{(\mathrm{d}c / \mathrm{d}x)_{x=x^*}}
#
# The main difficulty is to estimate the derivative of ``c`` with ``x`` from a noisy
# signal. It is therefore essential to smooth the signal before derivating it.
# Here we use the smooth derivative function from the ``derivative`` package. It uses
# a Savitsky-Golay filter, which fits a polynomial (here of order 4) inside a local
# window of width 0.1. 
smooth_derivative = dxdt(c, x, kind="savitzky_golay", left=.05, right=.05, order=4)

# Compute the integral. In Matano's formula it is possible to compute it 
# from the left or from the right. Here there are more variations in the right
# part of the signal so we compute the integral from the right (and hence reverse
# the signal for the integral).
integrate_signal = integrate.cumtrapz(x[::-1], c[::-1])[::-1]

# Matano's formula
D_matano = -1/2 * integrate_signal / smooth_derivative[:-1]

mask = np.logical_and(c > 0.25, c < 0.75)[:-1]

def exp_fit(c, bet, A):
    return A * np.exp(-2 * bet * c)

params_exp = optimize.curve_fit(exp_fit, c[:-1][mask], 
               D_matano[mask])[0]

print(params_exp)

fig, ax = plt.subplots(nrows=2)
ax[0].plot(x, c, 'o')
ax[0].plot(x, c_fitted)
ax[0].set_xlabel('$x$')
ax[0].set_ylabel('$C$')
ax[1].semilogy(c[:-1], D_matano, 'o', label='Matano')
ax[1].semilogy(c[:-1][mask], D_matano[mask], 'o')
ax[1].set_xlabel('$C$')
ax[1].set_ylabel('$D(C)$')
ax[1].semilogy(c_interval, D * np.exp(-2 * beta * c_interval), label='ground truth')
ax[1].semilogy(c_interval, params[1] * np.exp(-2 * params[0] * c_interval), label='fit')
ax[1].legend()
plt.tight_layout()
plt.show()

# %%
# Robustness of the fit
#
# As we see in the figure above, the Matano estimation is noisier than the direct fit. 
# In the figure below, we compute the same estimations but for different realizations 
# of the noise in the concentration profile. For some realizations, the Matano estimate
# is quite bad, while the direct fit is always very close to the ground truth.
#
# When a parametrization of the diffusivity is known and it is possible to compute 
# numerically the solution of the nonlinear diffusion equation with such a parametrization,
# it is therefore much robust to fit the parameters directly. However, the Matano method
# has the great advantage of being completely generic.

fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(12, 8))
for i in range(12):
    row = i // 4
    col = i % 4
    c = compute_profile_crank(beta, D, x)
    c += 0.03 * np.random.randn(l)
    params = fit_crank(x, c, (1, 1))
    c_fitted = compute_profile_crank(*params, x)

    # Matano
    smooth_derivative = dxdt(c, x, kind="savitzky_golay", left=.1, right=.1, order=3)

    # Matano method
    integrate_signal = integrate.cumtrapz(x[::-1], c[::-1])[::-1]
    D_matano = -1/2 * integrate_signal / smooth_derivative[:-1]

    params_exp = optimize.curve_fit(exp_fit, c[:-1][mask], 
               D_matano[mask])[0]
    print(params_exp)
    ax[row, col].semilogy(c[:-1], D_matano, 'o', label='Matano')
    ax[row, col].semilogy(c[:-1][mask], D_matano[mask], 'o')
    ax[row, col].semilogy(c_interval, D * np.exp(-2 * beta * c_interval), label='ground truth')
    ax[row, col].semilogy(c_interval, params[1] * np.exp(-2 * params[0] * c_interval), label='fit')
    ax[row, col].set_xlabel('$C$')
    ax[row, col].set_ylabel('$D(C)$')
plt.tight_layout()
plt.show()



