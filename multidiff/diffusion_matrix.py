import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy import optimize

# ---------------------------- Preprocessing ----------------------------------

def _dc_from_profiles(profiles):
    dc, c_mean = [], []
    for profile in profiles:
        n_comp, n_pts = profile.shape
        n_average = max(5, n_pts // 10)
        c_left = np.median(profile[:, :n_average], axis=1)
        c_right = np.median(profile[:, -n_average:], axis=1)
        dc.append(c_right - c_left)
        c_mean.append(0.5 * (c_right + c_left))
    return np.array(dc), np.array(c_mean)


def _normalize_profiles(profiles, c_mean):
    """
    Rescale concentration profiles to fluctuations around the average value.
    """
    normalized_profiles = []
    for profile, concs in zip(profiles, c_mean):
        normalized_profiles.append(profile - concs[:, None])
    return normalized_profiles


def preprocessing(profiles, nb_temp=1):
    """
    Parameters
    ----------

    profiles : list of arrays
        list containing n_exp arrays, where n_exp is the number of experiments,
        and each array contains the concentration profiles for different
        chemical elements.

    Returns
    -------
    normalized_profiles : list of arrays
        list of profiles for which the mean concentration has been substracted
        for each element, so that the mean concentration is zero for each
        profile

    dc : ndarray of shape n_exp x n_elements
        Concentration difference between the two ends of the profile.

    c_mean : ndarray of shape n_exp x n_elements
        Average of concentration along each profile.
    """
    if nb_temp > 1:
        dc, c_mean, normalized_profiles = [], [], []
        for profiles_temp in profiles:
            dc_temp, c_mean_temp = _dc_from_profiles(profiles_temp)
            normalized_profiles_temp = _normalize_profiles(profiles_temp,
                                                           c_mean_temp)
            dc.append(dc_temp)
            c_mean.append(c_mean_temp)
            normalized_profiles.append(normalized_profiles_temp)
    else:
        dc, c_mean = _dc_from_profiles(profiles)
        normalized_profiles = _normalize_profiles(profiles, c_mean)
    return normalized_profiles, dc, c_mean


# ------------------ Diffusion matrix -----------------------------------

def _best_number_subplots(i):
    # not used at the moment; test??
    subplots_number = {3:(1, 3), 4:(2, 2), 5:(2, 3), 6:(2,3), 7:(3, 3),
                       8:(3, 3)}
    return subplots_number[i]


def evolve_profile(diff_matrix, x_points, dc_init, exp_norm_profiles=None,
                   plot=True, return_data=False, labels=None):
    """
    Theoretical diffusion profile, given the diffusion matrix.

    Parameters
    ----------

    diff_matrix : tuple
        diffusion matrix, tuple of (eigenvalues, eigenvectors). In the
        reduced basis of dimension n - 1.
    x_points : array
        spatial coordinates (1-D array)
    dc_init : array
        concentration delta between endmembers (for the n species)
    exp_norm_profiles : array
        experimental profiles, to be compared with theoretical ones.
        The mean of the profile should be zero (ie,
        call ``normalize_profiles``).

    Returns
    -------
    """
    do_legend = True
    diag, P = diff_matrix
    P = np.matrix(P)
    dc = dc_init[:-1] #TODO
    orig = P.I * dc[:, None] / 2.
    n_comps = len(diag)
    profs = np.empty((n_comps, len(x_points)))
    for i in range(n_comps):
        profs[i] = orig[i] * erf(x_points / np.sqrt(4 * diag[i]))
    profiles = P * np.matrix(profs)
    if plot:
        if labels is None:
            do_legend = False
            labels = [str(ind) for ind in range(n_comps + 1)]
        plt.figure()
        colors = ['r', 'c', 'm', 'y', 'k', 'g']
        profiles = np.array(profiles)
        for i in range(n_comps):
            plt.plot(x_points, profiles[i], 
                                color=colors[i], label=labels[i], lw=2)
            if exp_norm_profiles is not None:
                plt.plot(x_points, 
                        exp_norm_profiles[i], 'o', color=colors[i])

        plt.plot(x_points, -profiles.sum(axis=0),
                    color=colors[i + 1], label=labels[i + 1], lw=2)
        if exp_norm_profiles is not None:
            plt.plot(x_points, exp_norm_profiles[-1], 'o', color=colors[i + 1])
        plt.ylabel('concentrations', fontsize=20)
        plt.xlabel(u'$x/\sqrt{t}$', fontsize=24)
        if do_legend:
            plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
    error = (profiles - (exp_norm_profiles[:-1]))
    if return_data:
        return profiles, exp_norm_profiles
    else:
        return np.ravel(error)


def optimize_profile(diff_matrix, x_points, dc_init, exp_norm_profiles,
                     display_result=True, labels=None):
    """
    Fit the diffusion matrix

    Parameters
    ----------

    diff_matrix : tuple
        tuple of (eigenvalues, eigenvectors) in reduced basis (dim n-1)
    x_points : 1-D array_like
        spatial coordinates
    dc_init : array
        concentration difference between endmembers
    exp_norm_profiles : list of arrays
        profiles to be fitted, of length the nb of experiments, with n
        profiles for each experiment. Profiles are normalized, that is, an
        estimation of the estimated mean concentration should be substracted.
    """
    n_comp = len(dc_init[0]) - 1
    n_exp = len(x_points)

    def cost_function(coeffs, x_points, dc_init, exp_norm_profiles):
        n_comp = len(dc_init[0]) - 1
        diag = coeffs[:n_comp]
        n_exp = len(x_points)
        P = np.matrix(coeffs[n_comp: n_comp + n_comp**2].reshape((n_comp,
                                                                  n_comp)))
        adjust_cmeans = coeffs[n_comp + n_comp**2:
                               n_comp + n_comp**2 + 
                               (n_comp) * n_exp].reshape((n_exp, n_comp))
        adjust_dc = coeffs[n_comp + n_comp**2 + (n_comp) * n_exp:
                           n_comp + n_comp**2 + 
                           2 * (n_comp) * n_exp].reshape((n_exp, n_comp))
        errors = np.array([])
        for i in range(n_exp):
            dc_corr = np.copy(dc_init[i])
            dc_corr[:-1] -= adjust_dc[i]
            profile_corr = np.copy(exp_norm_profiles[i])
            profile_corr[:-1, :] -= adjust_cmeans[i][:, None]
            error = evolve_profile((diag, P), x_points[i], dc_corr, profile_corr, plot=False)
            errors = np.concatenate((errors, error))
        return errors

    diag, P = diff_matrix
    coeffs = np.concatenate((diag, np.array(P).ravel(),
                             np.zeros(2 * n_exp * n_comp)))
    res = optimize.leastsq(cost_function, coeffs,
                           args=(x_points, dc_init, exp_norm_profiles),
                           full_output=True, factor=10)[0]
    diags, eigvecs, shifts =  res[:n_comp], \
           res[n_comp: n_comp + n_comp**2].reshape((n_comp, n_comp)), \
           res[n_comp + n_comp**2:].reshape((2, n_exp, n_comp))
    if display_result:
       for i in range(n_exp):
            dc_corr = np.copy(dc_init[i])
            dc_corr[:-1] -= shifts[1, i]
            prof_corr = np.copy(exp_norm_profiles[i])
            prof_corr[:-1] -= shifts[0, i][:, None]
            _ = evolve_profile((diags, eigvecs), x_points[i], dc_corr,
                    exp_norm_profiles=prof_corr, labels=labels) 
    return diags, eigvecs, shifts


def optimize_eigvals(diff_matrix, x_points, dc_init, exp_norm_profiles,
                     display_result=True, labels=None):
    """
    Fit the diffusion matrix

    Parameters
    ----------

    diff_matrix : tuple
        tuple of (eigenvalues, eigenvectors) in reduced basis (dim n-1)
    x_points : 1-D array_like
        spatial coordinates
    dc_init : array
        concentration difference between endmembers
    exp_norm_profiles : list of arrays
        profiles to be fitted, of length the nb of experiments, with n
        profiles for each experiment. Profiles are normalized, that is, an
        estimation of the estimated mean concentration should be substracted.
        
    """
    n_comp = len(dc_init[0]) - 1
    n_exp = len(x_points)

    def cost_function(coeffs, p_matrix, x_points, dc_init, exp_norm_profiles):
        n_comp = len(dc_init[0]) - 1
        diag = coeffs[:n_comp]
        n_exp = len(x_points)
        adjust_cmeans = coeffs[n_comp:
                               n_comp + (n_comp) * n_exp].reshape((
                                                            n_exp, n_comp))
        adjust_dc = coeffs[n_comp + (n_comp) * n_exp:
                           n_comp + 2 * (n_comp) * n_exp].reshape((
                                                            n_exp, n_comp))
        errors = np.array([])
        for i in range(n_exp):
            dc_corr = np.copy(dc_init[i])
            dc_corr[:-1] -= adjust_dc[i]
            profile_corr = np.copy(exp_norm_profiles[i])
            profile_corr[:-1, :] -= adjust_cmeans[i][:, None]
            error = evolve_profile((diag, p_matrix), x_points[i], dc_corr,
                                        profile_corr, plot=False)
            errors = np.concatenate((errors, error))
        return errors

    diag, P = diff_matrix
    coeffs = np.concatenate((diag, np.zeros(2 * n_exp * n_comp)))
    res = optimize.leastsq(cost_function, coeffs,
                           args=(P, x_points, dc_init, exp_norm_profiles),
                           ftol=1.e-15, full_output=True, factor=10)[0]
    diags, shifts =  res[:n_comp], \
                     res[n_comp:].reshape((2, n_exp, n_comp))
    if display_result:
       for i in range(n_exp):
            dc_corr = np.copy(dc_init[i])
            dc_corr[:-1] -= shifts[1, i]
            prof_corr = np.copy(exp_norm_profiles[i])
            prof_corr[:-1] -= shifts[0, i][:, None]
            _ = evolve_profile((diags, P), x_points[i], dc_corr,
                    exp_norm_profiles=prof_corr, labels=labels) 
    return diags, shifts


def optimize_profile_multi_temp(diff_matrix, x_points, dc_init,
                exp_norm_profiles, display_result=True, labels=None):
    """
    Fit the diffusion matrix
    Parameters
    ----------
    diff_matrix : tuple
        tuple of (eigenvalues, eigenvectors) in reduced basis (dim n-1)
    x_points : 1-D array_like
        spatial coordinates
    dc_init : array
        concentration difference between endmembers
    exp_norm_profiles : list of arrays
        profiles to be fitted, of length the nb of experiments, with n
        profiles for each experiment. Profiles are normalized, that is, an
        estimation of the estimated mean concentration should be substracted.
    """
    n_comp = len(dc_init[0][0]) - 1
    n_temp = len(x_points)
    n_eigvals = n_temp * n_comp
    n_exp = np.array([len(x) for x in x_points])

    def cost_function(coeffs, x_points, dc_init, exp_norm_profiles):
        diag = coeffs[:n_eigvals].reshape((n_temp, n_comp))
        P = np.matrix(coeffs[n_eigvals: n_eigvals + n_comp**2].reshape((n_comp,
                                                                        n_comp)))
        errors = np.array([])
        for i in range(n_temp):
            for j in range(n_exp[i]):
                profile_corr = np.copy(exp_norm_profiles[i][j])
                error = evolve_profile((diag[i], P), x_points[i][j], dc_init[i][j],
                                            profile_corr, plot=False)
                errors = np.concatenate((errors, error))
        return errors

    diag, P = diff_matrix
    coeffs = np.concatenate((diag, np.array(P).ravel(),
                             np.zeros(2 * n_exp.sum() * n_comp)))
    res = optimize.leastsq(cost_function, coeffs,
                           args=(x_points, dc_init, exp_norm_profiles),
                           ftol=1.e-15, full_output=True, factor=10)[0]
    diags, eigvecs =  res[:n_eigvals].reshape((n_temp, n_comp)), \
           res[n_eigvals: n_eigvals + n_comp**2].reshape((n_comp, n_comp))
    if display_result:
        for i in range(n_temp):
            for j in range(n_exp[i]):
                _ = evolve_profile((diags[i], eigvecs), x_points[i][j], dc_init[i][j],
                    exp_norm_profiles=exp_norm_profiles[i][j], labels=labels) 
    return diags, eigvecs


def compute_diffusion_matrix(diff_matrix, x_points, profiles, plot=True,
                                eigvals_only=False, labels=None, nb_temp=1):
    """
    Compute a best fit for the diffusion matrix, given a set of experimental
    concentration profiles.

    Parameters
    ----------

    diff_matrix : tuple
        diffusion matrix, tuple of (eigenvalues, eigenvectors). In the
        reduced basis of dimension n - 1.
    x_points : array
        spatial coordinates (1-D array)
    profiles : list of array
        experimental profiles, to be compared with theoretical ones. 
    plot : bool, default True
        if True, the result of the fit is plotted together with experimental
        profiles
    eigvals_only : bool, default False
        if True, the algorithm assumes that eigenvectors have already been
        determined, and optimizes over the eigenvalues only.
    nb_temp : int, default 1
        if > 1, input data have been taken at different temperatures

    Returns
    -------

    diags : array
        Eigenvalues of the diffusion matrix
    eigvecs : array
        Eigenvectors of the diffusion matrix, in the basis with all elements
        (ie, the sum of all coefficients is zero)
    norm_profiles : list of arrays
        theoretical profiles
    fitted_profiles : list of arrays
        experimental (normalized) profiles
    shifts : array
        small corrections to concentration mean and difference computed by the
        optimization algorithm
    """
    normalized_profiles, dc, c_mean = preprocessing(profiles, nb_temp=nb_temp)
    if eigvals_only:
        diags, shifts = optimize_eigvals(diff_matrix, x_points, dc,
                                              normalized_profiles,
                                              display_result=plot)
        eigvecs = diff_matrix[1]
    if nb_temp > 1:
        diags, eigvecs = optimize_profile_multi_temp(diff_matrix, x_points, dc,
                                              normalized_profiles,
                                              display_result=plot,
                                              labels=labels)
        eigvecs = eigvecs_to_fulloxides(eigvecs)
        return diags, eigvecs, normalized_profiles
    else:
        diags, eigvecs, shifts = optimize_profile(diff_matrix, x_points, dc,
                                              normalized_profiles,
                                              display_result=plot,
                                              labels=labels)
    fitted_profiles = []
    norm_profiles = []
    # Post-processing, only if nb_temp = 1
    for i in range(len(dc)):
        dc_corr = np.copy(dc[i])
        dc_corr[:-1] -= shifts[1, i]
        prof_corr = np.copy(normalized_profiles[i])
        prof_corr[:-1] -= shifts[0, i][:, None]
        output = evolve_profile((diags, eigvecs), x_points[i], dc_corr,
                                exp_norm_profiles=prof_corr,
                                return_data=True, plot=False)
        fitted_profiles.append(np.array(output[0]))
        norm_profiles.append(output[1])
    eigvecs = eigvecs_to_fulloxides(eigvecs)
    return diags, eigvecs, norm_profiles, \
            fitted_profiles, shifts

# ------------------ Post-processing ----------------------------------------

def eigvecs_to_fulloxides(eigvecs):
    all_eigvec = np.vstack((eigvecs, -eigvecs.sum(axis=0)))
    return all_eigvec / np.max(np.abs(all_eigvec), axis=0)
