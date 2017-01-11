import numpy as np
from scipy.special import erf


def create_diffusion_profiles(diff_matrix, x_points, exchange_vectors,
                              noise_level=0.02, seed=0):
    """
    Compute theoretical concentration profiles, given the diffusion matrix and
    exchange directions.

    Parameters
    ----------

    diff_matrix : tuple
        tuple of eigenvalues and eigenvectors

    x_points : list of arrays
        points at which profiles are measured. There are as many profiles as
        exchange vectors.

    exchange_vectors : array
        array where each column encodes the species that are exchanged in the
        diffusion experiment. There are as many columns as diffusion
        experiments.

    noise_level : float
        Gaussian noise can be added to the theoretical profiles, in order to
        simulation experimental noise.
    """
    gen = np.random.RandomState(seed)
    diags, P = diff_matrix
    exchanges = exchange_vectors[:-1]
    n_comps = exchanges.shape[0]
    if n_comps != P.shape[0]:
        raise ValueError("Exchange vectors must be given in the full basis")
    concentration_profiles = []
    for i_exp, x_prof in enumerate(x_points):
        orig = P.I * exchanges[:, i_exp][:, None] / 2.
        profs = np.empty((n_comps, len(x_prof)))
        for i in range(n_comps):
            profs[i] = orig[i] * erf(x_prof / np.sqrt(4 * diags[i]))
        profiles = np.array(P * np.matrix(profs))
        profiles = np.vstack((profiles, - profiles.sum(axis=0)))
        concentration_profiles.append(np.array(profiles) + 
                                noise_level * gen.randn(*profiles.shape))
    return concentration_profiles
