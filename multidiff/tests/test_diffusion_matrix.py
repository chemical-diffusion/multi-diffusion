import numpy as np
from multidiff import compute_diffusion_matrix, create_diffusion_profiles


def test_input_matrix():
    x_prof = np.linspace(-20, 20, 100)
    x_points = [x_prof, x_prof, x_prof]
    diags = np.array([1, 5])
    P = np.matrix([[1, 1], [-1, 0]])
    dc = np.array([[1, 1, 0],
                   [-1, 0, 1],
                   [0, -1, -1]])
    # No noise
    concentration_profiles = create_diffusion_profiles((diags, P),
                                            x_points, dc, noise_level=0)
    diags_init = np.array([1, 1])
    P_init = np.eye(2)
    diags_res, eigvecs, _, _, _ = compute_diffusion_matrix((diags_init, P_init),
                                           x_points, concentration_profiles,
                                           plot=False)
    assert np.allclose(np.sort(diags), np.sort(diags_res), rtol=1.e-5)

    # Small noise
    concentration_profiles = create_diffusion_profiles((diags, P),
                                            x_points, dc, noise_level=0.02)
    diags_init = np.array([1, 1])
    P_init = np.eye(2)
    diags_res, eigvecs, _, _, _ = compute_diffusion_matrix((diags_init, P_init),
                                           x_points, concentration_profiles,
                                           plot=False)
    assert np.allclose(np.sort(diags), np.sort(diags_res), rtol=0.06)




