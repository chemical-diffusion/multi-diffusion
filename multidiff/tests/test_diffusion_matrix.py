import numpy as np
from multidiff import compute_diffusion_matrix, create_diffusion_profiles


def colinearity_coefficient(x, y):
    return np.dot(x, y) / np.sqrt((x**2).sum() * (y**2).sum())


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
    order_init = np.argsort(diags)
    order_res = np.argsort(diags_res)
    assert np.allclose(diags[order_init], diags_res[order_res], rtol=0.06)
    assert abs(colinearity_coefficient(eigvecs[:-1, order_res][:, 0],
                          np.ravel(P[:, 0]))) > 0.95 
    assert abs(colinearity_coefficient(eigvecs[:-1, order_res][:, 1],
                         np.ravel(P[:, 1]))) > 0.95


def test_eigvals_only():
    x_prof = np.linspace(-20, 20, 100)
    x_points = [x_prof, x_prof, x_prof]
    diags = np.array([1, 5])
    P = np.matrix([[1, 1], [-1, 0]])
    dc = np.array([[1, 1, 0],
                   [-1, 0, 1],
                   [0, -1, -1]])
    concentration_profiles = create_diffusion_profiles((diags, P),
                                            x_points, dc, noise_level=0.06)
    diags_init = np.array([1, 1])
    P_estimate = np.matrix([[1, 1], [-1.1, 0.1]])
    diags_res, eigvecs, _, _, _ = compute_diffusion_matrix((diags_init,
                                                            P),
                                           x_points, concentration_profiles,
                                           plot=False, eigvals_only=True)
    #assert np.allclose(np.sort(diags), np.sort(diags_res), rtol=1.e-5)
    print(diags_res)

def test_multi_temp():
    n_comps = 2
    diags_T1 = np.array([1, 2])
    diags_T2 = np.array([2, 10])
    P = np.matrix([[1, 1], [-1, 0]])
    xpoints_exp = np.linspace(-15, 15, 100)
    x_points = [xpoints_exp] * 3
    exchange_vectors = np.array([[0, 1, 1], 
                                [1, -1, 0],
                                [-1, 0, -1]])
    concentration_profiles_T1 = create_diffusion_profiles((diags_T1, P),
                                                        x_points,
                                                        exchange_vectors,
                                                        noise_level=0.02)
    concentration_profiles_T2 = create_diffusion_profiles((diags_T2, P), 
                                                        x_points,
                                                        exchange_vectors,
                                                        noise_level=0.02)
    concentration_profiles = [concentration_profiles_T1,
                              concentration_profiles_T2]
    xpoints_all = [x_points, x_points]
    diags_init = np.array([1, 1, 1, 1])
    P_init = np.eye(2)
    diags_res, eigvecs, _ = compute_diffusion_matrix((diags_init, P_init), 
                               xpoints_all,
                               concentration_profiles, plot=False,
                               labels=['1', '2', '3'], nb_temp=2)
    assert abs(diags_res.max() - 10) < 0.1
