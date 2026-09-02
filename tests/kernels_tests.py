"""Direct tests of the numba visibility-prediction kernels against a naive DFT.

The kernels are the numerical core of skysim but are otherwise only exercised
end-to-end; these tests pin their maths to an independent numpy reference.
"""

import numpy as np
import pytest

from simms.constants import C
from simms.skymodel.kernels import NO_SMEAR_UVW, RENORM_INTERVAL, is_uniform_grid, predict_vis

# The three trailing kernel arguments that switch time/bandwidth smearing off; the
# smeared kernel is covered in smearing_tests.py.
NO_SMEAR = (False, 0.0, NO_SMEAR_UVW)

RNG = np.random.default_rng(42)


def reference_predict(uvw, freqs, lmn, gauss_shape, is_gauss, bmat, lightcurve, time_index):
    """Naive per-row, per-source DFT implementing the documented kernel maths."""
    nrow = uvw.shape[0]
    nchan = freqs.size
    nspec = bmat.shape[1]
    vis = np.zeros((nrow, nchan, nspec), dtype=np.complex128)
    for r in range(nrow):
        u, v, _ = uvw[r]
        for s in range(lmn.shape[0]):
            amp = lightcurve[s, time_index[r]]
            base = (uvw[r] @ lmn[s]) * 2.0 * np.pi / C
            phasor = amp * np.exp(1j * base * freqs)
            if is_gauss[s]:
                ell, emm, ecc = gauss_shape[s]
                fu1 = (u * emm - v * ell) * ecc
                fv1 = u * ell + v * emm
                phasor = phasor * np.exp(-(fu1**2 + fv1**2) * (freqs / C) ** 2)
            vis[r] += bmat[s].T * phasor[:, None]
    return vis


def make_inputs(nrow=6, nchan=8, nsrc=3, nspec=4, ntime=1, gauss=False, uniform=True):
    uvw = RNG.uniform(-2e3, 2e3, size=(nrow, 3))
    if uniform:
        freqs = np.linspace(1.0e9, 1.5e9, nchan)
    else:
        freqs = np.sort(RNG.uniform(1.0e9, 1.5e9, size=nchan))
    # small direction cosines; third component is n - 1
    lm = RNG.uniform(-0.01, 0.01, size=(nsrc, 2))
    n_minus_1 = np.sqrt(1.0 - (lm**2).sum(axis=1)) - 1.0
    lmn = np.column_stack([lm, n_minus_1])
    gauss_shape = RNG.uniform(1e-5, 1e-4, size=(nsrc, 3)) if gauss else np.zeros((nsrc, 3))
    is_gauss = np.full(nsrc, gauss)
    bmat = RNG.normal(size=(nsrc, nspec, nchan)) + 1j * RNG.normal(size=(nsrc, nspec, nchan))
    lightcurve = np.ones((nsrc, ntime)) if ntime == 1 else RNG.uniform(0.2, 1.0, size=(nsrc, ntime))
    time_index = np.zeros(nrow, dtype=np.int64) if ntime == 1 else RNG.integers(0, ntime, size=nrow)
    return uvw, freqs, lmn, gauss_shape, is_gauss, bmat, lightcurve, time_index


def run_kernel(inputs, uniform):
    uvw, freqs, lmn, gauss_shape, is_gauss, bmat, lightcurve, time_index = inputs
    vis = np.zeros((uvw.shape[0], freqs.size, bmat.shape[1]), dtype=np.complex128)
    predict_vis(uvw, freqs, uniform, lmn, gauss_shape, is_gauss, bmat, lightcurve, time_index, vis, *NO_SMEAR)
    return vis


@pytest.mark.parametrize("uniform", [True, False])
def test_predict_vis_points_match_reference(uniform):
    inputs = make_inputs(uniform=uniform)
    vis = run_kernel(inputs, uniform=uniform)
    np.testing.assert_allclose(vis, reference_predict(*inputs), rtol=1e-9, atol=1e-12)


def test_predict_vis_uniform_recurrence_across_renorm():
    # The uniform-grid path replaces per-channel sincos with a phasor
    # recurrence that is renormalised every RENORM_INTERVAL channels; span
    # several intervals so drift and the renorm step are both exercised.
    inputs = make_inputs(nchan=3 * RENORM_INTERVAL + 7)
    vis = run_kernel(inputs, uniform=True)
    np.testing.assert_allclose(vis, reference_predict(*inputs), rtol=1e-8, atol=1e-10)


def test_predict_vis_gaussian_matches_reference():
    inputs = make_inputs(gauss=True)
    vis = run_kernel(inputs, uniform=True)
    np.testing.assert_allclose(vis, reference_predict(*inputs), rtol=1e-9, atol=1e-12)


def test_predict_vis_mixed_point_and_gaussian():
    inputs = list(make_inputs(nsrc=4, gauss=True))
    is_gauss = inputs[4]
    is_gauss[:2] = False
    vis = run_kernel(tuple(inputs), uniform=True)
    np.testing.assert_allclose(vis, reference_predict(*inputs), rtol=1e-9, atol=1e-12)


def test_predict_vis_lightcurve_scaling():
    inputs = make_inputs(ntime=5)
    vis = run_kernel(inputs, uniform=True)
    np.testing.assert_allclose(vis, reference_predict(*inputs), rtol=1e-9, atol=1e-12)


def test_predict_vis_accumulates_into_buffer():
    inputs = make_inputs()
    uvw, freqs, lmn, gauss_shape, is_gauss, bmat, lightcurve, time_index = inputs
    vis = np.zeros((uvw.shape[0], freqs.size, bmat.shape[1]), dtype=np.complex128)
    predict_vis(uvw, freqs, True, lmn, gauss_shape, is_gauss, bmat, lightcurve, time_index, vis, *NO_SMEAR)
    once = vis.copy()
    predict_vis(uvw, freqs, True, lmn, gauss_shape, is_gauss, bmat, lightcurve, time_index, vis, *NO_SMEAR)
    np.testing.assert_allclose(vis, 2 * once, rtol=1e-12)


def test_is_uniform_grid():
    assert is_uniform_grid(np.linspace(1e9, 2e9, 64))
    assert is_uniform_grid(np.array([1e9]))
    assert is_uniform_grid(np.array([1e9, 1.1e9]))
    jittered = np.linspace(1e9, 2e9, 64)
    jittered[10] += 1e5
    assert not is_uniform_grid(jittered)
    # descending but uniform spacing is still a uniform grid
    assert is_uniform_grid(np.linspace(2e9, 1e9, 64))
