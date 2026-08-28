"""Tests for visibility prediction from ASCII sky models.

The reference in these tests is a direct transcription of the RIME, evaluated
with an explicit triple loop, so it shares no code with the prediction kernel.
"""

import numpy as np
import pytest

from simms import SCHEMADIR
from simms.constants import C
from simms.skymodel.ascii_skies import ASCIISkymodel
from simms.skymodel.kernels import is_uniform_grid
from simms.skymodel.mstools import compute_vis, predict_block, prepare_skymodel
from simms.skymodel.source_factory import contspec

from . import InitTest

SCHEMA = f"{SCHEMADIR}/source_schema.yaml"
RA0, DEC0 = np.deg2rad(15.0), np.deg2rad(-31.0)

UNIFORM_FREQS = 1.4e9 + np.arange(4) * 1e7
IRREGULAR_FREQS = np.array([1.40e9, 1.41e9, 1.43e9, 1.48e9])


class InitThisTest(InitTest):
    def write_temp_file(self, content: str, suffix: str = ".txt") -> str:
        path = self.random_named_file(suffix=suffix)
        with open(path, "w") as fh:
            fh.write(content)
        return path


@pytest.fixture
def params():
    return InitThisTest()


@pytest.fixture
def uvw():
    return np.random.default_rng(11).normal(0, 1500, (6, 3))


def brute_force_vis(sources, uvw, freqs, ncorr, polarisation, linear_basis):
    """Textbook RIME reference, sharing no code with the prediction kernel."""
    vis = np.zeros((uvw.shape[0], freqs.size, ncorr), np.complex128)
    for src in sources:
        dra = src.ra - RA0
        el = np.cos(src.dec) * np.sin(dra)
        em = np.sin(src.dec) * np.cos(DEC0) - np.cos(src.dec) * np.sin(DEC0) * np.cos(dra)
        en = np.sqrt(1 - el * el - em * em) - 1

        stokes = [getattr(src, f"stokes_{x}", 0.0) for x in "iquv"]
        # In the circular basis the ascii stokes_q/u/v columns are read as V/Q/U.
        if linear_basis:
            i_, q_, u_, v_ = stokes
        else:
            i_, v_, q_, u_ = stokes

        # emaj, emin are FWHM angles; the visibility of an elliptical Gaussian is
        # the Fourier transform of the image-plane Gaussian, evaluated from its
        # sigma_maj, sigma_min directly (no kernel-specific parameterisation).
        emaj = src.value_or_default("emaj")
        emin = src.value_or_default("emin")
        pa = src.value_or_default("pa")
        sigma_maj = emaj / (2 * np.sqrt(2 * np.log(2)))
        sigma_min = emin / (2 * np.sqrt(2 * np.log(2)))

        if linear_basis:
            corrs = [i_ + q_, u_ + 1j * v_, u_ - 1j * v_, i_ - q_]
        else:
            corrs = [i_ + v_, q_ + 1j * u_, q_ - 1j * u_, i_ - v_]
        if not polarisation:
            corrs = [corrs[0], 0, 0, corrs[0]]
        selection = [0, 3] if ncorr == 2 else [0, 1, 2, 3]

        for row in range(uvw.shape[0]):
            u, v, w = uvw[row]
            for chan in range(freqs.size):
                scale = freqs[chan] / C
                phasor = np.exp(2j * np.pi * (u * el + v * em + w * en) * scale)
                if emaj or emin:
                    u_lambda, v_lambda = u * scale, v * scale
                    # rotate uv into the source frame (major axis along position angle pa)
                    u_maj = u_lambda * np.sin(pa) + v_lambda * np.cos(pa)
                    u_min = u_lambda * np.cos(pa) - v_lambda * np.sin(pa)
                    phasor *= np.exp(-2 * np.pi**2 * (sigma_maj**2 * u_maj**2 + sigma_min**2 * u_min**2))
                for corr, idx in enumerate(selection):
                    vis[row, chan, corr] += corrs[idx] * phasor
    return vis


POINTS = "\n".join(
    [
        "#format: name ra dec stokes_i emaj emin pa",
        "s0 15.06deg -31.20deg 1.25 0 0 0",
        "s1 14.88deg -30.91deg 0.61 0 0 0",
    ]
)
GAUSSIANS = "\n".join(
    [
        "#format: name ra dec stokes_i emaj emin pa",
        "g0 15.06deg -31.20deg 1.25 12arcsec 4arcsec 30deg",
        "g1 14.88deg -30.91deg 0.61 20arcsec 9arcsec 110deg",
    ]
)
POLARISED = "\n".join(
    [
        "#format: name ra dec stokes_i stokes_q stokes_u stokes_v emaj emin pa",
        "p0 15.06deg -31.20deg 1.25 0.09 -0.11 0.03 0 0 0",
        "p1 14.88deg -30.91deg 0.61 -0.04 0.07 -0.02 12arcsec 4arcsec 30deg",
    ]
)


@pytest.mark.parametrize("model", [POINTS, GAUSSIANS, POLARISED], ids=["point", "gaussian", "polarised"])
@pytest.mark.parametrize("freqs", [UNIFORM_FREQS, IRREGULAR_FREQS], ids=["uniform", "irregular"])
@pytest.mark.parametrize("ncorr", [2, 4])
@pytest.mark.parametrize("polarisation", [False, True])
@pytest.mark.parametrize("linear_basis", [True, False])
def test_predict_matches_brute_force_rime(params, uvw, model, freqs, ncorr, polarisation, linear_basis):
    sky = ASCIISkymodel(params.write_temp_file(model), source_schema_file=SCHEMA)
    got = compute_vis(
        sky, uvw, freqs, ncorr=ncorr, polarisation=polarisation, linear_basis=linear_basis, ra0=RA0, dec0=DEC0
    )
    expected = brute_force_vis(sky.sources, uvw, freqs, ncorr, polarisation, linear_basis)
    np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-13)


def test_unpolarised_four_correlations_do_not_accumulate_into_yy(params, uvw):
    """Regression: YY used to be summed cumulatively, so source k was added n-k+1 times."""
    sky = ASCIISkymodel(params.write_temp_file(POINTS), source_schema_file=SCHEMA)
    vis = compute_vis(sky, uvw, UNIFORM_FREQS, ncorr=4, polarisation=False, ra0=RA0, dec0=DEC0)

    np.testing.assert_array_equal(vis[..., 0], vis[..., 3])
    assert not np.any(vis[..., 1])
    assert not np.any(vis[..., 2])


def test_point_source_without_shape_columns(params, uvw):
    """Regression: a catalogue with no emaj/emin columns raised AttributeError."""
    model = "\n".join(["#format: name ra dec stokes_i", "s0 15.06deg -31.20deg 1.25"])
    sky = ASCIISkymodel(params.write_temp_file(model), source_schema_file=SCHEMA)
    vis = compute_vis(sky, uvw, UNIFORM_FREQS, ncorr=2, ra0=RA0, dec0=DEC0)
    assert np.all(np.isfinite(vis))
    assert np.abs(vis).max() == pytest.approx(1.25, rel=1e-6)


def test_noise_seed_reaches_the_predict_entry_points(params, uvw):
    """`noise_vis` on the predict API drew from an unseeded RNG, so a caller had no way to
    reproduce a run. Check the seed reaches the noise through the real call chain, and that
    omitting it still gives a fresh draw each time.
    """
    sky = ASCIISkymodel(params.write_temp_file(POINTS), source_schema_file=SCHEMA)
    common = dict(ncorr=2, ra0=RA0, dec0=DEC0, noise_vis=0.1)

    seeded = compute_vis(sky, uvw, UNIFORM_FREQS, seed=4, **common)
    same = compute_vis(sky, uvw, UNIFORM_FREQS, seed=4, **common)
    other = compute_vis(sky, uvw, UNIFORM_FREQS, seed=5, **common)
    unseeded = [compute_vis(sky, uvw, UNIFORM_FREQS, **common) for _ in range(2)]

    assert np.array_equal(seeded, same)
    assert not np.array_equal(seeded, other)
    assert not np.array_equal(*unseeded)

    # Noise is the only difference: the noiseless prediction is unchanged by the seed.
    clean = compute_vis(sky, uvw, UNIFORM_FREQS, ncorr=2, ra0=RA0, dec0=DEC0)
    assert np.abs(seeded - clean).max() < 1.0
    assert not np.array_equal(seeded, clean)


def test_predict_block_takes_a_seed(params, uvw):
    sky = ASCIISkymodel(params.write_temp_file(POINTS), source_schema_file=SCHEMA)
    prepared = prepare_skymodel(sky, UNIFORM_FREQS, RA0, DEC0, ncorr=2)
    a = predict_block(prepared, uvw, noise_vis=0.1, seed=9)
    b = predict_block(prepared, uvw, noise_vis=0.1, seed=9)
    c = predict_block(prepared, uvw, noise_vis=0.1, seed=10)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_transient_lightcurve_is_independent_of_row_blocking(params):
    """Regression: the lightcurve was referenced to each row block's own first time."""
    model = "\n".join(
        [
            "#format: name ra dec stokes_i transient_start transient_period transient_ingress transient_absorb",
            "t0 15.06deg -31.20deg 1.25 100s 200 40 0.02",
        ]
    )
    sky = ASCIISkymodel(params.write_temp_file(model), source_schema_file=SCHEMA)
    assert sky.has_transient

    rng = np.random.default_rng(3)
    uvw = rng.normal(0, 1500, (80, 3))
    # 40 time slots of 8s spans 312s, so the 100s-300s transit falls inside it
    times = np.repeat(5.0e9 + np.arange(40) * 8.0, 2)
    unique_times = np.unique(times)

    prepared = prepare_skymodel(sky, UNIFORM_FREQS, RA0, DEC0, ncorr=2, unique_times=unique_times)
    whole = predict_block(prepared, uvw, times=times)
    blocked = np.concatenate(
        [predict_block(prepared, uvw[i : i + 20], times=times[i : i + 20]) for i in range(0, 80, 20)]
    )

    np.testing.assert_array_equal(whole, blocked)
    # the transit actually dips by transient_absorb
    assert prepared.lightcurve[0].min() == pytest.approx(0.98, abs=1e-6)
    assert prepared.lightcurve[0].max() == pytest.approx(1.0, abs=1e-6)


def test_mixed_catalogue_modulates_only_the_transient_source(params):
    """A catalogue may hold a line, a transient and a plain source at once; only the
    transient one gets a lightcurve, and the line one keeps its Gaussian spectrum."""
    model = "\n".join(
        [
            "#format: name ra dec stokes_i line_peak line_width "
            "transient_start transient_period transient_ingress transient_absorb",
            "line 15.06deg -31.20deg 2.0 1.41e9 2e7 null null null null",
            "trans 15.02deg -31.05deg 1.25 null null 100s 200 40 0.02",
            "plain 14.95deg -30.90deg 3.0 null null null null null null",
        ]
    )
    sky = ASCIISkymodel(params.write_temp_file(model), source_schema_file=SCHEMA)
    assert [(s.is_line, s.is_transient) for s in sky.sources] == [(True, False), (False, True), (False, False)]

    unique_times = 5.0e9 + np.arange(40) * 8.0
    prepared = prepare_skymodel(sky, UNIFORM_FREQS, RA0, DEC0, ncorr=2, unique_times=unique_times)

    # only the transient source varies in time
    assert prepared.lightcurve[1].min() == pytest.approx(0.98, abs=1e-6)
    assert np.array_equal(prepared.lightcurve[[0, 2]], np.ones((2, unique_times.size)))

    # the line source's Stokes I follows the Gaussian, peaking at the channel nearest 1.41 GHz
    stokes_i = prepared.bmat[0, 0].real
    assert np.argmax(stokes_i) == int(np.argmin(np.abs(UNIFORM_FREQS - 1.41e9)))
    assert stokes_i.max() == pytest.approx(2.0)  # UNIFORM_FREQS[1] sits on the line centre
    assert stokes_i[0] < 2.0 and stokes_i[-1] < stokes_i[0]  # and falls off either side
    # the plain source is flat at its catalogue flux
    assert np.allclose(prepared.bmat[2, 0].real, 3.0)


def test_transient_requires_unique_times(params):
    model = "\n".join(
        [
            "#format: name ra dec stokes_i transient_start transient_period transient_ingress transient_absorb",
            "t0 15.06deg -31.20deg 1.25 100s 200 40 0.02",
        ]
    )
    sky = ASCIISkymodel(params.write_temp_file(model), source_schema_file=SCHEMA)
    with pytest.raises(ValueError, match="unique_times"):
        prepare_skymodel(sky, UNIFORM_FREQS, RA0, DEC0, ncorr=2)


def test_frequency_recurrence_matches_per_channel_sincos(params):
    """The uniform-grid rotation recurrence must agree with an explicit sincos."""
    sky = ASCIISkymodel(params.write_temp_file(POINTS), source_schema_file=SCHEMA)
    uvw = np.random.default_rng(5).normal(0, 8000, (16, 3))
    freqs = 1.4e9 + np.arange(4096) * 1e5

    recurrence = prepare_skymodel(sky, freqs, RA0, DEC0, ncorr=2)
    assert recurrence.uniform_freqs

    per_channel = prepare_skymodel(sky, freqs, RA0, DEC0, ncorr=2)
    per_channel.uniform_freqs = False

    np.testing.assert_allclose(predict_block(recurrence, uvw), predict_block(per_channel, uvw), rtol=1e-10)


def test_is_uniform_grid():
    assert is_uniform_grid(np.array([1.4e9]))
    assert is_uniform_grid(np.array([1.4e9, 1.5e9]))
    assert is_uniform_grid(1.4e9 + np.arange(64) * 1e6)
    assert not is_uniform_grid(np.array([1.0, 2.0, 3.0, 9.0]))


def test_invalid_ncorr(params, uvw):
    sky = ASCIISkymodel(params.write_temp_file(POINTS), source_schema_file=SCHEMA)
    with pytest.raises(ValueError, match="two or four correlations"):
        compute_vis(sky, uvw, UNIFORM_FREQS, ncorr=3, ra0=RA0, dec0=DEC0)


@pytest.mark.parametrize("coeff", [[-0.7], [-0.7, 0.05], [-0.7, 0.05, -0.01]])
def test_contspec_log_polynomial(coeff):
    """Regression: a multi-term coefficient list raised TypeError (array ** Polynomial)."""
    freqs = np.array([1.0e9, 1.4e9, 2.0e9])
    nu_ref, flux = 1.4e9, 2.5

    log_ratio = np.log(freqs / nu_ref)
    exponent = sum(c * log_ratio**i for i, c in enumerate(coeff))
    expected = flux * (freqs / nu_ref) ** exponent

    np.testing.assert_allclose(contspec(freqs, flux, coeff, nu_ref), expected, rtol=1e-12)


def test_contspec_without_reference_frequency_is_flat():
    freqs = np.array([1.0e9, 1.4e9, 2.0e9])
    np.testing.assert_allclose(contspec(freqs, 2.5, [-0.7], None), 2.5)


def test_continuum_source_predicts_a_sloped_spectrum(params, uvw):
    model = "\n".join(
        [
            "#format: name ra dec stokes_i cont_reffreq cont_coeff_1",
            "c0 15.0deg -31.0deg 1.0 1.4GHz -0.7",
        ]
    )
    sky = ASCIISkymodel(params.write_temp_file(model), source_schema_file=SCHEMA)
    # place the source at the phase centre so the phasor is unity and |vis| is the spectrum
    vis = compute_vis(sky, np.zeros((1, 3)), UNIFORM_FREQS, ncorr=2, ra0=RA0, dec0=DEC0)
    expected = 1.0 * (UNIFORM_FREQS / 1.4e9) ** -0.7
    np.testing.assert_allclose(vis[0, :, 0].real, expected, rtol=1e-6)


def test_gaussian_emaj_is_the_fwhm(params, uvw):
    """A circular ASCII Gaussian's |vis| must be the analytic FT of its FWHM."""
    fwhm_arcsec = 30.0
    model = "\n".join(
        [
            "#format: name ra dec stokes_i emaj emin pa",
            f"g0 15.0deg -31.0deg 1.0 {fwhm_arcsec}arcsec {fwhm_arcsec}arcsec 0deg",
        ]
    )
    sky = ASCIISkymodel(params.write_temp_file(model), source_schema_file=SCHEMA)
    freqs = np.array([1.4e9])
    probe = np.array([[400.0, 0.0, 0.0], [0.0, 700.0, 0.0]])
    vis = compute_vis(sky, probe, freqs, ncorr=2, ra0=RA0, dec0=DEC0)[:, 0, 0]

    sigma = np.deg2rad(fwhm_arcsec / 3600) / (2 * np.sqrt(2 * np.log(2)))
    u_lambda = probe[:, :2] * freqs[0] / C
    expected = np.exp(-2 * np.pi**2 * sigma**2 * (u_lambda**2).sum(axis=1))
    np.testing.assert_allclose(np.abs(vis), expected, rtol=1e-6)


def test_channel_chunking_matches_full_prediction(params, uvw):
    """Predicting a subset of channels then concatenating equals the whole grid."""
    sky = ASCIISkymodel(params.write_temp_file(GAUSSIANS), source_schema_file=SCHEMA)
    freqs = 1.3e9 + np.arange(9) * 2e7
    prepared = prepare_skymodel(sky, freqs, RA0, DEC0, ncorr=2)

    whole = predict_block(prepared, uvw)
    chunked = np.concatenate(
        [predict_block(prepared.select_channels(np.arange(a, a + 3)), uvw) for a in range(0, 9, 3)],
        axis=1,
    )
    np.testing.assert_array_equal(whole, chunked)


def test_select_channels_preserves_uniform_flag(params):
    sky = ASCIISkymodel(params.write_temp_file(POINTS), source_schema_file=SCHEMA)
    prepared = prepare_skymodel(sky, UNIFORM_FREQS, RA0, DEC0, ncorr=2)
    assert prepared.uniform_freqs
    # a gappy channel selection is no longer uniform
    assert not prepared.select_channels(np.array([0, 1, 3])).uniform_freqs
