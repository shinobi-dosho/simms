"""Antenna pointing errors: the ``pointing:`` corruption block, applied through the beam kernels."""

from __future__ import annotations

import logging
import pickle
from dataclasses import replace

import numpy as np
import pytest
from daskms import xds_from_ms, xds_from_table

from simms.apps import skysim
from simms.skymodel.beams import (
    CosineTaperBeam,
    FitsBeamProvider,
    JimBeamProvider,
    _beam_grid_gib,
    build_beam_derivative_grids,
    build_beam_derivative_grids_jones,
    corr_basis_transform,
    pa_sample_grid,
)
from simms.skymodel.corruptions import (
    PointingModel,
    PointingSpec,
    build_pointing_model,
    load_corruption_spec,
    validate_spec,
)
from simms.skymodel.kernels import is_uniform_grid
from simms.skymodel.mstools import PreparedSky, attach_beam, predict_block
from simms.telescope.generate_ms import create_ms

from . import InitTest, skysim_opts

# MeerKAT reference site (geodetic, radians) and a modern epoch, as in beam_tests.
MKAT_LON = np.deg2rad(21.4439)
MKAT_LAT = np.deg2rad(-30.7130)
BASE_TIME = 59_000.0 * 86400.0 + 20 * 3600.0

ARCSEC = np.radians(1.0 / 3600.0)
SIGMA = 30.0 * ARCSEC
JIM = JimBeamProvider(CosineTaperBeam.from_builtin("MKAT-AA-L-JIM-2020"))

RADII_DEG = np.array([0.0, 0.2, 0.4, 0.55, 0.7, 0.9, 1.1, 1.3])
GRID_FREQS = np.array([0.9e9, 1.4e9])
GRID_CHI = np.array([0.0, 0.6])


# --------------------------------------------------------------------------- helpers


def _ring(radii_deg, angle_deg=30.0):
    """``(ell, emm)`` of sources at ``radii_deg`` along a fixed position angle."""
    r = np.radians(np.asarray(radii_deg, dtype=np.float64))
    a = np.radians(angle_deg)
    return r * np.cos(a), r * np.sin(a)


def _lmn(ell, emm):
    ell, emm = np.atleast_1d(ell), np.atleast_1d(emm)
    return np.stack([ell, emm, np.sqrt(1 - ell * ell - emm * emm) - 1], axis=-1)


def _prepared(lmn, freqs, ncorr, flux=1.0):
    """A full-correlation PreparedSky of unpolarised point sources."""
    nsrc, nchan = lmn.shape[0], freqs.size
    bmat = np.zeros((nsrc, ncorr, nchan), dtype=np.complex128)
    bmat[:, 0, :] = flux
    bmat[:, -1, :] = flux
    return PreparedSky(
        lmn=lmn,
        gauss_shape=np.zeros((nsrc, 3)),
        is_gauss=np.zeros(nsrc, dtype=bool),
        bmat=bmat,
        lightcurve=np.ones((nsrc, 1)),
        unique_times=None,
        freqs=freqs,
        uniform_freqs=is_uniform_grid(freqs),
        ncorr=ncorr,
        polarisation=True,
    )


def _attach(
    prepared,
    pointing,
    nant,
    providers=(JIM,),
    is_altaz=(False,),
    ant_type=None,
    full_jones=False,
    transform=None,
    duration=1.0,
    **kw,
):
    """attach_beam at the MeerKAT site, pointed at dec = latitude (rows at BASE_TIME)."""
    ant_type = np.zeros(nant, dtype=np.int64) if ant_type is None else ant_type
    if full_jones and transform is None:
        transform = corr_basis_transform(False)
    return attach_beam(
        prepared,
        ant_type,
        list(providers),
        np.asarray(is_altaz),
        0.0,
        MKAT_LAT,
        MKAT_LON,
        MKAT_LAT,
        BASE_TIME,
        duration,
        1.0,
        prepared.ncorr,
        full_jones=full_jones,
        basis_transform=transform,
        pointing=pointing,
        **kw,
    )


def _fixed_pointing(offsets, taylor="laplacian"):
    """A PointingModel with given static offsets and no drift (bypasses spec validation)."""
    offsets = np.ascontiguousarray(offsets, dtype=np.float64)
    return PointingModel(
        static=offsets,
        phase=np.zeros_like(offsets),
        amplitude=0.0,
        period=None,
        t0=BASE_TIME,
        static_rms=float(np.sqrt(np.mean(offsets**2))),
        taylor=taylor,
    )


def _predict(prepared, ant1, ant2, times=None, uvw=None):
    nrow = len(ant1)
    times = np.full(nrow, BASE_TIME) if times is None else times
    uvw = np.zeros((nrow, 3)) if uvw is None else uvw
    return predict_block(prepared, uvw, times=times, antenna1=np.asarray(ant1), antenna2=np.asarray(ant2))


def _ref_derivs(prov, ell, emm, freqs, chi, h=1e-4):
    """Independent float64 5-point-stencil ``(D_l, D_m, L)`` through ``prov.voltage``."""
    chi = np.atleast_1d(chi)

    def ev(o=None):
        return prov.voltage(ell, emm, freqs, chi, offset=o)

    pl, ml, pm, mm, c = ev((h, 0.0)), ev((-h, 0.0)), ev((0.0, h)), ev((0.0, -h)), ev()
    return (pl - ml) / (2 * h), (pm - mm) / (2 * h), (pl + ml + pm + mm - 4 * c) / h**2


class _Files(InitTest):
    def write_yaml(self, content: str) -> str:
        path = self.random_named_file(suffix=".yaml")
        with open(path, "w") as fh:
            fh.write(content)
        return path


@pytest.fixture
def files():
    return _Files()


# --------------------------------------------------------------------------- spec parsing

_GAINS = """
gains:
  terms: [G]
  spec:
    - {label: G, type: scalar, axes: [time], period: 120.0, amplitude: 0.1}
"""


def test_pointing_only_spec_loads(files):
    path = files.write_yaml("pointing:\n  static: 30arcsec\n  amplitude: 10arcsec\n  period: 20min\n")
    spec = load_corruption_spec(path)
    assert not spec.has_gains and spec.terms == [] and spec.spec == []
    p = spec.pointing
    assert p.taylor == "laplacian"
    assert p.static == pytest.approx(30 * ARCSEC, rel=1e-12)
    assert p.amplitude == pytest.approx(10 * ARCSEC, rel=1e-12)
    assert p.period == pytest.approx(1200.0)


def test_pointing_bare_period_is_seconds_and_bare_zero_is_allowed(files):
    path = files.write_yaml("pointing:\n  static: 0\n  amplitude: 1arcmin\n  period: 600\n  taylor: first\n")
    p = load_corruption_spec(path).pointing
    assert p.static == 0.0 and p.period == 600.0 and p.taylor == "first"
    assert p.amplitude == pytest.approx(60 * ARCSEC)


def test_spec_with_both_blocks_loads(files):
    path = files.write_yaml("pointing:\n  static: 30arcsec\n" + _GAINS)
    spec = load_corruption_spec(path)
    assert spec.has_gains and spec.terms == ["G"] and spec.pointing.static == pytest.approx(SIGMA)
    validate_spec(spec, ncorr=2)


def test_spec_with_neither_block_is_an_error(files):
    path = files.write_yaml("corruptions:\n  static: 30arcsec\n")
    with pytest.raises(RuntimeError, match="no top-level 'gains' block"):
        load_corruption_spec(path)


@pytest.mark.parametrize(
    "block, match",
    [
        ("pointing:\n  static: 30\n", "explicit angle unit"),
        ("pointing:\n  static: 30.5\n", "explicit angle unit"),
        ("pointing:\n  static: 30s\n", "not an angle"),
        ("pointing:\n  static: '30'\n", "not an angle"),
        ("pointing:\n  static: true\n", "must be an angle"),
        ("pointing:\n  static: -30arcsec\n", "non-negative"),
        ("pointing:\n  amplitude: 10arcsec\n", "period' is required"),
        ("pointing:\n  amplitude: 10arcsec\n  period: -5\n", "positive"),
        ("pointing:\n  static: 0\n  amplitude: 0arcsec\n", "unchanged"),
        ("pointing: {}\n", "unchanged"),
        ("pointing:\n  static: 30arcsec\n  sttic: 1arcsec\n", "unknown key.*'pointing'"),
        ("pointing:\n", "mapping"),
        ("pointing: [30arcsec]\n", "mapping"),
        ("pointing:\n  static: 30arcsec\n  taylor: second\n", "first', 'laplacian"),
        ("pointing:\n  static: 30arcsec\n  taylor: 2\n", "first', 'laplacian"),
        ("pointing:\n  static: 30arcsec\n  amplitude: 1arcsec\n  period: 2MHz\n", "pointing.period"),
    ],
)
def test_pointing_block_errors(files, block, match):
    with pytest.raises(RuntimeError, match=match):
        load_corruption_spec(files.write_yaml(block))


def test_empty_gains_terms_still_fail_alongside_pointing(files):
    spec = load_corruption_spec(files.write_yaml("pointing:\n  static: 30arcsec\ngains:\n  terms: []\n  spec: []\n"))
    assert spec.has_gains
    with pytest.raises(RuntimeError, match="lists no terms"):
        validate_spec(spec, ncorr=2)


def test_gains_term_labelled_pointing_collides_with_the_pointing_draw(files):
    term = "    - {label: pointing, type: scalar, axes: [time], period: 120.0, amplitude: 0.1}\n"
    gains = "gains:\n  terms: [pointing]\n  spec:\n" + term
    with pytest.raises(RuntimeError, match="same random stream"):
        load_corruption_spec(files.write_yaml("pointing:\n  static: 30arcsec\n" + gains))
    # Without a pointing block the label is just a label.
    assert load_corruption_spec(files.write_yaml(gains)).terms == ["pointing"]


def test_null_gains_beside_pointing_is_an_error(files):
    with pytest.raises(RuntimeError, match="'gains' must be a mapping"):
        load_corruption_spec(files.write_yaml("pointing:\n  static: 30arcsec\ngains:\n"))


def test_unknown_top_level_key_warns(files, caplog):
    path = files.write_yaml("pointing:\n  static: 30arcsec\nextra: 1\n")
    with caplog.at_level(logging.WARNING, logger="skysim"):
        load_corruption_spec(path)
    assert any("extra" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- the offset model


def test_offsets_are_seeded():
    spec = PointingSpec(static=SIGMA, amplitude=5 * ARCSEC, period=600.0)
    times = BASE_TIME + np.arange(6) * 100.0
    ant = np.arange(6) % 3
    a = build_pointing_model(spec, 3, BASE_TIME, random_seed=11).offsets(times, ant)
    b = build_pointing_model(spec, 3, BASE_TIME, random_seed=11).offsets(times, ant)
    c = build_pointing_model(spec, 3, BASE_TIME, random_seed=12).offsets(times, ant)
    d1 = build_pointing_model(spec, 3, BASE_TIME).offsets(times, ant)
    d2 = build_pointing_model(spec, 3, BASE_TIME).offsets(times, ant)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(d1, d2)
    assert not np.allclose(a, c)
    assert a.shape == (6, 2) and a.dtype == np.float64 and a.flags["C_CONTIGUOUS"]


def test_offset_statistics():
    model = build_pointing_model(PointingSpec(static=SIGMA, amplitude=ARCSEC, period=60.0), 20000, 0.0, random_seed=1)
    assert model.static.std() == pytest.approx(SIGMA, rel=0.02)
    assert abs(np.cos(model.phase).mean()) < 0.02
    assert model.phase.min() >= 0.0 and model.phase.max() < 2 * np.pi
    assert model.sigma_eff == pytest.approx(np.sqrt(SIGMA**2 + 0.5 * ARCSEC**2))
    assert model.nant == 20000


def test_drift_formula_and_period():
    amp, period, t0 = 10 * ARCSEC, 1200.0, BASE_TIME
    model = build_pointing_model(PointingSpec(static=SIGMA, amplitude=amp, period=period), 4, t0, random_seed=5)
    times = t0 + np.array([0.0, 37.0, 500.0, 3601.5, 7000.0])
    ant = np.array([0, 1, 2, 3, 1])
    expected = model.static[ant] + amp * np.cos(2 * np.pi * (times - t0)[:, None] / period + model.phase[ant])
    np.testing.assert_allclose(model.offsets(times, ant), expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(model.offsets(times + period, ant), expected, rtol=0, atol=1e-15)


def test_static_only_and_drift_does_not_change_static_draw():
    still = build_pointing_model(PointingSpec(static=SIGMA), 5, BASE_TIME, random_seed=3)
    assert still.period is None and still.amplitude == 0.0
    ant = np.array([4, 0, 2])
    np.testing.assert_array_equal(still.offsets(np.full(3, BASE_TIME + 99.0), ant), still.static[ant])
    drift = build_pointing_model(PointingSpec(static=SIGMA, amplitude=ARCSEC, period=60.0), 5, BASE_TIME, random_seed=3)
    np.testing.assert_array_equal(still.static, drift.static)
    np.testing.assert_array_equal(still.phase, drift.phase)


def test_pointing_model_and_pointed_sky_pickle():
    model = build_pointing_model(PointingSpec(static=SIGMA, amplitude=ARCSEC, period=60.0), 3, BASE_TIME)
    back = pickle.loads(pickle.dumps(model))
    times, ant = BASE_TIME + np.arange(3.0), np.arange(3)
    np.testing.assert_array_equal(back.offsets(times, ant), model.offsets(times, ant))

    sky = _attach(_prepared(_lmn(*_ring([0.5])), np.array([1.4e9]), 2), model, 3)
    sky2 = pickle.loads(pickle.dumps(sky))
    assert sky2.pointing_order == 2
    np.testing.assert_array_equal(sky2.beam_lap, sky.beam_lap)
    np.testing.assert_array_equal(_predict(sky2, [0, 1], [1, 2]), _predict(sky, [0, 1], [1, 2]))


# --------------------------------------------------------------------------- derivative grids


def test_derivative_grids_match_reference_stencil():
    ell, emm = _ring(RADII_DEG)
    d_l, d_m, lap = build_beam_derivative_grids([JIM], [True], ell, emm, GRID_FREQS, GRID_CHI, laplacian=True)
    assert d_l.dtype == np.complex64 and lap.shape == (1, 2, ell.size, 2, 2)
    ref_l, ref_m, ref_lap = _ref_derivs(JIM, ell, emm, GRID_FREQS, GRID_CHI)
    # An offset moves the beam, not the source: D = dE/d(offset) is computed here the
    # same way, so the comparison checks the step and the bookkeeping, not the sign.
    np.testing.assert_allclose(d_l[0], ref_l, rtol=0, atol=1e-3 * np.abs(ref_l).max())
    np.testing.assert_allclose(d_m[0], ref_m, rtol=0, atol=1e-3 * np.abs(ref_m).max())
    np.testing.assert_allclose(lap[0], ref_lap, rtol=0, atol=1e-3 * np.abs(ref_lap).max())
    # The flanks beyond ~0.95 deg at 1.4 GHz curve upward: that is where the mean brightens.
    assert (ref_lap[0, RADII_DEG >= 1.1, 1, 0].real > 0).all()
    assert (ref_lap[0, RADII_DEG <= 0.7, 1, 0].real < 0).all()
    first = build_beam_derivative_grids([JIM], [True], ell, emm, GRID_FREQS, GRID_CHI, laplacian=False)
    assert first[2] is None
    np.testing.assert_array_equal(first[0], d_l)


def test_jones_derivative_grids_fold_the_basis_transform():
    ell, emm = _ring(RADII_DEG)
    diag = build_beam_derivative_grids([JIM], [True], ell, emm, GRID_FREQS, GRID_CHI, laplacian=True)
    lin = build_beam_derivative_grids_jones(
        [JIM], [True], ell, emm, GRID_FREQS, GRID_CHI, corr_basis_transform(False), laplacian=True
    )
    S = corr_basis_transform(True)
    circ = build_beam_derivative_grids_jones([JIM], [True], ell, emm, GRID_FREQS, GRID_CHI, S, laplacian=True)
    for d, j_lin, j_circ in zip(diag, lin, circ, strict=True):
        assert j_lin.shape == (*d.shape[:-1], 2, 2)
        np.testing.assert_array_equal(j_lin[..., 0, 0], d[..., 0])
        np.testing.assert_array_equal(j_lin[..., 1, 1], d[..., 1])
        assert not j_lin[..., 0, 1].any() and not j_lin[..., 1, 0].any()
        expected = np.einsum("ij,...jk->...ik", S, j_lin.astype(np.complex128))
        np.testing.assert_allclose(j_circ, expected, rtol=0, atol=1e-6 * np.abs(expected).max())


# --------------------------------------------------------------------------- mean loss


def _antithetic():
    s = SIGMA * np.sqrt(2.0)
    return np.array([[s, 0.0], [-s, 0.0], [0.0, s], [0.0, -s]])


@pytest.mark.parametrize("taylor", ["laplacian", "first"])
def test_model_mean_over_antithetic_offsets(taylor):
    """Kernel-level: the 4-point mean of the model beam is E + sigma^2/2 L (laplacian) or E (first)."""
    freqs = np.array([1.4e9])
    offsets = np.vstack([_antithetic(), np.zeros((1, 2))])  # antenna 4 points nominally
    for radius in (0.0, 0.55, 1.1, 1.3):
        sky = _attach(_prepared(_lmn(*_ring([radius])), freqs, 2), _fixed_pointing(offsets, taylor), 5)
        vis = _predict(sky, [0, 1, 2, 3], [4, 4, 4, 4])
        e = sky.beam_grid[0, 0, 0, 0].astype(np.complex128)  # (2,) feeds
        if taylor == "laplacian":
            e_mean = e + 0.5 * SIGMA**2 * sky.beam_lap[0, 0, 0, 0]
        else:
            assert sky.beam_lap is None and sky.pointing_order == 1
            e_mean = e
        np.testing.assert_allclose(vis[:, 0, [0, 1]].mean(axis=0), e_mean * np.conj(e), rtol=0, atol=1e-7)


def test_exact_mean_loss_matches_half_sigma_squared_laplacian():
    """Provider-level: the exact 4-point mean minus E is sigma^2/2 L, including on the rising flanks."""
    ell, emm = _ring(RADII_DEG)
    e = JIM.voltage(ell, emm, GRID_FREQS, GRID_CHI)
    mean = np.mean([JIM.voltage(ell, emm, GRID_FREQS, GRID_CHI, offset=o) for o in _antithetic()], axis=0)
    pred = 0.5 * SIGMA**2 * _ref_derivs(JIM, ell, emm, GRID_FREQS, GRID_CHI)[2]
    np.testing.assert_allclose(mean - e, pred, rtol=0, atol=1e-2 * np.abs(pred).max())
    assert (pred[0, RADII_DEG >= 1.1, 1, 0].real > 0).all()  # flank brightening is predicted and real


@pytest.mark.parametrize("full_jones", [False, True])
def test_statistical_mean_loss_through_kernels(full_jones):
    """Many antennas, one on-axis source: mean(V) = |E|^2 + sigma^2 Re(E* L) in laplacian mode."""
    nant, freqs = 20000, np.array([1.4e9])
    ncorr = 4 if full_jones else 2
    ptg = build_pointing_model(PointingSpec(static=SIGMA), nant, BASE_TIME, random_seed=2)
    ant1, ant2 = np.arange(0, nant, 2), np.arange(1, nant, 2)
    e = JIM.voltage(np.zeros(1), np.zeros(1), freqs, [0.0])[0, 0, 0]  # (2,) feeds
    lap = _ref_derivs(JIM, np.zeros(1), np.zeros(1), freqs, [0.0])[2][0, 0, 0]
    loss = SIGMA**2 * np.real(np.conj(e) * lap)
    for taylor in ("laplacian", "first"):
        sky = _attach(_prepared(_lmn(0.0, 0.0), freqs, ncorr), replace(ptg, taylor=taylor), nant, full_jones=full_jones)
        mean = _predict(sky, ant1, ant2)[:, 0, [0, -1]].mean(axis=0)
        excess = mean.real - np.abs(e) ** 2
        if taylor == "laplacian":
            np.testing.assert_allclose(excess, loss, rtol=0.1)
        else:
            assert (np.abs(excess) < 0.1 * np.abs(loss)).all()


def test_per_realisation_accuracy():
    """Laplacian never loses to first in RMS, wins >10x on axis; first's error is quadratic."""
    ell, emm = _ring(RADII_DEG)
    freqs, chi = GRID_FREQS, np.zeros(1)
    e = JIM.voltage(ell, emm, freqs, chi)[0]
    d_l, d_m, lap = (g[0, 0] for g in build_beam_derivative_grids([JIM], [False], ell, emm, freqs, chi, True))
    draws = np.random.default_rng(1).standard_normal((4000, 2)) * SIGMA

    def rms_errors(scale):
        err_first, err_lap = [], []
        for dl, dm in draws * scale:
            exact = JIM.voltage(ell, emm, freqs, chi, offset=(dl, dm))[0]
            first = e + dl * d_l + dm * d_m
            err_first.append(exact - first)
            err_lap.append(exact - first - 0.25 * (dl * dl + dm * dm) * lap)
        rms = [np.sqrt(np.mean(np.abs(np.array(x)) ** 2, axis=0)).max(axis=(1, 2)) for x in (err_first, err_lap)]
        return rms  # each (nsrc,)

    first, lapl = rms_errors(1.0)
    assert (lapl <= 1.05 * first).all()
    assert lapl[0] < 0.1 * first[0]
    assert first.max() < 3e-4
    first_half, _ = rms_errors(0.5)
    ratio = first / first_half
    assert ((ratio >= 3) & (ratio <= 5)).all()


# --------------------------------------------------------------------------- the kernels


@pytest.mark.parametrize("full_jones", [False, True])
@pytest.mark.parametrize("taylor", ["laplacian", "first"])
def test_kernel_matches_exact_offset_beam(taylor, full_jones):
    """Rows on a PA sample: V == E_exact(p) conj(E_exact(q)), so D's sign and axes are right."""
    freqs = np.array([1.4e9])
    ncorr = 4 if full_jones else 2
    offsets = np.array([[30.0, -20.0], [-25.0, 30.0], [20.0, 25.0]]) * ARCSEC
    ell, emm = _ring([0.8], angle_deg=60.0)
    prepared = _prepared(_lmn(ell, emm), freqs, ncorr)
    sky = _attach(
        prepared, _fixed_pointing(offsets, taylor), 3, is_altaz=(True,), full_jones=full_jones, duration=3600.0
    )
    unpointed = _attach(prepared, None, 3, is_altaz=(True,), full_jones=full_jones, duration=3600.0)
    tgrid, chi_grid = pa_sample_grid(BASE_TIME, 3600.0, 0.0, MKAT_LAT, MKAT_LON, MKAT_LAT, 1.0)
    np.testing.assert_array_equal(tgrid, sky.tgrid)
    chi0 = chi_grid[:1]
    assert abs(chi0[0]) > 0.05

    a1, a2 = np.array([0, 1, 0]), np.array([1, 2, 2])
    times = np.full(3, tgrid[0])
    vis = _predict(sky, a1, a2, times=times)[:, 0, [0, -1]]
    base = _predict(unpointed, a1, a2, times=times)[:, 0, [0, -1]]
    exact = np.array([JIM.voltage(ell, emm, freqs, chi0, offset=o)[0, 0, 0] for o in offsets])  # (nant, 2)
    expected = exact[a1] * np.conj(exact[a2])
    tol = 2e-3
    np.testing.assert_allclose(vis, expected, rtol=tol)
    assert np.abs(vis - base).max() > 10 * tol * np.abs(expected).max()


@pytest.mark.parametrize("full_jones", [False, True])
@pytest.mark.parametrize("taylor", ["first", "laplacian"])
def test_kernel_interpolates_derivatives_in_pa(taylor, full_jones):
    """A row between two PA samples sees (E, D_l, D_m, L) each linearly interpolated."""
    freqs = np.array([1.4e9, 1.41e9])
    ncorr = 4 if full_jones else 2
    offsets = np.array([[30.0, -20.0], [-25.0, 30.0]]) * ARCSEC
    ell, emm = _ring([0.6, 0.9])
    sky = _attach(
        _prepared(_lmn(ell, emm), freqs, ncorr),
        _fixed_pointing(offsets, taylor),
        2,
        is_altaz=(True,),
        full_jones=full_jones,
        duration=7200.0,
    )
    tg = sky.tgrid
    assert tg.size > 2
    k, wt = 1, 0.3
    vis = _predict(sky, [0], [1], times=np.array([tg[k] + wt * (tg[k + 1] - tg[k])]))[0]

    def interp(grid):
        g = grid[0].astype(np.complex128)
        return g[k] * (1 - wt) + g[k + 1] * wt  # (nsrc, nchan, 2[, 2])

    e, dl, dm = interp(sky.beam_grid), interp(sky.beam_dl), interp(sky.beam_dm)
    lap = interp(sky.beam_lap) if taylor == "laplacian" else 0.0
    ep, eq = (e + o[0] * dl + o[1] * dm + 0.25 * (o @ o) * lap for o in offsets)
    if full_jones:
        expected = np.einsum("sfij,sfkj->sfik", ep, np.conj(eq)).sum(axis=0).reshape(freqs.size, 4)
    else:
        expected = (ep * np.conj(eq)).sum(axis=0)
    np.testing.assert_allclose(vis, expected, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("full_jones", [False, True])
def test_zero_offsets_are_bit_identical_to_no_pointing(full_jones):
    freqs = np.array([1.4e9, 1.404e9, 1.408e9])
    ncorr = 4 if full_jones else 2
    rng = np.random.default_rng(3)
    uvw = rng.normal(0, 500, (6, 3))
    a1, a2 = np.array([0, 0, 1, 1, 2, 0]), np.array([1, 2, 2, 3, 3, 3])
    times = BASE_TIME + np.linspace(0, 3000, 6)
    prepared = _prepared(_lmn(*_ring([0.3, 0.7])), freqs, ncorr)
    kw = dict(is_altaz=(True,), full_jones=full_jones, duration=3600.0)
    base = _predict(_attach(prepared, None, 4, **kw), a1, a2, times=times, uvw=uvw)
    for taylor in ("first", "laplacian"):
        sky = _attach(prepared, _fixed_pointing(np.zeros((4, 2)), taylor), 4, **kw)
        np.testing.assert_array_equal(_predict(sky, a1, a2, times=times, uvw=uvw), base)

    # A laplacian run with its L slab zeroed is bit-identical to the first-order run.
    offsets = rng.normal(0, SIGMA, (4, 2))
    lapl = _attach(prepared, _fixed_pointing(offsets, "laplacian"), 4, **kw)
    first = replace(lapl, pointing_order=1, beam_lap=None)
    zeroed = replace(lapl, beam_lap=np.zeros_like(lapl.beam_lap))
    np.testing.assert_array_equal(
        _predict(zeroed, a1, a2, times=times, uvw=uvw), _predict(first, a1, a2, times=times, uvw=uvw)
    )
    assert not np.array_equal(_predict(lapl, a1, a2, times=times, uvw=uvw), base)


def test_offset_toward_source_brightens_it():
    freqs = np.array([1.4e9])
    prepared = _prepared(_lmn(np.radians(0.5), 0.0), freqs, 2)
    nominal = np.abs(_predict(_attach(prepared, None, 2), [0], [1]))
    toward = _attach(prepared, _fixed_pointing(np.full((2, 2), [30 * ARCSEC, 0.0]), "first"), 2)
    away = _attach(prepared, _fixed_pointing(np.full((2, 2), [-30 * ARCSEC, 0.0]), "first"), 2)
    assert (np.abs(_predict(toward, [0], [1])) > nominal).all()
    assert (np.abs(_predict(away, [0], [1])) < nominal).all()


@pytest.mark.parametrize("taylor", ["first", "laplacian"])
def test_diagonal_and_jones_kernels_agree(taylor):
    freqs = np.array([1.4e9, 1.42e9])
    rng = np.random.default_rng(8)
    offsets = rng.normal(0, SIGMA, (4, 2))
    a1, a2 = np.array([0, 0, 1, 2]), np.array([1, 3, 2, 3])
    times = BASE_TIME + np.array([0.0, 900.0, 1800.0, 3000.0])
    uvw = rng.normal(0, 300, (4, 3))
    prepared = _prepared(_lmn(*_ring([0.2, 0.6, 1.0])), freqs, 4)
    kw = dict(is_altaz=(True,), duration=3600.0)
    diag = _attach(prepared, _fixed_pointing(offsets, taylor), 4, **kw)
    full = _attach(prepared, _fixed_pointing(offsets, taylor), 4, full_jones=True, **kw)
    np.testing.assert_allclose(
        _predict(diag, a1, a2, times=times, uvw=uvw), _predict(full, a1, a2, times=times, uvw=uvw), atol=1e-6
    )


def test_large_offsets_warn_that_the_taylor_model_breaks_down(caplog):
    prepared = _prepared(_lmn(*_ring([0.5])), np.array([1.4e9]), 2)
    with caplog.at_level(logging.WARNING, logger="skysim"):
        _attach(prepared, _fixed_pointing(np.full((2, 2), SIGMA)), 2)
    assert not any("Taylor model" in r.getMessage() for r in caplog.records)
    with caplog.at_level(logging.WARNING, logger="skysim"):
        _attach(prepared, _fixed_pointing(np.full((2, 2), 600 * ARCSEC)), 2)
    assert any("Taylor model" in r.getMessage() for r in caplog.records)


def test_predict_block_refuses_pointing_without_a_beam():
    sky = replace(_prepared(_lmn(0.0, 0.0), np.array([1.4e9]), 2), pointing=_fixed_pointing(np.zeros((2, 2))))
    with pytest.raises(ValueError, match="primary beam"):
        _predict(sky, [0], [1])


# --------------------------------------------------------------------------- FITS-cube beams


def _fits_jim(name="cube"):
    beam = CosineTaperBeam.from_builtin("MKAT-AA-L-JIM-2020")
    grid = np.linspace(-0.04, 0.04, 161)
    freqs = np.array([1.4e9])
    ll, mm = np.meshgrid(grid, grid, indexing="ij")
    cube = beam.voltages(np.degrees(ll.ravel()), np.degrees(mm.ravel()), freqs / 1e6).reshape(161, 161, 1, 2)
    return FitsBeamProvider.from_arrays(grid, grid, freqs, cube, name=name)


def test_fits_only_laplacian_falls_back_to_first(caplog):
    prepared = _prepared(_lmn(*_ring([0.3, 0.6])), np.array([1.4e9]), 2)
    fits = _fits_jim()
    one = _beam_grid_gib(1, 2, 2, 1, 2)
    with caplog.at_level(logging.WARNING, logger="skysim"):
        # 3x the beam grid (E + 2 derivative grids) fits a 3.5x ceiling.
        sky = _attach(
            prepared, _fixed_pointing(np.full((2, 2), SIGMA)), 2, providers=(fits,), beam_grid_max_gib=3.5 * one
        )
    assert sky.pointing_order == 1 and sky.beam_lap is None
    assert any("effectively taylor: first" in r.getMessage() for r in caplog.records)


def test_mixed_fits_and_jimbeam_types(caplog):
    prepared = _prepared(_lmn(*_ring([0.3, 0.6])), np.array([1.4e9]), 2)
    with caplog.at_level(logging.WARNING, logger="skysim"):
        sky = _attach(
            prepared,
            _fixed_pointing(np.full((2, 2), SIGMA)),
            2,
            providers=(JIM, _fits_jim("holo-cube")),
            is_altaz=(False, False),
            ant_type=np.array([0, 1]),
        )
    assert sky.pointing_order == 2
    assert any("holo-cube" in r.getMessage() for r in caplog.records)
    assert not sky.beam_lap[1].any()
    assert np.abs(sky.beam_lap[0]).min() > 0
    # The FITS type still carries first-order derivatives.
    assert np.abs(sky.beam_dl[1]).max() > 0


def test_laplacian_memory_ceiling_is_checked_before_allocation(monkeypatch):
    import simms.skymodel.beams as beams

    prepared = _prepared(_lmn(*_ring([0.3, 0.6])), np.array([1.4e9]), 2)
    one = _beam_grid_gib(1, 2, 2, 1, 2)

    def boom(*a, **k):
        raise AssertionError("the beam grid was built before the footprint check")

    monkeypatch.setattr(beams, "build_beam_grid", boom)
    with pytest.raises(MemoryError, match=r"with pointing errors.*3 pointing-error derivative grids"):
        _attach(prepared, _fixed_pointing(np.full((2, 2), SIGMA)), 2, beam_grid_max_gib=3.5 * one)


@pytest.mark.parametrize("taylor, copies", [("first", 3), ("laplacian", 4)])
def test_one_footprint_warning_per_pointed_run(caplog, taylor, copies):
    prepared = _prepared(_lmn(*_ring([0.3, 0.6])), np.array([1.4e9]), 2)
    one = _beam_grid_gib(1, 2, 2, 1, 2)
    with caplog.at_level(logging.WARNING, logger="skysim"):
        # Over half the ceiling for the whole set, so the combined check warns -- once.
        _attach(prepared, _fixed_pointing(np.full((2, 2), SIGMA), taylor), 2, beam_grid_max_gib=1.5 * one * copies)
    grid_warnings = [r.getMessage() for r in caplog.records if "Primary-beam grid" in r.getMessage()]
    assert len(grid_warnings) == 1
    assert f"x {copies} for the beam plus {copies - 1} pointing-error derivative grids" in grid_warnings[0]


def test_loss_log_probes_a_smooth_type(caplog):
    prepared = _prepared(_lmn(0.0, 0.0), np.array([1.4e9]), 2)
    with caplog.at_level(logging.INFO, logger="skysim"):
        _attach(
            prepared,
            _fixed_pointing(np.full((2, 2), SIGMA)),
            2,
            providers=(_fits_jim("holo-cube"), JIM),
            is_altaz=(False, False),
            ant_type=np.array([0, 1]),
        )
    (line,) = [r.getMessage() for r in caplog.records if "Predicted mean voltage change" in r.getMessage()]
    assert "beam type 1 (" in line
    assert float(line.split("= ")[1].split(" ")[0]) < -1e-5  # a real on-axis loss, not a FITS zero


# --------------------------------------------------------------------------- channel chunks


def test_select_channels_matches_whole_band():
    freqs = 1.4e9 + np.arange(3) * 4e6
    offsets = np.random.default_rng(4).normal(0, SIGMA, (3, 2))
    sky = _attach(_prepared(_lmn(*_ring([0.4, 0.8])), freqs, 2), _fixed_pointing(offsets), 3)
    a1, a2 = np.array([0, 0, 1]), np.array([1, 2, 2])
    uvw = np.random.default_rng(5).normal(0, 300, (3, 3))
    whole = _predict(sky, a1, a2, uvw=uvw)
    parts = [_predict(sky.select_channels(np.array(c)), a1, a2, uvw=uvw) for c in ([0], [1, 2])]
    np.testing.assert_allclose(np.concatenate(parts, axis=1), whole, rtol=1e-12)
    sub = sky.select_channels(np.array([1, 2]))
    assert sub.beam_dl.flags["C_CONTIGUOUS"] and sub.beam_lap.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(sub.beam_lap, sky.beam_lap[:, :, :, 1:])


# --------------------------------------------------------------------------- end to end


class _PtgTest(InitTest):
    def __init__(self, correlations):
        self.test_files = []
        self.ncorr = len(correlations)
        self.ms = self.random_named_directory(suffix=".ms")
        create_ms(
            self.ms,
            telescope_name="skamid",
            pointing_direction=["J2000", "0h0m20s", "-30deg"],
            dtime=600,
            ntimes=4,
            start_freq="1420MHz",
            dfreq="4MHz",
            nchan=2,
            correlations=correlations,
            row_chunks=100000,
            sefd=None,
            column="DATA",
            start_time="2025-03-06T20:00:00",
            smooth=None,
            fit_order=None,
            subarray_range=[60, 68],
        )
        # A source ~0.5 deg off the pointing centre, on the beam's steep shoulder.
        self.sky = self.random_named_file(suffix=".txt")
        with open(self.sky, "w") as fh:
            fh.write("#format: name ra dec stokes_i\nS 0h0m20s -30d30m0s 4.0\n")
        self.beams = self.random_named_file(suffix=".yaml")
        with open(self.beams, "w") as fh:
            fh.write("MKAT-MA:\n  jimbeam: MKAT-MA-L-JIM-2026\nMKAT-EA:\n  jimbeam: MKAT-EA-L-JIM-2026\n")

    def write_yaml(self, content: str) -> str:
        path = self.random_named_file(suffix=".yaml")
        with open(path, "w") as fh:
            fh.write(content)
        return path

    def run(self, column, corruptions=None, **kw):
        kw.setdefault("primary_beam", self.beams)
        if self.ncorr == 4:
            kw.setdefault("beam_jones", "full")
        skysim.runit(skysim_opts(self.ms, ascii_sky=self.sky, column=column, corruptions=corruptions, **kw))
        return getattr(xds_from_ms(self.ms)[0], column).data.compute()


@pytest.fixture(scope="module", params=[["XX", "YY"], ["XX", "XY", "YX", "YY"]], ids=["diag", "jones"])
def e2e(request):
    return _PtgTest(request.param)


@pytest.fixture(scope="module")
def e2e2():
    return _PtgTest(["XX", "YY"])


_POINTING = "pointing:\n  static: 30arcsec\n  amplitude: 10arcsec\n  period: 20min\n"


def test_end_to_end_pointing(e2e):
    spec = e2e.write_yaml(_POINTING)
    beam = e2e.run("BEAM")
    ptg = e2e.run("PTG", spec, seed_gains=3)
    again = e2e.run("PTG_AGAIN", spec, seed_gains=3)
    other = e2e.run("PTG_OTHER", spec, seed_gains=4)
    first = e2e.run("PTG_FIRST", e2e.write_yaml(_POINTING + "  taylor: first\n"), seed_gains=3)

    hands = [0, -1]
    assert np.isfinite(ptg).all()
    rel = np.abs(ptg[..., hands] / beam[..., hands] - 1).max()
    assert 1e-4 <= rel <= 0.1
    np.testing.assert_array_equal(ptg, again)
    assert not np.array_equal(ptg, other)
    assert not np.array_equal(ptg, first)


def test_channel_and_row_chunking_do_not_change_the_result(e2e2):
    spec = e2e2.write_yaml(_POINTING)
    whole = e2e2.run("CHUNK_WHOLE", spec, seed_gains=9)
    chunked = e2e2.run("CHUNK_SPLIT", spec, seed_gains=9, chan_chunks=1, row_chunks=7)
    np.testing.assert_allclose(chunked, whole, rtol=1e-12)


def test_pointing_model_is_wired_from_the_whole_ms(e2e2, monkeypatch):
    seen = {}
    real = skysim.build_pointing_model

    def spy(spec, nant, t0, random_seed=None):
        seen.update(nant=nant, t0=t0, seed=random_seed)
        return real(spec, nant, t0, random_seed=random_seed)

    monkeypatch.setattr(skysim, "build_pointing_model", spy)
    e2e2.run("WIRED", e2e2.write_yaml(_POINTING), seed_gains=21)
    nant = xds_from_table(f"{e2e2.ms}::ANTENNA")[0].sizes["row"]
    t0 = float(xds_from_ms(e2e2.ms, columns=["TIME"], group_cols=[])[0].TIME.data.min().compute())
    assert seen == {"nant": nant, "t0": t0, "seed": 21}


def test_gains_compose_with_pointing(e2e2):
    both = e2e2.write_yaml(_POINTING + _GAINS)
    gains = e2e2.write_yaml(_GAINS)
    pointing_only = e2e2.write_yaml(_POINTING)
    v_beam = e2e2.run("C_BEAM")
    v_ptg = e2e2.run("C_PTG", pointing_only, seed_gains=5)
    v_gain = e2e2.run("C_GAIN", gains, seed_gains=5)
    v_both = e2e2.run("C_BOTH", both, seed_gains=5)
    np.testing.assert_allclose(v_both / v_ptg, v_gain / v_beam, rtol=1e-6)
    assert not np.allclose(v_both, v_gain)


@pytest.mark.parametrize(
    "overrides",
    [
        {"ascii_sky": None, "fits_sky": "missing.fits"},
        {"ascii_sky": None, "wsclean_sky": "missing.txt"},
        {"ascii_sky": None, "sefd": 500.0},
    ],
    ids=["fits", "wsclean", "noise-only"],
)
def test_pointing_needs_an_ascii_sky(e2e2, overrides):
    opts = skysim_opts(e2e2.ms, ascii_sky=e2e2.sky, column="ERR", primary_beam=e2e2.beams)
    opts.corruptions = e2e2.write_yaml(_POINTING)
    for key, value in overrides.items():
        setattr(opts, key, value)
    with pytest.raises(RuntimeError, match="only supported for --ascii-sky"):
        skysim.runit(opts)


def test_pointing_needs_a_primary_beam(e2e2):
    with pytest.raises(RuntimeError, match="--primary-beam"):
        e2e2.run("ERR", e2e2.write_yaml(_POINTING), primary_beam=None)


def test_malformed_pointing_fails_on_a_noise_only_run(e2e2):
    opts = skysim_opts(e2e2.ms, column="ERR", sefd=500.0, corruptions=e2e2.write_yaml("pointing:\n  static: 30\n"))
    with pytest.raises(RuntimeError, match="explicit angle unit"):
        skysim.runit(opts)
