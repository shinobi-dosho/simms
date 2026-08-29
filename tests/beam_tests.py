"""Tests for the vendored cosine-taper primary-beam model (simms.skymodel.beams)."""

import numpy as np
import pytest

from simms.skymodel.beams import (
    R_FWHM,
    CosineTaperBeam,
    JimBeamProvider,
    UnityBeamProvider,
    cosine_taper,
    load_beam_config,
    local_sidereal_time,
    parallactic_angle,
    resolve_antenna_beams,
)

# MeerKAT reference site (geodetic, radians).
MKAT_LON = np.deg2rad(21.4439)
MKAT_LAT = np.deg2rad(-30.7130)
# A modern epoch as MS TIME seconds (MJD * 86400); exact date is irrelevant.
BASE_TIME = 59_000.0 * 86400.0 + 20 * 3600.0


def test_cosine_taper_analytic_properties():
    # Peaks at 1 on axis.
    assert cosine_taper(np.array([0.0]))[0] == pytest.approx(1.0)
    # Half-power at r = 0.5: |voltage|**2 == 0.5.
    assert cosine_taper(np.array([0.5]))[0] ** 2 == pytest.approx(0.5, abs=1e-6)


def test_cosine_taper_removable_singularity():
    # 1 - 4*rr**2 vanishes at rr = 0.5, i.e. r = 0.5 / R_FWHM. Limit is pi/4.
    r_sing = 0.5 / R_FWHM
    val = cosine_taper(np.array([r_sing]))[0]
    assert np.isfinite(val)
    assert val == pytest.approx(np.pi / 4.0, abs=1e-6)


def test_half_power_at_fwhm_over_two():
    # A source offset by FWHM/2 along one feed axis sits at the half-power point,
    # so the power beam |g|**2 == 0.5 there (guards voltage-vs-power confusion).
    beam = CosineTaperBeam.from_builtin("MKAT-AA-L-JIM-2020")
    freq = np.array([1400.0])
    squint, fwhm = beam._interp(freq)  # (4, 1)
    hx_squint, hx_fwhm = squint[0, 0], fwhm[0, 0]
    x = np.array([hx_squint + hx_fwhm / 2.0])
    y = np.array([squint[1, 0]])  # centre in y for the H feed
    g = beam.voltages(x, y, freq)  # (1, 1, 2)
    assert np.abs(g[0, 0, 0]) ** 2 == pytest.approx(0.5, abs=1e-6)


@pytest.mark.parametrize("name", ["MKAT-AA-L-JIM-2020", "MKAT-AA-UHF-JIM-2020"])
def test_matches_katbeam_oracle(name):
    # Faithful reimplementation: agree with upstream katbeam where it is installed.
    katbeam = pytest.importorskip("katbeam")
    jb = katbeam.JimBeam(name)
    ours = CosineTaperBeam.from_builtin(name)

    freq = 900.0 if "UHF" not in name else 700.0
    margin = np.linspace(-3.0, 3.0, 25)
    x, y = np.meshgrid(margin, margin)
    xf, yf = x.ravel(), y.ravel()

    ref_h = jb.HH(xf, yf, freq)
    ref_v = jb.VV(xf, yf, freq)
    g = ours.voltages(xf, yf, np.array([freq]))[:, 0, :]

    np.testing.assert_allclose(g[:, 0], ref_h, atol=1e-5)
    np.testing.assert_allclose(g[:, 1], ref_v, atol=1e-5)


def test_parallactic_angle_transit_and_antisymmetry():
    # Put the field transit (hour angle 0) at the middle timestamp by setting ra0 = LST there.
    t_mid = BASE_TIME
    ra0 = float(local_sidereal_time(np.array([t_mid]), MKAT_LON)[0])
    dec0 = np.deg2rad(-45.0)  # south of the -30.7 deg zenith

    # At transit the zenith is due north of a source south of it -> chi == 0.
    chi_mid = parallactic_angle(np.array([t_mid]), ra0, dec0, MKAT_LON, MKAT_LAT)[0]
    assert chi_mid == pytest.approx(0.0, abs=1e-6)

    # Antisymmetric about transit: chi(+dt) == -chi(-dt).
    dt = 1800.0  # seconds
    chi = parallactic_angle(np.array([t_mid - dt, t_mid + dt]), ra0, dec0, MKAT_LON, MKAT_LAT)
    assert chi[0] == pytest.approx(-chi[1], abs=1e-6)


def test_parallactic_angle_matches_astropy():
    # Independent oracle: the parallactic angle is the position angle of the zenith at
    # the field centre. astropy's full apparent-place transform differs from our mean-LST
    # closed form by the (sub-degree) apparent-place modelling; 0.5 deg validates sign+geometry.
    pytest.importorskip("astropy")
    from astropy import units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    ra0, dec0 = np.deg2rad(45.0), np.deg2rad(-45.0)
    times = BASE_TIME + np.linspace(0, 4 * 3600, 7)
    try:
        loc = EarthLocation.from_geodetic(MKAT_LON * u.rad, MKAT_LAT * u.rad)
        t = Time(times / 86400.0, format="mjd", scale="utc")
        src = SkyCoord(ra0 * u.rad, dec0 * u.rad, frame="icrs")
        zenith = SkyCoord(alt=90 * u.deg, az=0 * u.deg, frame=AltAz(obstime=t, location=loc)).icrs
        q = src.position_angle(zenith).to_value(u.rad)
    except Exception as exc:  # pragma: no cover - offline IERS etc.
        pytest.skip(f"astropy transform unavailable: {exc}")

    chi = parallactic_angle(times, ra0, dec0, MKAT_LON, MKAT_LAT)
    diff = (chi - q + np.pi) % (2 * np.pi) - np.pi  # wrapped difference
    assert np.max(np.abs(diff)) < np.deg2rad(0.5)


def test_from_csv_roundtrip(tmp_path):
    # Arbitrary CSVs (e.g. the MeerKAT-Extension tables) load the same way.
    csv = tmp_path / "toy.csv"
    csv.write_text(
        "freq,Hx squint,Hy squint,Vx squint,Vy squint,Hx fwhm,Hy fwhm,Vx fwhm,Vy fwhm\n"
        "MHz,arcmin,arcmin,arcmin,arcmin,arcmin,arcmin,arcmin,arcmin\n"
        "1000,0,0,0,0,60,60,60,60\n"
        "2000,0,0,0,0,30,30,30,30\n"
    )
    beam = CosineTaperBeam.from_csv(csv)
    # FWHM 60 arcmin = 1 deg at 1000 MHz; half-power at 0.5 deg offset.
    g = beam.voltages(np.array([0.5]), np.array([0.0]), np.array([1000.0]))
    assert np.abs(g[0, 0, 0]) ** 2 == pytest.approx(0.5, abs=1e-6)
    # On-axis is unity at both feeds and both channels.
    g0 = beam.voltages(np.array([0.0]), np.array([0.0]), np.array([1000.0, 2000.0]))
    np.testing.assert_allclose(g0[0], 1.0, atol=1e-9)


@pytest.mark.parametrize(
    "name",
    ["MKAT-MA-L-JIM-2026", "MKAT-EA-L-JIM-2026", "MKAT-MA-S-JIM-2020", "MKAT-EA-S3-JIM-2026"],
)
def test_meerkat_extension_builtins(name):
    # Bundled MeerKAT / MeerKAT-Extension tables load and are physically sane:
    # unity at the squint centre, half power at FWHM/2 from it.
    beam = CosineTaperBeam.from_builtin(name)
    freq = np.array([float(np.mean(beam.freqs_mhz))])  # in-band for L or S
    sq, fw = beam._interp(freq)  # (4, 1)
    cx, cy = np.array([sq[0, 0]]), np.array([sq[1, 0]])  # H-feed squint centre
    peak = beam.voltages(cx, cy, freq)[0, 0, 0]
    hwhm = beam.voltages(cx + fw[0, 0] / 2, cy, freq)[0, 0, 0]
    assert abs(peak) == pytest.approx(1.0, abs=1e-9)
    assert abs(hwhm) ** 2 == pytest.approx(0.5, abs=1e-6)


@pytest.mark.parametrize(
    "mk_name, mke_name, freq",
    [
        ("MKAT-MA-L-JIM-2026", "MKAT-EA-L-JIM-2026", 1400.0),
        ("MKAT-MA-S-JIM-2020", "MKAT-EA-S3-JIM-2026", 2800.0),
    ],
)
def test_mke_beam_is_narrower_than_mk(mk_name, mke_name, freq):
    # The MeerKAT-Extension (15 m) beam is narrower than the MeerKAT (13.5 m) beam.
    f = np.array([freq])
    mk = CosineTaperBeam.from_builtin(mk_name)._interp(f)[1][0, 0]
    mke = CosineTaperBeam.from_builtin(mke_name)._interp(f)[1][0, 0]
    assert mke < mk


def test_jimbeam_provider_matches_cosine_taper_at_zero_pa():
    beam = CosineTaperBeam.from_builtin("MKAT-AA-L-JIM-2020")
    provider = JimBeamProvider(beam)
    ell = np.array([0.0, 0.01, -0.02])
    emm = np.array([0.0, -0.015, 0.005])
    freqs_hz = np.array([1.3e9, 1.4e9])
    g = provider.voltage(ell, emm, freqs_hz, chi=np.array([0.0]))  # (1, nsrc, nchan, 2)
    ref = beam.voltages(np.degrees(ell), np.degrees(emm), freqs_hz / 1e6)
    assert g.shape == (1, ell.size, freqs_hz.size, 2)
    # The provider is just the degrees/MHz-converted cosine-taper voltage at chi = 0.
    # (Not exactly unity on axis: the L-band beam has a small squint offset.)
    np.testing.assert_allclose(g[0].real, ref, atol=1e-12)


def test_beam_rotation_by_parallactic_angle():
    # Rotating the sky by chi = pi/2 maps (l, m) = (a, 0) to feed (0, -a).
    provider = JimBeamProvider(CosineTaperBeam.from_builtin("MKAT-AA-L-JIM-2020"))
    freqs = np.array([1.4e9])
    a = 0.02
    rotated = provider.voltage(np.array([a]), np.array([0.0]), freqs, chi=np.array([np.pi / 2]))
    direct = provider.voltage(np.array([0.0]), np.array([-a]), freqs, chi=np.array([0.0]))
    np.testing.assert_allclose(rotated, direct, atol=1e-12)


def test_unity_provider_is_flat():
    g = UnityBeamProvider().voltage(
        np.array([0.0, 0.3]), np.array([0.1, -0.2]), np.array([1e9, 2e9]), chi=np.array([0.0, 1.0])
    )
    assert g.shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(g, 1.0)


def test_resolve_antenna_beams_heterogeneous():
    telescope_names = ["MKAT-MA", "MKAT-MA", "MKAT-EA"]
    mount = ["ALT-AZ", "ALT-AZ", "ALT-AZ"]
    config = {"MKAT-MA": {"jimbeam": "L"}, "MKAT-EA": {"jimbeam": "MKAT-AA-L-JIM-2020"}}
    ant_type, providers, is_altaz = resolve_antenna_beams(telescope_names, mount, config)
    np.testing.assert_array_equal(ant_type, [0, 0, 1])
    assert len(providers) == 2
    assert all(isinstance(p, JimBeamProvider) for p in providers)
    np.testing.assert_array_equal(is_altaz, [True, True])


def test_resolve_antenna_beams_unmapped_and_equatorial():
    ant_type, providers, is_altaz = resolve_antenna_beams(
        ["MKAT-MA", "XX"], ["EQUATORIAL", "ALT-AZ"], {"MKAT-MA": {"jimbeam": "L"}}
    )
    np.testing.assert_array_equal(ant_type, [0, 1])
    assert isinstance(providers[0], JimBeamProvider)
    assert isinstance(providers[1], UnityBeamProvider)  # XX has no config entry
    np.testing.assert_array_equal(is_altaz, [False, True])


def test_load_beam_config(tmp_path):
    cfg = tmp_path / "beams.yaml"
    cfg.write_text("MKAT-MA:\n  jimbeam: L\nMKAT-EA:\n  jimbeam: MKAT-EA-L-JIM-2026.csv\n")
    loaded = load_beam_config(cfg)
    assert loaded["MKAT-MA"]["jimbeam"] == "L"
    assert loaded["MKAT-EA"]["jimbeam"] == "MKAT-EA-L-JIM-2026.csv"


# --- End-to-end beam application through predict_block ---------------------------

from simms.skymodel.kernels import is_uniform_grid  # noqa: E402
from simms.skymodel.mstools import PreparedSky, attach_beam, predict_block  # noqa: E402


def _make_prepared(lmn, flux, freqs, ncorr):
    """A minimal full-correlation PreparedSky of unpolarised point sources."""
    nsrc, nchan = lmn.shape[0], freqs.size
    bmat = np.zeros((nsrc, ncorr, nchan), dtype=np.complex128)
    bmat[:, 0, :] = flux[:, None]  # XX = I
    bmat[:, -1, :] = flux[:, None]  # YY = I
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


def _lmn(ell, emm):
    return np.array([[ell, emm, np.sqrt(1 - ell * ell - emm * emm) - 1]])


def _attach(
    prepared,
    telescope_names,
    mount,
    config,
    ncorr,
    ra0=0.0,
    dec0=MKAT_LAT,
    duration=1.0,
    full_jones=False,
    basis_transform=None,
):
    ant_type, providers, is_altaz = resolve_antenna_beams(telescope_names, mount, config)
    return attach_beam(
        prepared,
        ant_type,
        providers,
        is_altaz,
        ra0,
        dec0,
        MKAT_LON,
        MKAT_LAT,
        BASE_TIME,
        duration,
        1.0,
        ncorr,
        full_jones=full_jones,
        basis_transform=basis_transform,
    )


def test_predict_beam_boresight_vs_offcentre():
    freqs = np.array([1.4e9])
    config = {"MKAT-MA": {"jimbeam": "L"}}
    uvw = np.array([[120.0, 60.0, 5.0]])
    times = np.array([BASE_TIME])
    a1, a2 = np.array([0]), np.array([1])
    provider = JimBeamProvider(CosineTaperBeam.from_builtin("MKAT-AA-L-JIM-2020"))

    for ell in (0.0, 0.05):  # boresight, then ~2.9 deg off-centre
        prepared = _attach(
            _make_prepared(_lmn(ell, 0.0), np.array([2.0]), freqs, 2),
            ["MKAT-MA", "MKAT-MA"],
            ["ALT-AZ", "ALT-AZ"],
            config,
            2,
        )
        vis = predict_block(prepared, uvw, times=times, antenna1=a1, antenna2=a2)
        # No-beam reference (same phasor), so the ratio is exactly the power beam.
        ref = predict_block(_make_prepared(_lmn(ell, 0.0), np.array([2.0]), freqs, 2), uvw)
        chi = parallactic_angle(times, 0.0, MKAT_LAT, MKAT_LON, MKAT_LAT)
        g = provider.voltage(np.array([ell]), np.array([0.0]), freqs, chi)[0, 0, 0]
        np.testing.assert_allclose(vis[0, 0, 0], ref[0, 0, 0] * g[0] * np.conj(g[0]), rtol=1e-6)
        np.testing.assert_allclose(vis[0, 0, 1], ref[0, 0, 1] * g[1] * np.conj(g[1]), rtol=1e-6)

    # Off-centre power beam attenuates; boresight is ~unity.
    assert abs(g[0]) ** 2 < 0.98


def test_predict_beam_heterogeneous_baseline():
    freqs = np.array([1.4e9])
    config = {"MKAT-MA": {"jimbeam": "L"}, "MKAT-EA": {"jimbeam": "MKAT-AA-UHF-JIM-2020"}}
    uvw = np.array([[120.0, 60.0, 5.0]])
    times = np.array([BASE_TIME])
    ell = 0.04
    # Baseline mixes an MK antenna (type 0) with an MKE antenna (type 1).
    prepared = _attach(
        _make_prepared(_lmn(ell, 0.0), np.array([2.0]), freqs, 2),
        ["MKAT-MA", "MKAT-EA"],
        ["ALT-AZ", "ALT-AZ"],
        config,
        2,
    )
    vis = predict_block(prepared, uvw, times=times, antenna1=np.array([0]), antenna2=np.array([1]))
    ref = predict_block(_make_prepared(_lmn(ell, 0.0), np.array([2.0]), freqs, 2), uvw)

    chi = parallactic_angle(times, 0.0, MKAT_LAT, MKAT_LON, MKAT_LAT)
    g_mk = JimBeamProvider(CosineTaperBeam.from_builtin("MKAT-AA-L-JIM-2020")).voltage(
        np.array([ell]), np.array([0.0]), freqs, chi
    )[0, 0, 0]
    g_mke = JimBeamProvider(CosineTaperBeam.from_builtin("MKAT-AA-UHF-JIM-2020")).voltage(
        np.array([ell]), np.array([0.0]), freqs, chi
    )[0, 0, 0]
    np.testing.assert_allclose(vis[0, 0, 0], ref[0, 0, 0] * g_mk[0] * np.conj(g_mke[0]), rtol=1e-6)
    # A mixed baseline differs from a same-type (MK-MK) one.
    same = _attach(
        _make_prepared(_lmn(ell, 0.0), np.array([2.0]), freqs, 2),
        ["MKAT-MA", "MKAT-MA"],
        ["ALT-AZ", "ALT-AZ"],
        config,
        2,
    )
    vis_same = predict_block(same, uvw, times=times, antenna1=np.array([0]), antenna2=np.array([1]))
    assert not np.isclose(vis[0, 0, 0], vis_same[0, 0, 0])


def test_predict_beam_squint_breaks_xx_yy_and_zeros_crosshands():
    freqs = np.array([1.4e9])
    config = {"MKAT-MA": {"jimbeam": "L"}}
    uvw = np.array([[150.0, 70.0, 8.0]])
    times = np.array([BASE_TIME])
    prepared = _attach(
        _make_prepared(_lmn(0.05, 0.03), np.array([3.0]), freqs, 4),
        ["MKAT-MA", "MKAT-MA"],
        ["ALT-AZ", "ALT-AZ"],
        config,
        4,
    )
    vis = predict_block(prepared, uvw, times=times, antenna1=np.array([0]), antenna2=np.array([1]))
    # Unpolarised source: cross-hands stay exactly zero.
    np.testing.assert_array_equal(vis[0, 0, 1], 0.0)
    np.testing.assert_array_equal(vis[0, 0, 2], 0.0)
    # Per-feed squint/FWHM differ, so XX != YY off-centre.
    assert not np.isclose(vis[0, 0, 0], vis[0, 0, 3])


def test_predict_beam_mount_altaz_vs_equatorial():
    freqs = np.array([1.4e9])
    config = {"MKAT-MA": {"jimbeam": "L"}}
    uvw = np.array([[120.0, 60.0, 5.0]])
    ell = 0.05
    # Two timestamps 2 h apart -> different parallactic angles.
    times = np.array([BASE_TIME, BASE_TIME + 2 * 3600.0])
    uvw2 = np.repeat(uvw, 2, axis=0)
    a1, a2 = np.array([0, 0]), np.array([1, 1])

    altaz = _attach(
        _make_prepared(_lmn(ell, 0.02), np.array([2.0]), freqs, 2),
        ["MKAT-MA", "MKAT-MA"],
        ["ALT-AZ", "ALT-AZ"],
        config,
        2,
        duration=2 * 3600.0,
    )
    v_altaz = predict_block(altaz, uvw2, times=times, antenna1=a1, antenna2=a2)
    # ALT-AZ beam rotates with PA -> the two timestamps differ.
    assert not np.isclose(v_altaz[0, 0, 0], v_altaz[1, 0, 0])

    equ = _attach(
        _make_prepared(_lmn(ell, 0.02), np.array([2.0]), freqs, 2),
        ["MKAT-MA", "MKAT-MA"],
        ["EQUATORIAL", "EQUATORIAL"],
        config,
        2,
        duration=2 * 3600.0,
    )
    v_equ = predict_block(equ, uvw2, times=times, antenna1=a1, antenna2=a2)
    # EQUATORIAL beam does not rotate -> identical across timestamps.
    np.testing.assert_allclose(v_equ[0, 0, 0], v_equ[1, 0, 0], rtol=1e-6)


# --- FITS-cube provider (Phase 4) ------------------------------------------------

from simms.skymodel.beams import FitsBeamProvider  # noqa: E402


def _jimbeam_cube(beam, l_grid, m_grid, freqs_hz):
    """Sample a CosineTaperBeam onto an (nl, nm, nfreq, 2) feed-voltage cube."""
    ll, mm = np.meshgrid(l_grid, m_grid, indexing="ij")
    v = beam.voltages(np.degrees(ll.ravel()), np.degrees(mm.ravel()), freqs_hz / 1e6)
    return v.reshape(l_grid.size, m_grid.size, freqs_hz.size, 2)


def test_fits_provider_matches_jimbeam_cross_check():
    # Sample the analytic beam onto a grid, then interpolate it back: the FITS-cube
    # provider must agree with the analytic one (also validates the shared -chi rotation).
    beam = CosineTaperBeam.from_builtin("MKAT-AA-L-JIM-2020")
    l_grid = np.linspace(-0.06, 0.06, 121)
    m_grid = np.linspace(-0.06, 0.06, 121)
    freqs = np.array([1.3e9, 1.4e9])
    cube = _jimbeam_cube(beam, l_grid, m_grid, freqs)

    fits_prov = FitsBeamProvider.from_arrays(l_grid, m_grid, freqs, cube)
    jim_prov = JimBeamProvider(beam)
    ell = np.array([0.0, 0.01, -0.02])
    emm = np.array([0.005, -0.01, 0.0])
    chi = np.array([0.0, 0.3])
    gf = fits_prov.voltage(ell, emm, freqs, chi)
    gj = jim_prov.voltage(ell, emm, freqs, chi)
    np.testing.assert_allclose(gf, gj, atol=2e-3)


def test_fits_provider_from_fits_roundtrip(tmp_path):
    from astropy.io import fits

    beam = CosineTaperBeam.from_builtin("MKAT-AA-L-JIM-2020")
    l_grid = np.linspace(-0.06, 0.06, 61)
    m_grid = np.linspace(-0.06, 0.06, 61)
    freqs = np.array([1.3e9, 1.4e9])
    cube = _jimbeam_cube(beam, l_grid, m_grid, freqs)  # (nl, nm, nfreq, 2)

    hh = cube[..., 0].transpose(2, 1, 0)  # -> (nfreq, nm, nl)
    vv = cube[..., 1].transpose(2, 1, 0)
    data = np.stack([hh.real, hh.imag, vv.real, vv.imag], axis=0)  # (4, nfreq, nm, nl)

    hdr = fits.Header()
    hdr["CRPIX1"], hdr["CRVAL1"], hdr["CDELT1"] = 1, np.degrees(l_grid[0]), np.degrees(l_grid[1] - l_grid[0])
    hdr["CRPIX2"], hdr["CRVAL2"], hdr["CDELT2"] = 1, np.degrees(m_grid[0]), np.degrees(m_grid[1] - m_grid[0])
    hdr["CRPIX3"], hdr["CRVAL3"], hdr["CDELT3"] = 1, freqs[0], freqs[1] - freqs[0]
    hdr["CRPIX4"], hdr["CRVAL4"], hdr["CDELT4"] = 1, 0, 1
    path = tmp_path / "beam_cube.fits"
    fits.PrimaryHDU(data=data, header=hdr).writeto(path)

    prov = FitsBeamProvider.from_fits(path)
    jim = JimBeamProvider(beam)
    ell = np.array([0.0, 0.015])
    emm = np.array([0.0, -0.02])
    g_fits = prov.voltage(ell, emm, freqs, np.array([0.0]))
    g_jim = jim.voltage(ell, emm, freqs, np.array([0.0]))
    np.testing.assert_allclose(g_fits, g_jim, atol=2e-3)


def test_resolve_antenna_beams_fits_entry(tmp_path):
    # A {fits: path} entry builds a FitsBeamProvider.
    from astropy.io import fits

    l_grid = np.linspace(-0.05, 0.05, 21)
    cube_data = np.zeros((4, 1, l_grid.size, l_grid.size))
    cube_data[0] = 1.0  # HH real = 1
    cube_data[2] = 1.0  # VV real = 1
    crval, cdelt = np.degrees(l_grid[0]), np.degrees(l_grid[1] - l_grid[0])
    hdr = fits.Header()
    for ax in (1, 2):
        hdr[f"CRPIX{ax}"], hdr[f"CRVAL{ax}"], hdr[f"CDELT{ax}"] = 1, crval, cdelt
    hdr["CRPIX3"], hdr["CRVAL3"], hdr["CDELT3"] = 1, 1.4e9, 1.0
    hdr["CRPIX4"], hdr["CRVAL4"], hdr["CDELT4"] = 1, 0, 1
    path = tmp_path / "flat.fits"
    fits.PrimaryHDU(data=cube_data, header=hdr).writeto(path)

    _, providers, _ = resolve_antenna_beams(["MKAT-EA"], ["ALT-AZ"], {"MKAT-EA": {"fits": str(path)}})
    assert isinstance(providers[0], FitsBeamProvider)


# --- Cattery-schema FITS beams (read side) ----------------------------------------

from simms.skymodel.beams import (  # noqa: E402
    _axis_sign,
    _cattery_substitute,
    load_cattery_beam_json,
    resolve_cattery_antenna_beams,
    write_beam_fits_cattery,
)


def test_cattery_substitute():
    assert _cattery_substitute("beam_$(corr)_$(reim).fits", corr="xx", reim="re") == "beam_xx_re.fits"
    assert _cattery_substitute("beam_$(xy)_$(reim).fits", corr="rl", reim="im") == "beam_rl_im.fits"
    assert _cattery_substitute("beam_$(CORR)_$(REIM).fits", corr="xx", reim="re") == "beam_XX_RE.fits"
    assert _cattery_substitute("beam_$(realimag).fits", reim="im") == "beam_imag.fits"
    assert _cattery_substitute("beam_$(REALIMAG).fits", reim="im") == "beam_IMAG.fits"
    assert _cattery_substitute("$(stype)_$(corr)_$(reim).fits", stype="ska", corr="yy", reim="im") == "ska_yy_im.fits"
    assert _cattery_substitute("$(STYPE)_beam.fits", stype="ska") == "SKA_beam.fits"


@pytest.mark.parametrize("spec, expected", [("-X", -1.0), ("X", 1.0), ("-Y", -1.0), ("Y", 1.0)])
def test_axis_sign_valid(spec, expected):
    assert _axis_sign(spec) == expected


@pytest.mark.parametrize("spec", ["-x", "x", "foo", "-Z", ""])
def test_axis_sign_invalid(spec):
    with pytest.raises(ValueError, match="Invalid axis sign convention"):
        _axis_sign(spec)


@pytest.mark.parametrize("pol_basis", ["linear", "circular"])
def test_fits_provider_from_cattery_roundtrip(tmp_path, pol_basis):
    # write_beam_fits_cattery -> FitsBeamProvider.from_cattery must recover the same
    # feed-frame voltages as the analytic beam, for both the linear and (un-rotated)
    # circular on-disk basis.
    beam = CosineTaperBeam.from_builtin("MKAT-EA-L-JIM-2026")
    npix = 33
    grid = (np.arange(npix) - npix // 2) * np.radians(2.0 / 60.0)
    freqs = np.array([1.3e9, 1.4e9])
    prefix = str(tmp_path / "beam")

    write_beam_fits_cattery(beam, grid, grid, freqs, prefix, pol_basis=pol_basis)
    prov = FitsBeamProvider.from_cattery(prefix, pol_basis=pol_basis)
    assert prov.has_leakage

    jim = JimBeamProvider(beam)
    ell = np.array([0.001, -0.002])
    emm = np.array([0.0005, 0.0008])
    got = prov.voltage(ell, emm, freqs, np.array([0.0]))
    want = jim.voltage(ell, emm, freqs, np.array([0.0]))
    np.testing.assert_allclose(got, want, atol=2e-3)

    # No leakage in the cosine-taper model.
    jones = prov.jones(ell, emm, freqs, np.array([0.0]))
    np.testing.assert_allclose(jones[..., 0, 1], 0.0, atol=1e-9)
    np.testing.assert_allclose(jones[..., 1, 0], 0.0, atol=1e-9)


def test_resolve_antenna_beams_cattery_entry(tmp_path):
    # A {cattery: prefix} entry builds a FitsBeamProvider from the 8-file schema.
    beam = CosineTaperBeam.from_builtin("MKAT-EA-L-JIM-2026")
    npix = 17
    grid = (np.arange(npix) - npix // 2) * np.radians(2.0 / 60.0)
    freqs = np.array([1.4e9])
    prefix = str(tmp_path / "beam")
    write_beam_fits_cattery(beam, grid, grid, freqs, prefix, pol_basis="linear")

    _, providers, _ = resolve_antenna_beams(["MKAT-EA"], ["ALT-AZ"], {"MKAT-EA": {"cattery": prefix}})
    assert isinstance(providers[0], FitsBeamProvider)


def test_load_cattery_beam_json_and_resolve(tmp_path):
    import json

    beam_meerkat = CosineTaperBeam.from_builtin("MKAT-EA-L-JIM-2026")
    beam_ska = CosineTaperBeam.from_builtin("MKAT-AA-L-JIM-2020")
    npix = 17
    grid = (np.arange(npix) - npix // 2) * np.radians(2.0 / 60.0)
    freqs = np.array([1.4e9])
    write_beam_fits_cattery(beam_meerkat, grid, grid, freqs, str(tmp_path / "meerkat"), pol_basis="linear")
    write_beam_fits_cattery(beam_ska, grid, grid, freqs, str(tmp_path / "ska"), pol_basis="linear")

    cfg = {
        "lband": {
            "patterns": {"cmd::default": [str(tmp_path / "$(stype)_$(corr)_$(reim).fits")]},
            "define-stationtypes": {"cmd::default": "meerkat", "~SKA[0-9]{3}": "ska"},
        }
    }
    cfg_path = tmp_path / "beams.json"
    cfg_path.write_text(json.dumps(cfg))

    cattery_cfg = load_cattery_beam_json(cfg_path)
    names = ["M060", "M061", "SKA001", "SKA002"]
    mount = ["ALT-AZ"] * 4
    ant_type, providers, is_altaz = resolve_cattery_antenna_beams(names, mount, cattery_cfg)

    # M060/M061 share a provider (meerkat), SKA001/SKA002 share a different one (ska).
    assert ant_type[0] == ant_type[1]
    assert ant_type[2] == ant_type[3]
    assert ant_type[0] != ant_type[2]
    assert np.all(is_altaz)

    ell, emm = np.array([0.001]), np.array([0.0005])
    got_meerkat = providers[ant_type[0]].voltage(ell, emm, freqs, np.array([0.0]))
    got_ska = providers[ant_type[2]].voltage(ell, emm, freqs, np.array([0.0]))
    want_meerkat = JimBeamProvider(beam_meerkat).voltage(ell, emm, freqs, np.array([0.0]))
    want_ska = JimBeamProvider(beam_ska).voltage(ell, emm, freqs, np.array([0.0]))
    np.testing.assert_allclose(got_meerkat, want_meerkat, atol=2e-3)
    np.testing.assert_allclose(got_ska, want_ska, atol=2e-3)


def test_load_cattery_beam_json_rejects_multiple_patterns(tmp_path):
    import json

    cfg = {
        "lband": {
            "patterns": {"a": ["$(stype)_$(corr)_$(reim).fits"], "b": ["other_$(corr)_$(reim).fits"]},
            "define-stationtypes": {"cmd::default": "meerkat"},
        }
    }
    cfg_path = tmp_path / "beams.json"
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="distinct file patterns"):
        load_cattery_beam_json(cfg_path)


def test_resolve_cattery_antenna_beams_no_match_raises(tmp_path):
    beam = CosineTaperBeam.from_builtin("MKAT-EA-L-JIM-2026")
    npix = 9
    grid = (np.arange(npix) - npix // 2) * np.radians(2.0 / 60.0)
    write_beam_fits_cattery(beam, grid, grid, np.array([1.4e9]), str(tmp_path / "meerkat"), pol_basis="linear")
    cattery_cfg = {
        "stationtypes": [("~M0[0-9]{2}", True, "meerkat")],
        "pattern": str(tmp_path / "$(stype)_$(corr)_$(reim).fits"),
    }
    with pytest.raises(RuntimeError, match="No Cattery station type"):
        resolve_cattery_antenna_beams(["SKA001"], ["ALT-AZ"], cattery_cfg)


def test_fits_provider_from_cattery_reim_placeholder_no_double_suffix(tmp_path):
    # A pattern that contains $(reim) but no $(corr) placeholder must not have the
    # default _$(corr)_$(reim).fits suffix appended (the $(reim) token is enough).
    pattern = str(tmp_path / "beam_$(reim).fits")
    with pytest.raises(FileNotFoundError, match=r"beam_re\.fits"):
        FitsBeamProvider.from_cattery(pattern, pol_basis="linear")


def test_fits_provider_from_cattery_missing_file(tmp_path):
    pattern = str(tmp_path / "nowhere_$(corr)_$(reim).fits")
    with pytest.raises(FileNotFoundError, match="resolved from pattern"):
        FitsBeamProvider.from_cattery(pattern, pol_basis="linear")


def test_load_cattery_beam_json_rejects_non_object(tmp_path):
    import json

    cfg_path = tmp_path / "bad.json"
    cfg_path.write_text(json.dumps(["not", "a", "dict"]))
    with pytest.raises(ValueError, match="top-level JSON object"):
        load_cattery_beam_json(cfg_path)


# --- FITS-image approximate power beam (Phase 5) ---------------------------------

from simms.skymodel.beams import image_power_beam, pa_sample_grid  # noqa: E402
from simms.skymodel.fits_skies import FitsGrid, PreparedFitsSky, attach_image_beam  # noqa: E402

_IMG_RA0 = 0.0
_IMG_DEC0 = MKAT_LAT
_IMG_DUR = 3600.0


def test_image_beam_dft_components():
    freqs = np.array([1.35e9, 1.45e9])
    provider = JimBeamProvider(CosineTaperBeam.from_builtin("MKAT-AA-L-JIM-2020"))
    lmn = np.array([[0.0, 0.0, 0.0], [0.05, -0.03, np.sqrt(1 - 0.05**2 - 0.03**2) - 1]])
    prepared = PreparedFitsSky(
        chan_freqs=freqs,
        spectrum=None,
        backend="dft",
        ncorr=2,
        polarisation=False,
        linear_basis=True,
        ncomp=2,
        npix_l=0,
        npix_m=0,
        lmn=lmn,
        bmat=np.ones((2, 1, 2), dtype=np.complex128),
    )
    before = prepared.bmat.copy()
    attach_image_beam(
        prepared, provider, True, _IMG_RA0, _IMG_DEC0, MKAT_LON, MKAT_LAT, BASE_TIME, _IMG_DUR, 1.0, mid_freq=1.4e9
    )

    _, chi = pa_sample_grid(BASE_TIME, _IMG_DUR, _IMG_RA0, _IMG_DEC0, MKAT_LON, MKAT_LAT, 1.0)
    power = image_power_beam(provider, True, lmn[:, 0], lmn[:, 1], freqs, chi)  # (2, 2)
    np.testing.assert_allclose(prepared.bmat, before * power[:, None, :])
    # Boresight component less attenuated than the off-centre one.
    assert power[0].mean() > power[1].mean()
    assert power[1].mean() < 1.0


def test_image_beam_planes_use_mid_freq_for_single_plane():
    freqs = np.array([1.35e9, 1.45e9])  # 2 MS channels
    provider = JimBeamProvider(CosineTaperBeam.from_builtin("MKAT-AA-L-JIM-2020"))
    grid = FitsGrid(delta_l=0.01, delta_m=0.01, ref_l=2, ref_m=2, l_ref=0.0, m_ref=0.0, deviation_pixels=0.0)
    planes = np.ones((1, 5, 5, 1))  # single reference plane (FLAT/POLY)
    prepared = PreparedFitsSky(
        chan_freqs=freqs,
        spectrum=None,
        backend="fft",
        ncorr=2,
        polarisation=False,
        linear_basis=True,
        ncomp=25,
        npix_l=5,
        npix_m=5,
        planes=planes,
        grid=grid,
    )
    attach_image_beam(
        prepared, provider, True, _IMG_RA0, _IMG_DEC0, MKAT_LON, MKAT_LAT, BASE_TIME, _IMG_DUR, 1.0, mid_freq=1.4e9
    )

    ii, jj = (a.ravel() for a in np.meshgrid(np.arange(5), np.arange(5), indexing="ij"))
    lmn = grid.pixel_lmn(ii, jj)
    _, chi = pa_sample_grid(BASE_TIME, _IMG_DUR, _IMG_RA0, _IMG_DEC0, MKAT_LON, MKAT_LAT, 1.0)
    power = image_power_beam(provider, True, lmn[:, 0], lmn[:, 1], np.array([1.4e9]), chi)
    np.testing.assert_allclose(prepared.planes, power.reshape(5, 5, 1)[None])
    # Centre pixel (l = m = 0) attenuated least; corners most.
    assert prepared.planes[0, 2, 2, 0] > prepared.planes[0, 0, 0, 0]


# --- Review fixes: PA-grid cap and descending FITS axes --------------------------

from simms.skymodel.beams import MAX_PA_SAMPLES  # noqa: E402


def test_pa_sample_grid_caps_near_zenith_transit():
    # A field transiting through the zenith (dec == latitude) has a diverging PA rate;
    # the grid must be capped instead of blowing up to millions of samples.
    duration = 2 * 3600.0
    ra0 = float(local_sidereal_time(np.array([BASE_TIME + duration / 2]), MKAT_LON)[0])
    tgrid, chi_grid = pa_sample_grid(BASE_TIME, duration, ra0, MKAT_LAT, MKAT_LON, MKAT_LAT, 1.0)
    assert tgrid.size == chi_grid.size == MAX_PA_SAMPLES
    # A normal (well off-zenith) field stays far below the cap.
    tg2, _ = pa_sample_grid(BASE_TIME, duration, ra0, np.deg2rad(-55.0), MKAT_LON, MKAT_LAT, 1.0)
    assert tg2.size < 200


# --- Beam-grid memory ceiling (#140) ---------------------------------------------

from simms.skymodel.beams import (  # noqa: E402
    BEAM_GRID_MAX_GIB_DEFAULT,
    build_beam_grid,
    build_beam_grid_jones,
)


def _small_grid_args():
    providers = [UnityBeamProvider()]
    type_is_altaz = np.array([False])
    ell = np.array([0.0, 0.01, -0.02])
    emm = np.array([0.0, -0.01, 0.02])
    freqs = np.array([1.3e9, 1.4e9])
    chi_grid = np.zeros(4)
    return providers, type_is_altaz, ell, emm, freqs, chi_grid


def test_build_beam_grid_raises_above_ceiling():
    providers, type_is_altaz, ell, emm, freqs, chi_grid = _small_grid_args()
    with pytest.raises(MemoryError, match="beam-pa-step"):
        build_beam_grid(providers, type_is_altaz, ell, emm, freqs, chi_grid, max_gib=1e-9)
    # The Jones grid (fold 4) is checked the same way.
    with pytest.raises(MemoryError, match="beam-grid-max-gib"):
        build_beam_grid_jones(
            providers, type_is_altaz, ell, emm, freqs, chi_grid, corr_basis_transform(False), max_gib=1e-9
        )


def test_build_beam_grid_warns_in_soft_band(caplog):
    providers, type_is_altaz, ell, emm, freqs, chi_grid = _small_grid_args()
    # 1 x 4 x 3 x 2 x 2 x 8 B = 384 B; a ceiling just above it puts us over half.
    max_gib = 5e-7
    with caplog.at_level("WARNING", logger="skysim"):
        grid = build_beam_grid(providers, type_is_altaz, ell, emm, freqs, chi_grid, max_gib=max_gib)
    assert grid.shape == (1, 4, 3, 2, 2)
    assert any("half" in r.message for r in caplog.records)


def test_build_beam_grid_default_limit_passes_quietly(caplog):
    providers, type_is_altaz, ell, emm, freqs, chi_grid = _small_grid_args()
    with caplog.at_level("WARNING", logger="skysim"):
        grid = build_beam_grid(providers, type_is_altaz, ell, emm, freqs, chi_grid, max_gib=BEAM_GRID_MAX_GIB_DEFAULT)
    assert grid.shape == (1, 4, 3, 2, 2)
    assert not caplog.records


def test_fits_provider_handles_descending_grid():
    # FITS L/M axes commonly descend (negative CDELT); the constructor must re-sort
    # ascending rather than let RegularGridInterpolator raise, with identical results.
    beam = CosineTaperBeam.from_builtin("MKAT-AA-L-JIM-2020")
    l_grid = np.linspace(-0.06, 0.06, 81)
    m_grid = np.linspace(-0.06, 0.06, 81)
    freqs = np.array([1.35e9, 1.45e9])
    cube = _jimbeam_cube(beam, l_grid, m_grid, freqs)

    asc = FitsBeamProvider.from_arrays(l_grid, m_grid, freqs, cube)
    desc = FitsBeamProvider.from_arrays(l_grid[::-1], m_grid[::-1], freqs, cube[::-1, ::-1])

    ell = np.array([0.0, 0.02, -0.03])
    emm = np.array([0.01, -0.02, 0.0])
    chi = np.array([0.0, 0.2])
    np.testing.assert_allclose(desc.voltage(ell, emm, freqs, chi), asc.voltage(ell, emm, freqs, chi), atol=1e-12)


def test_fits_provider_from_fits_negative_cdelt(tmp_path):
    from astropy.io import fits

    beam = CosineTaperBeam.from_builtin("MKAT-AA-L-JIM-2020")
    l_grid = np.linspace(-0.05, 0.05, 41)  # ascending physical l per pixel index
    m_grid = np.linspace(-0.05, 0.05, 41)
    freqs = np.array([1.4e9])
    cube = _jimbeam_cube(beam, l_grid, m_grid, freqs)
    hh = cube[..., 0].transpose(2, 1, 0)
    vv = cube[..., 1].transpose(2, 1, 0)
    data = np.stack([hh.real, hh.imag, vv.real, vv.imag], axis=0)

    # Negative CDELT1: pixel 0 holds the largest l, so l decreases with pixel index.
    hdr = fits.Header()
    hdr["CRPIX1"], hdr["CRVAL1"], hdr["CDELT1"] = 1, np.degrees(l_grid[-1]), -np.degrees(l_grid[1] - l_grid[0])
    hdr["CRPIX2"], hdr["CRVAL2"], hdr["CDELT2"] = 1, np.degrees(m_grid[0]), np.degrees(m_grid[1] - m_grid[0])
    hdr["CRPIX3"], hdr["CRVAL3"], hdr["CDELT3"] = 1, freqs[0], 1.0
    hdr["CRPIX4"], hdr["CRVAL4"], hdr["CDELT4"] = 1, 0, 1
    # Data pixel index runs opposite to physical l, so flip the l axis of the planes too.
    path = tmp_path / "neg_cdelt.fits"
    fits.PrimaryHDU(data=data[:, :, :, ::-1], header=hdr).writeto(path)

    prov = FitsBeamProvider.from_fits(path)  # must not raise
    jim = JimBeamProvider(beam)
    ell = np.array([0.0, 0.015])
    emm = np.array([0.0, -0.02])
    np.testing.assert_allclose(
        prov.voltage(ell, emm, freqs, np.array([0.0])),
        jim.voltage(ell, emm, freqs, np.array([0.0])),
        atol=2e-3,
    )


# --- Full 2x2 Jones (components) + circular basis --------------------------------

from simms.constants import C  # noqa: E402
from simms.skymodel.beams import corr_basis_transform  # noqa: E402


def _coherency_bmat(stokes, freqs):
    """Linear-feed 2x2 coherency for one source: bmat (1, 4, nchan) = [XX,XY,YX,YY]."""
    si, sq, su, sv = stokes
    nchan = freqs.size
    b = np.array([si + sq, su + 1j * sv, su - 1j * sv, si - sq], dtype=np.complex128)
    return np.tile(b[None, :, None], (1, 1, nchan))


def test_basis_transform_matches_brightness_convention():
    # S . B_linear . S^H must equal the circular coherency for random Stokes.
    S = corr_basis_transform(True)
    rng = np.random.default_rng(1)
    for _ in range(200):
        si, sq, su, sv = rng.normal(size=4)
        b_lin = np.array([[si + sq, su + 1j * sv], [su - 1j * sv, si - sq]])
        b_circ = np.array([[si + sv, sq + 1j * su], [sq - 1j * su, si - sv]])
        np.testing.assert_allclose(S @ b_lin @ S.conj().T, b_circ, atol=1e-12)


def test_full_jones_equals_diagonal_for_jimbeam():
    # JimBeam has no cross-pol, so full 2x2 Jones must reproduce the diagonal kernel.
    freqs = np.array([1.35e9, 1.45e9])
    config = {"MKAT-MA": {"jimbeam": "L"}}
    uvw = np.array([[120.0, 60.0, 5.0]])
    times = np.array([BASE_TIME])
    a1, a2 = np.array([0]), np.array([1])
    lmn = np.concatenate([_lmn(0.0, 0.0), _lmn(0.05, -0.03)])
    flux = np.array([2.0, 1.5])

    diag = _attach(_make_prepared(lmn, flux, freqs, 4), ["MKAT-MA", "MKAT-MA"], ["ALT-AZ", "ALT-AZ"], config, 4)
    full = _attach(
        _make_prepared(lmn, flux, freqs, 4),
        ["MKAT-MA", "MKAT-MA"],
        ["ALT-AZ", "ALT-AZ"],
        config,
        4,
        full_jones=True,
        basis_transform=corr_basis_transform(False),
    )
    vd = predict_block(diag, uvw, times=times, antenna1=a1, antenna2=a2)
    vf = predict_block(full, uvw, times=times, antenna1=a1, antenna2=a2)
    assert full.beam_full_jones and full.beam_grid.shape[-2:] == (2, 2)
    np.testing.assert_allclose(vd, vf, atol=1e-6)


def test_full_jones_leakage_matches_einsum_reference():
    # A cube with non-zero HV/VH produces cross-hands, matching V = E B E^H per source.
    beam = CosineTaperBeam.from_builtin("MKAT-AA-L-JIM-2020")
    l_grid = np.linspace(-0.08, 0.08, 81)
    m_grid = np.linspace(-0.08, 0.08, 81)
    freqs = np.array([1.4e9])
    diag = _jimbeam_cube(beam, l_grid, m_grid, freqs)  # (nl,nm,nfreq,2) = HH,VV (real)
    cube4 = np.zeros((*diag.shape[:3], 4), dtype=np.complex128)  # HH,HV,VH,VV
    cube4[..., 0] = diag[..., 0]
    cube4[..., 3] = diag[..., 1]
    cube4[..., 1] = 0.05 + 0.02j  # HV leakage
    cube4[..., 2] = 0.05 + 0.02j  # VH leakage
    prov = FitsBeamProvider.from_arrays(l_grid, m_grid, freqs, cube4)
    assert prov.has_leakage

    ant_type = np.array([0, 0])
    S = corr_basis_transform(False)
    ell, emm = 0.04, -0.02
    prepared = _make_prepared(_lmn(ell, emm), np.array([2.0]), freqs, 4)
    from simms.skymodel.mstools import attach_beam

    full = attach_beam(
        prepared,
        ant_type,
        [prov],
        np.array([True]),
        0.0,
        MKAT_LAT,
        MKAT_LON,
        MKAT_LAT,
        BASE_TIME,
        1.0,
        1.0,
        4,
        full_jones=True,
        basis_transform=S,
    )
    uvw = np.array([[150.0, 70.0, 8.0]])
    times = np.array([BASE_TIME])
    vf = predict_block(full, uvw, times=times, antenna1=np.array([0]), antenna2=np.array([1]))

    # einsum reference at the single PA sample used (duration -> pa_wt = 0).
    chi = parallactic_angle(times, 0.0, MKAT_LAT, MKAT_LON, MKAT_LAT)
    E = prov.jones(np.array([ell]), np.array([emm]), freqs, chi)[0, 0, 0]  # (2,2)
    B = np.array([[2.0, 0.0], [0.0, 2.0]])
    n = np.sqrt(1 - ell**2 - emm**2)
    base = (uvw[0, 0] * ell + uvw[0, 1] * emm + uvw[0, 2] * (n - 1)) * 2 * np.pi / C
    phasor = np.exp(1j * base * freqs[0])
    ref = (E @ B @ E.conj().T) * phasor
    np.testing.assert_allclose(vf[0, 0].reshape(2, 2), ref, atol=1e-4)
    assert np.abs(vf[0, 0, 1]) > 0  # cross-hands non-zero (leakage)


def test_full_jones_circular_equals_S_Vlin_SH():
    # A polarised source: circular visibilities equal S . V_linear . S^H.
    freqs = np.array([1.4e9])
    config = {"MKAT-MA": {"jimbeam": "L"}}
    uvw = np.array([[130.0, 55.0, 6.0]])
    times = np.array([BASE_TIME])
    a1, a2 = np.array([0]), np.array([1])
    lmn = _lmn(0.05, -0.02)

    def prep():
        p = _make_prepared(lmn, np.array([1.0]), freqs, 4)
        p.bmat[:] = _coherency_bmat((2.0, 0.4, 0.3, 0.5), freqs)  # I,Q,U,V
        return p

    lin = _attach(
        prep(),
        ["MKAT-MA", "MKAT-MA"],
        ["ALT-AZ", "ALT-AZ"],
        config,
        4,
        full_jones=True,
        basis_transform=corr_basis_transform(False),
    )
    circ = _attach(
        prep(),
        ["MKAT-MA", "MKAT-MA"],
        ["ALT-AZ", "ALT-AZ"],
        config,
        4,
        full_jones=True,
        basis_transform=corr_basis_transform(True),
    )
    vl = predict_block(lin, uvw, times=times, antenna1=a1, antenna2=a2)[0, 0].reshape(2, 2)
    vc = predict_block(circ, uvw, times=times, antenna1=a1, antenna2=a2)[0, 0].reshape(2, 2)
    S = corr_basis_transform(True)
    np.testing.assert_allclose(vc, S @ vl @ S.conj().T, atol=1e-5)


# --- POINTING.DIRECTION measure frame (#144) -------------------------------------

import os  # noqa: E402

from simms.skymodel.beams import read_pointing_centre  # noqa: E402

from . import InitTest  # noqa: E402


def _write_pointing_ms(base_dir, ra, dec, ref):
    """A minimal MS whose ``POINTING.DIRECTION`` carries the given measure ``Ref``.

    ``ref=None`` omits the MEASINFO keyword entirely (as older MSs may).
    """
    from casacore.tables import makearrcoldesc, maketabdesc, table

    ms = os.path.join(base_dir, "pnt.ms")
    pnt = os.path.join(ms, "POINTING")
    mt = table(ms, maketabdesc([makearrcoldesc("DUMMY", 0.0)]), nrow=1, ack=False)
    pt = table(pnt, maketabdesc([makearrcoldesc("DIRECTION", 0.0, ndim=2)]), nrow=1, ack=False)
    pt.putcell("DIRECTION", 0, np.array([[ra, dec]], dtype=float))  # (npoly=1, radec=2)
    if ref is not None:
        pt.putcolkeyword("DIRECTION", "MEASINFO", {"type": "direction", "Ref": ref})
    pt.flush()
    pt.close()
    mt.putkeyword("POINTING", f"Table: {pnt}")
    mt.flush()
    mt.close()
    return ms


def test_read_pointing_centre_j2000_fast_path():
    pytest.importorskip("casacore")
    it = InitTest()
    ms = _write_pointing_ms(it.random_named_directory(), 0.5, -0.7, "J2000")
    assert read_pointing_centre(ms, 1.0, 1.0) == pytest.approx((0.5, -0.7))


def test_read_pointing_centre_missing_measinfo_falls_through():
    # An absent MEASINFO keyword is treated as equatorial (legacy behaviour): the stored
    # direction is returned rather than the phase-centre fallback.
    pytest.importorskip("casacore")
    it = InitTest()
    ms = _write_pointing_ms(it.random_named_directory(), 0.3, -0.4, None)
    assert read_pointing_centre(ms, 1.0, 1.0) == pytest.approx((0.3, -0.4))


def test_read_pointing_centre_rejects_azel():
    pytest.importorskip("casacore")
    it = InitTest()
    ms = _write_pointing_ms(it.random_named_directory(), 0.5, -0.7, "AZEL")
    with pytest.raises(NotImplementedError, match="AZEL"):
        read_pointing_centre(ms, 1.0, 1.0)


def test_read_pointing_centre_warns_on_nonstandard_equatorial(caplog):
    pytest.importorskip("casacore")
    it = InitTest()
    ms = _write_pointing_ms(it.random_named_directory(), 0.5, -0.7, "B1950")
    with caplog.at_level("WARNING", logger="skysim"):
        centre = read_pointing_centre(ms, 1.0, 1.0)
    assert centre == pytest.approx((0.5, -0.7))
    assert any("B1950" in r.message for r in caplog.records)


# --- FIELD fallbacks for the beam centre (#175) ----------------------------------


def _write_field_ms(base_dir, pointing=None, field=None, field_ref=None, nrow=1):
    """A minimal MS with an optional ``POINTING`` table and chosen ``FIELD`` direction columns.

    ``pointing`` is an ``(ra, dec)`` pair or ``None`` (no POINTING table at all). ``field`` maps
    a FIELD column name to a per-row list of ``(ra, dec)`` pairs; a column absent from the mapping
    is absent from the table. ``field_ref`` sets the MEASINFO ``Ref`` on every FIELD column.
    """
    from casacore.tables import makearrcoldesc, maketabdesc, table

    ms = os.path.join(base_dir, "field.ms")
    mt = table(ms, maketabdesc([makearrcoldesc("DUMMY", 0.0)]), nrow=1, ack=False)
    if pointing is not None:
        pnt = os.path.join(ms, "POINTING")
        pt = table(pnt, maketabdesc([makearrcoldesc("DIRECTION", 0.0, ndim=2)]), nrow=1, ack=False)
        pt.putcell("DIRECTION", 0, np.array([list(pointing)], dtype=float))
        pt.putcolkeyword("DIRECTION", "MEASINFO", {"type": "direction", "Ref": "J2000"})
        pt.flush()
        pt.close()
        mt.putkeyword("POINTING", f"Table: {pnt}")
    field = field or {}
    fld = os.path.join(ms, "FIELD")
    cols = [makearrcoldesc(col, 0.0, ndim=2) for col in field] or [makearrcoldesc("DUMMY", 0.0)]
    ft = table(fld, maketabdesc(cols), nrow=nrow, ack=False)
    for col, rows in field.items():
        for row, radec in enumerate(rows):
            ft.putcell(col, row, np.array([list(radec)], dtype=float))  # (npoly=1, radec=2)
        if field_ref is not None:
            ft.putcolkeyword(col, "MEASINFO", {"type": "direction", "Ref": field_ref})
    ft.flush()
    ft.close()
    mt.putkeyword("FIELD", f"Table: {fld}")
    mt.flush()
    mt.close()
    return ms


def test_read_pointing_centre_prefers_pointing_over_field():
    pytest.importorskip("casacore")
    it = InitTest()
    ms = _write_field_ms(it.random_named_directory(), pointing=(0.5, -0.7), field={"REFERENCE_DIR": [(0.9, -0.1)]})
    assert read_pointing_centre(ms, 1.0, 1.0) == pytest.approx((0.5, -0.7))


def test_read_pointing_centre_falls_back_to_reference_dir():
    # The rephased-MS case: no POINTING, and PHASE_DIR (the fallback) has been moved away from
    # the pointing that REFERENCE_DIR still records.
    pytest.importorskip("casacore")
    it = InitTest()
    ms = _write_field_ms(it.random_named_directory(), field={"REFERENCE_DIR": [(0.3, -0.4)]})
    assert read_pointing_centre(ms, 1.0, 1.0) == pytest.approx((0.3, -0.4))


def test_read_pointing_centre_falls_back_to_delay_dir():
    pytest.importorskip("casacore")
    it = InitTest()
    ms = _write_field_ms(it.random_named_directory(), field={"DELAY_DIR": [(0.3, -0.4)]})
    assert read_pointing_centre(ms, 1.0, 1.0) == pytest.approx((0.3, -0.4))


def test_read_pointing_centre_skips_nonfinite_reference_dir():
    # Some writers leave REFERENCE_DIR unpopulated; preferring garbage over the next candidate
    # would be a regression. (0, 0) is *not* a sentinel -- it is a real sky position.
    pytest.importorskip("casacore")
    it = InitTest()
    ms = _write_field_ms(
        it.random_named_directory(),
        field={"REFERENCE_DIR": [(np.nan, np.nan)], "DELAY_DIR": [(0.3, -0.4)]},
    )
    assert read_pointing_centre(ms, 1.0, 1.0) == pytest.approx((0.3, -0.4))


def test_read_pointing_centre_field_row_follows_field_id():
    pytest.importorskip("casacore")
    it = InitTest()
    ms = _write_field_ms(it.random_named_directory(), field={"REFERENCE_DIR": [(0.3, -0.4), (0.8, -0.9)]}, nrow=2)
    assert read_pointing_centre(ms, 1.0, 1.0, field_id=1) == pytest.approx((0.8, -0.9))


def test_read_pointing_centre_skips_azel_field_column():
    # A horizontal frame disqualifies a FIELD candidate rather than raising: unlike
    # POINTING.DIRECTION, there is something further down the chain to fall back to.
    pytest.importorskip("casacore")
    it = InitTest()
    ms = _write_field_ms(it.random_named_directory(), field={"REFERENCE_DIR": [(0.3, -0.4)]}, field_ref="AZEL")
    assert read_pointing_centre(ms, 1.0, 1.0) == pytest.approx((1.0, 1.0))


def test_read_pointing_centre_warns_when_chain_reaches_phase_centre(caplog):
    pytest.importorskip("casacore")
    it = InitTest()
    ms = _write_field_ms(it.random_named_directory())
    with caplog.at_level("WARNING", logger="skysim"):
        centre = read_pointing_centre(ms, 1.0, 1.0)
    assert centre == pytest.approx((1.0, 1.0))
    assert any("phase centre as the beam centre" in r.message for r in caplog.records)
