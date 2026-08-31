"""Tests for the `simms primary-beam` utility (to-fits, tag-ms, apply, correct)."""

import logging
import os
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from daskms import xds_from_table

from simms.apps import primary_beam
from simms.skymodel.beams import CosineTaperBeam, FitsBeamProvider, JimBeamProvider
from simms.telescope.generate_ms import create_ms

from . import InitTest
from .predict_fits_tests import DEC0_DEG, RA0_DEG, make_header


def _opts(mode, **over):
    base = {
        "mode": mode,
        "beam_pattern": "MKAT-EA-L-JIM-2026",
        "beam_band": "L",
        "beam_pa_step": 1.0,
        "fits_format": "simms",
        "pol_basis": "linear",
        "beam_l_axis": "-X",
        "beam_m_axis": "Y",
        "ms": None,
        "fits_sky": None,
        "ascii_sky": None,
        "ascii_delimiter": None,
        "source_schema": None,
        "output": None,
        "telescope_name_column": "TELESCOPE_NAME",
        "label": None,
        "label_map": None,
        "from_layout": None,
        "pb_cutoff": 0.1,
        "field_id": 0,
        "spw_id": 0,
        "pixel_size": "2arcmin",
        "npix": 64,
        "start_freq": None,
        "chan_width": None,
        "nchan": None,
        "nworkers": 1,
        "log_level": "CRITICAL",
    }
    base.update(over)
    return SimpleNamespace(**base)


class _Fixtures(InitTest):
    def __init__(self):
        self.test_files = []
        # A small heterogeneous skamid subarray (M060..M063 + SKA001..SKA004).
        self.ms = self.random_named_directory(suffix=".ms")
        create_ms(
            self.ms,
            telescope_name="skamid",
            pointing_direction=["J2000", "1h0m0s", "-31deg"],  # matches make_header RA0/DEC0
            dtime=600,
            ntimes=3,
            start_freq="1420MHz",
            dfreq="4MHz",
            nchan=2,
            correlations=["XX", "YY"],
            row_chunks=100000,
            sefd=None,
            column="DATA",
            start_time="2025-03-06T20:00:00",
            smooth=None,
            fit_order=None,
            subarray_range=[60, 68],
        )


@pytest.fixture
def fx():
    return _Fixtures()


def test_to_fits_roundtrips_through_provider(fx):
    out = fx.random_named_file(suffix=".fits")
    # start 1300 MHz, 100 MHz channels x 3 -> [1.3, 1.4, 1.5] GHz; 1.4 GHz is a node, so the
    # round-trip has no frequency-interp error.
    primary_beam.runit(
        _opts(
            "to-fits",
            beam_pattern="MKAT-EA-L-JIM-2026",
            pixel_size="1arcmin",
            npix=128,
            start_freq="1300MHz",
            chan_width="100MHz",
            nchan=3,
            output=out,
        )
    )

    prov = FitsBeamProvider.from_fits(out)  # must reload
    jim = JimBeamProvider(CosineTaperBeam.from_builtin("MKAT-EA-L-JIM-2026"))
    # Off-beam-core points; bilinear interpolation off the 1' grid is accurate to a few 1e-3.
    ell, emm = np.array([0.0, 0.004]), np.array([0.0, -0.003])
    freqs = np.array([1.4e9])
    np.testing.assert_allclose(
        prov.voltage(ell, emm, freqs, np.array([0.0])),
        jim.voltage(ell, emm, freqs, np.array([0.0])),
        atol=3e-3,
    )


def _cattery_paths(prefix, labels):
    return {(corr, ri): f"{prefix}_{corr}_{ri}.fits" for corr in labels for ri in ("re", "im")}


def test_to_fits_cattery_writes_eight_files_and_matches_beam(fx):
    prefix = os.path.join(fx.random_named_directory(), "beam")
    beam_pattern = "MKAT-EA-L-JIM-2026"

    primary_beam.runit(
        _opts(
            "to-fits",
            beam_pattern=beam_pattern,
            fits_format="cattery",
            pixel_size="2arcmin",
            npix=32,
            start_freq="1300MHz",
            chan_width="100MHz",
            nchan=2,
            output=prefix,
        )
    )

    paths = _cattery_paths(prefix, ["xx", "xy", "yx", "yy"])
    for path in paths.values():
        assert os.path.exists(path)

    # xy/yx: the cosine-taper model has no leakage.
    for corr in ("xy", "yx"):
        for ri in ("re", "im"):
            assert np.all(fits.getdata(paths[(corr, ri)]) == 0.0)

    # Independent axis/value check: invert the WCS of xx_re/xx_im at a few pixels and
    # compare against CosineTaperBeam.voltages directly -- not the implementation's own
    # reshape/transpose.
    header = fits.getheader(paths[("xx", "re")])
    assert header["CDELT1"] < 0  # beam_l_axis="-X" default
    assert header["CDELT2"] > 0  # beam_m_axis="Y" default
    wcs = WCS(header)
    data_re = fits.getdata(paths[("xx", "re")])
    data_im = fits.getdata(paths[("xx", "im")])
    beam = CosineTaperBeam.from_builtin(beam_pattern)

    for i_l, j_m, k_f in [(5, 20, 0), (16, 16, 1), (30, 2, 1)]:
        l_axis_val, m_axis_val, freq_hz = wcs.wcs_pix2world([[i_l, j_m, k_f]], 0)[0]
        l_deg, m_deg = -l_axis_val, m_axis_val  # sign_l=-1 (default "-X"), sign_m=+1 (default "Y")
        expected = beam.voltages(np.array([l_deg]), np.array([m_deg]), np.array([freq_hz / 1e6]))[0, 0, 0]
        got = data_re[k_f, j_m, i_l] + 1j * data_im[k_f, j_m, i_l]
        assert got == pytest.approx(expected, abs=1e-6)


def test_to_fits_cattery_axis_sign_flags(fx):
    prefix = os.path.join(fx.random_named_directory(), "beam")
    primary_beam.runit(
        _opts(
            "to-fits",
            fits_format="cattery",
            npix=16,
            nchan=1,
            output=prefix,
            beam_l_axis="X",
            beam_m_axis="-Y",
        )
    )
    header = fits.getheader(f"{prefix}_xx_re.fits")
    assert header["CDELT1"] > 0
    assert header["CDELT2"] < 0


def test_to_fits_cattery_circular_basis(fx):
    prefix = os.path.join(fx.random_named_directory(), "beam")
    beam_pattern = "MKAT-EA-L-JIM-2026"

    primary_beam.runit(
        _opts(
            "to-fits",
            beam_pattern=beam_pattern,
            fits_format="cattery",
            pol_basis="circular",
            npix=16,
            nchan=1,
            start_freq="1400MHz",
            output=prefix,
        )
    )

    paths = _cattery_paths(prefix, ["rr", "rl", "lr", "ll"])
    for path in paths.values():
        assert os.path.exists(path)

    header = fits.getheader(paths[("rr", "re")])
    wcs = WCS(header)
    beam = CosineTaperBeam.from_builtin(beam_pattern)
    i_l, j_m, k_f = 10, 4, 0
    l_axis_val, m_axis_val, freq_hz = wcs.wcs_pix2world([[i_l, j_m, k_f]], 0)[0]
    l_deg, m_deg = -l_axis_val, m_axis_val
    hh, vv = beam.voltages(np.array([l_deg]), np.array([m_deg]), np.array([freq_hz / 1e6]))[0, 0]

    def _plane(corr):
        re = fits.getdata(paths[(corr, "re")])[k_f, j_m, i_l]
        im = fits.getdata(paths[(corr, "im")])[k_f, j_m, i_l]
        return re + 1j * im

    # corr_basis_transform is a single left-multiply E' = S @ diag(hh, vv) (see
    # build_beam_grid_jones), not the baseline-coherency S @ B @ S^H form.
    assert _plane("rr") == pytest.approx(hh / np.sqrt(2), abs=1e-6)
    assert _plane("rl") == pytest.approx(1j * vv / np.sqrt(2), abs=1e-6)
    assert _plane("lr") == pytest.approx(hh / np.sqrt(2), abs=1e-6)
    assert _plane("ll") == pytest.approx(-1j * vv / np.sqrt(2), abs=1e-6)


def test_to_fits_cattery_flags_ignored_warning_for_simms_format(fx, caplog):
    out = fx.random_named_file(suffix=".fits")
    with caplog.at_level(logging.WARNING):
        primary_beam.runit(
            _opts(
                "to-fits",
                fits_format="simms",
                beam_l_axis="X",
                npix=8,
                nchan=1,
                output=out,
                log_level="WARNING",
            )
        )
    assert any("only apply to --fits-format cattery" in r.message for r in caplog.records)


def test_tag_ms_scalar_and_layout_and_map(fx):
    col = "TELESCOPE_NAME"

    primary_beam.runit(_opts("tag-ms", ms=fx.ms, label="FOO"))
    tnames = np.asarray(xds_from_table(f"{fx.ms}::ANTENNA")[0][col].data.compute()).astype(str)
    assert set(tnames) == {"FOO"}

    primary_beam.runit(_opts("tag-ms", ms=fx.ms, from_layout="skamid"))
    tnames = np.asarray(xds_from_table(f"{fx.ms}::ANTENNA")[0][col].data.compute()).astype(str)
    assert set(tnames) == {"MKAT-MA", "MKAT-EA"}

    names = [str(x) for x in np.asarray(xds_from_table(f"{fx.ms}::ANTENNA")[0].NAME.data.compute()).astype(str)]
    mp = fx.random_named_file(suffix=".yaml")
    with open(mp, "w") as fh:
        fh.write("\n".join(f"{n}: T{i}" for i, n in enumerate(names)) + "\n")
    primary_beam.runit(_opts("tag-ms", ms=fx.ms, label_map=mp))
    tnames = np.asarray(xds_from_table(f"{fx.ms}::ANTENNA")[0][col].data.compute()).astype(str)
    assert set(tnames) == {f"T{i}" for i in range(len(names))}


def test_outputs_declare_both_passthrough_paths():
    # `tag-ms` mutates the MS in place and produces no new file, so the
    # echoed-back `ms` is the only handle a dependent step can chain onto;
    # `output` covers to-fits/apply/correct. Dropping either makes that mode
    # unwireable in a shinobi Recipe (or in dosho, which transcribes this
    # model) -- invisible here, a build-time AttributeError downstream.
    # `files` carries every written path, which `output` alone cannot for
    # cattery to-fits (one prefix, eight files).
    assert set(primary_beam.PrimaryBeamOutputs.model_fields) == {"ms", "output", "files"}


# --- the resolved-output contract -------------------------------------------------
#
# Each mode defaults its filename when --output is omitted, so the outputs model has to
# report the name the run *resolved*. Echoing the raw --output back handed a dependent
# step None in exactly the case where it could not have guessed the name itself.


@pytest.mark.parametrize(
    "mode,over,expected",
    [
        ("to-fits", {}, "beam.fits"),
        ("apply", {"fits": True}, "apparent.fits"),
        ("correct", {"fits": True}, "corrected.fits"),
        ("apply", {"ascii": True}, "apparent.txt"),
        ("correct", {"ascii": True}, "corrected.txt"),
    ],
)
def test_defaulted_output_name_is_reported(fx, monkeypatch, mode, over, expected):
    monkeypatch.chdir(fx.random_named_directory())
    opts = _opts(mode, npix=16, nchan=1)
    if over.get("fits"):
        opts.ms, opts.fits_sky = fx.ms, _write_image(fx)[0]
    elif over.get("ascii"):
        opts.ms, opts.ascii_sky = fx.ms, _write_ascii_sky(fx)

    res = primary_beam.runit(opts)

    assert res.output == expected  # not None, and not the raw --output
    assert os.path.exists(res.output)
    assert res.files == [res.output]


def test_explicit_output_is_reported_verbatim(fx):
    out = fx.random_named_file(suffix=".fits")
    res = primary_beam.runit(_opts("apply", ms=fx.ms, fits_sky=_write_image(fx)[0], output=out))
    assert res.output == out
    assert res.files == [out]


def test_cattery_reports_the_prefix_and_all_eight_files(fx, monkeypatch):
    # Cattery writes eight files from one prefix. `output` is the prefix because that is
    # what from_cattery and DDFacet's --Beam-FITSFile consume -- any single one of the
    # eight would not round-trip into a later --beam-pattern -- so assert it does.
    monkeypatch.chdir(fx.random_named_directory())
    res = primary_beam.runit(_opts("to-fits", fits_format="cattery", npix=16, nchan=1))

    assert res.output == "beam"
    assert sorted(res.files) == sorted(_cattery_paths("beam", ["xx", "xy", "yx", "yy"]).values())
    assert all(os.path.exists(f) for f in res.files)
    assert FitsBeamProvider.from_cattery(res.output) is not None


def test_cattery_reports_the_prefix_stripped_of_a_fits_suffix(fx):
    # to_fits strips a .fits suffix off the prefix before writing; the reported handle
    # has to be the stripped one, or it names beam.fits_xx_re.fits.
    prefix = os.path.join(fx.random_named_directory(), "beam")
    res = primary_beam.runit(_opts("to-fits", fits_format="cattery", npix=16, nchan=1, output=f"{prefix}.fits"))
    assert res.output == prefix
    assert all(os.path.exists(f) for f in res.files)


def test_tag_ms_reports_the_ms_and_writes_no_files(fx):
    res = primary_beam.runit(_opts("tag-ms", ms=fx.ms, label="FOO"))
    assert res.ms == fx.ms
    assert res.output is None
    assert res.files == []


def _write_ascii_sky(fx):
    path = fx.random_named_file(suffix=".txt")
    with open(path, "w") as fh:
        fh.write("#format: name ra dec stokes_i\n")
        fh.write("A 1h0m0s -31d0m0s 5.0\n")
    return path


def _write_image(fx, npix=256, off=90):
    data = np.zeros((npix, npix), dtype=np.float32)
    data[npix // 2, npix // 2] = 3.0  # centre source
    data[npix // 2 - off, npix // 2] = 2.0  # ~0.5 deg south source (within the beam)
    path = fx.random_named_file(suffix=".fits")
    fits.PrimaryHDU(data=data, header=make_header(npix, nstokes=1, nchan=1)).writeto(path)
    return path, (npix // 2, npix // 2), (npix // 2 - off, npix // 2)


def test_apply_then_correct_image_is_identity(fx):
    img, centre, offsrc = _write_image(fx)
    original = fits.getdata(img)

    apparent = fx.random_named_file(suffix=".fits")
    primary_beam.runit(_opts("apply", ms=fx.ms, fits_sky=img, output=apparent))
    app = fits.getdata(apparent)
    # Off-centre source is attenuated; centre source ~unchanged.
    assert app[offsrc] < original[offsrc]
    assert app[offsrc] > 0

    recovered = fx.random_named_file(suffix=".fits")
    primary_beam.runit(_opts("correct", ms=fx.ms, fits_sky=apparent, output=recovered))
    rec = fits.getdata(recovered)
    # Round-trip recovers the source fluxes (both are inside the beam).
    np.testing.assert_allclose(rec[centre], original[centre], rtol=1e-4)
    np.testing.assert_allclose(rec[offsrc], original[offsrc], rtol=1e-4)
    # Corners (beam below cutoff) are blanked by correct.
    assert np.isnan(rec[0, 0])


def test_apply_centres_beam_on_pointing_centre(fx, caplog):
    # Image reference pixel offset 0.4 deg north of the pointing centre (dec -31). The primary
    # beam belongs to the antenna pointing centre (POINTING.DIRECTION), not the image reference.
    npix = 256
    header = make_header(npix, nstokes=1, nchan=1, crval2=DEC0_DEG + 0.4)
    img = fx.random_named_file(suffix=".fits")
    fits.PrimaryHDU(data=np.ones((npix, npix), np.float32), header=header).writeto(img)

    out = fx.random_named_file(suffix=".fits")
    with caplog.at_level(logging.WARNING):
        primary_beam.runit(_opts("apply", ms=fx.ms, fits_sky=img, output=out, log_level="WARNING"))
    assert any("pointing centre" in r.message for r in caplog.records)

    # Uniform input -> the apparent image *is* the power beam A(l, m). It must peak on the
    # pointing centre, not on the image reference pixel (which is 0.4 deg off it).
    app = fits.getdata(out)
    ((col, row),) = WCS(header).celestial.wcs_world2pix([[RA0_DEG, DEC0_DEG]], 0)
    pc = (int(round(row)), int(round(col)))  # numpy [dec, ra]
    ref = (npix // 2, npix // 2)  # image reference pixel
    assert app[pc] > app[ref]  # would fail if the beam were centred on the image reference
    assert app[pc] > 0.9  # ~on-axis at the pointing centre


def _set_pointing_direction(ms, ra_rad, dec_rad):
    """Overwrite POINTING.DIRECTION so it differs from FIELD.PHASE_DIR (which simms keeps equal)."""
    import dask
    import dask.array as da
    from daskms import xds_from_table, xds_to_table

    pnt = xds_from_table(f"{ms}::POINTING")[0]
    nrow = pnt.DIRECTION.shape[0]
    newdir = np.broadcast_to(np.array([ra_rad, dec_rad]), (nrow, 1, 2)).copy()
    pnt = pnt.assign(DIRECTION=(("row", "point-poly", "radec"), da.from_array(newdir, chunks=(nrow, 1, 2))))
    dask.compute(xds_to_table([pnt], f"{ms}::POINTING", columns=["DIRECTION"]))


def test_apply_uses_pointing_direction_not_phase_centre(fx):
    # Point the dishes 0.3 deg north of the phase centre; the image WCS stays on the phase centre.
    _set_pointing_direction(fx.ms, np.radians(RA0_DEG), np.radians(DEC0_DEG + 0.3))

    npix = 256
    header = make_header(npix, nstokes=1, nchan=1)  # reference == phase centre (dec -31)
    img = fx.random_named_file(suffix=".fits")
    fits.PrimaryHDU(data=np.ones((npix, npix), np.float32), header=header).writeto(img)

    out = fx.random_named_file(suffix=".fits")
    primary_beam.runit(_opts("apply", ms=fx.ms, fits_sky=img, output=out))

    # The beam must follow POINTING.DIRECTION (dec -30.7), not FIELD.PHASE_DIR (dec -31).
    app = fits.getdata(out)
    wcs = WCS(header).celestial
    ((pc_col, pc_row),) = wcs.wcs_world2pix([[RA0_DEG, DEC0_DEG + 0.3]], 0)  # pointing pixel
    pnt_pix = (int(round(pc_row)), int(round(pc_col)))
    phase_pix = (npix // 2, npix // 2)  # phase centre / image reference pixel
    assert app[pnt_pix] > app[phase_pix]  # follows POINTING, not PHASE_DIR
    assert app[pnt_pix] > 0.9


def test_apply_then_correct_ascii_is_identity(fx):
    sky = fx.random_named_file(suffix=".txt")
    with open(sky, "w") as fh:
        fh.write("#format: name ra dec stokes_i\n")
        fh.write("A 1h0m0s -31d0m0s 5.0\n")  # at phase centre
        fh.write("\n")  # blank line between sources
        fh.write("# a comment between sources -- lineno mapping must skip it\n")
        fh.write("B 1h0m0s -31d30m0s 3.0\n")  # ~0.5 deg off

    apparent = fx.random_named_file(suffix=".txt")
    primary_beam.runit(_opts("apply", ms=fx.ms, ascii_sky=sky, output=apparent))
    app_flux = _read_ascii_flux(apparent)
    assert app_flux["A"] == pytest.approx(5.0, rel=1e-3)  # on-axis ~unattenuated
    assert app_flux["B"] < 3.0  # off-axis attenuated

    recovered = fx.random_named_file(suffix=".txt")
    primary_beam.runit(_opts("correct", ms=fx.ms, ascii_sky=apparent, output=recovered))
    rec_flux = _read_ascii_flux(recovered)
    assert rec_flux["A"] == pytest.approx(5.0, rel=1e-3)
    assert rec_flux["B"] == pytest.approx(3.0, rel=1e-3)


def _read_ascii_flux(path, delimiter=None):
    flux = {}
    with open(path) as fh:
        for ln in fh:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            f = s.split(delimiter)
            flux[f[0]] = float(f[3])  # name ra dec stokes_i
    return flux


def test_apply_then_correct_ascii_custom_schema(fx):
    # A CSV sky model whose columns are renamed via a custom source schema;
    # apply/correct must honour --source-schema and --ascii-delimiter, and find
    # the flux column through its alias.
    schema_yaml = "\n".join(
        [
            "info: Aliased schema",
            "parameters:",
            "  name: {info: Name, alias: NAME, units: null, ptype: string}",
            "  ra: {info: RA, alias: RA, units: deg, ptype: longitude, required: true}",
            "  dec: {info: Dec, alias: DEC, units: deg, ptype: latitude, required: true}",
            "  stokes_i: {info: Stokes I, alias: I, units: Jy, ptype: flux, required: true}",
        ]
    )
    schema = fx.random_named_file(suffix=".yaml")
    with open(schema, "w") as fh:
        fh.write(schema_yaml + "\n")

    sky = fx.random_named_file(suffix=".csv")
    with open(sky, "w") as fh:
        fh.write("#format: NAME,RA,DEC,I\n")
        fh.write("A,15.0,-31.0,5.0\n")  # at phase centre (1h0m0s -31d)
        fh.write("B,15.0,-31.5,3.0\n")  # ~0.5 deg off

    apparent = fx.random_named_file(suffix=".csv")
    primary_beam.runit(
        _opts("apply", ms=fx.ms, ascii_sky=sky, ascii_delimiter=",", source_schema=schema, output=apparent)
    )
    app_flux = _read_ascii_flux(apparent, delimiter=",")
    assert app_flux["A"] == pytest.approx(5.0, rel=1e-3)  # on-axis ~unattenuated
    assert app_flux["B"] < 3.0  # off-axis attenuated

    recovered = fx.random_named_file(suffix=".csv")
    primary_beam.runit(
        _opts("correct", ms=fx.ms, ascii_sky=apparent, ascii_delimiter=",", source_schema=schema, output=recovered)
    )
    rec_flux = _read_ascii_flux(recovered, delimiter=",")
    assert rec_flux["A"] == pytest.approx(5.0, rel=1e-3)
    assert rec_flux["B"] == pytest.approx(3.0, rel=1e-3)


def _set_mount(ms, value):
    """Overwrite every ANTENNA.MOUNT entry (the layouts simms ships are all ALT-AZ)."""
    from casacore.tables import table

    with table(f"{ms}::ANTENNA", readonly=False, ack=False) as tab:
        tab.putcol("MOUNT", [value] * tab.nrows())


def test_averaged_beam_honours_the_antenna_mount(fx):
    # An equatorial mount does not carry parallactic rotation into the feed frame, so the
    # averaged beam must be the chi=0 beam and not a smear of the squinted pattern over
    # the PA track. apply/correct used to average unconditionally.
    pytest.importorskip("casacore")
    from simms.skymodel import pb_ops
    from simms.skymodel.beams import image_power_beam, resolve_beam

    provider = resolve_beam("MKAT-EA-L-JIM-2026", "L")
    ell = np.array([0.0, 0.010, 0.018])
    emm = np.array([0.0, 0.012, -0.015])

    obs_altaz = pb_ops._observation(fx.ms, 0, 0)
    assert obs_altaz["is_altaz"]  # the skamid layout is ALT-AZ
    got_altaz = pb_ops._averaged_beam(provider, ell, emm, obs_altaz["ra0"], obs_altaz["dec0"], obs_altaz, 1.0)

    _set_mount(fx.ms, "EQUATORIAL")
    obs_eq = pb_ops._observation(fx.ms, 0, 0)
    assert not obs_eq["is_altaz"]
    got_eq = pb_ops._averaged_beam(provider, ell, emm, obs_eq["ra0"], obs_eq["dec0"], obs_eq, 1.0)

    want_eq = image_power_beam(provider, False, ell, emm, obs_eq["freqs"], np.zeros(1)).mean(axis=1)
    np.testing.assert_allclose(got_eq, want_eq, rtol=1e-12)
    # Off-axis, the two answers genuinely disagree -- the mount is load-bearing.
    assert not np.allclose(got_eq[1:], got_altaz[1:])


def test_averaged_beam_warns_on_mixed_mounts(fx, caplog):
    pytest.importorskip("casacore")
    from casacore.tables import table

    from simms.skymodel import pb_ops

    with table(f"{fx.ms}::ANTENNA", readonly=False, ack=False) as tab:
        mounts = ["ALT-AZ"] * tab.nrows()
        mounts[-1] = "EQUATORIAL"
        tab.putcol("MOUNT", mounts)
    with caplog.at_level(logging.WARNING, logger="primary-beam"):
        obs = pb_ops._observation(fx.ms, 0, 0)
    assert obs["is_altaz"]  # follows the first antenna
    assert any("mixes rotating and non-rotating mounts" in r.message for r in caplog.records)


def test_missing_mount_column_fails_clearly(fx):
    # MOUNT decides whether the beam rotates. Absent, there is nothing to infer it from --
    # assuming either way is silently wrong -- so say what is missing instead of letting
    # an AttributeError out of daskms.
    pytest.importorskip("casacore")
    from casacore.tables import table

    from simms.skymodel import pb_ops

    with table(f"{fx.ms}::ANTENNA", readonly=False, ack=False) as tab:
        tab.removecols("MOUNT")
    with pytest.raises(RuntimeError, match="MOUNT"):
        pb_ops._observation(fx.ms, 0, 0)
