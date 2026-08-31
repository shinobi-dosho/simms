"""End-to-end primary-beam test: telsim MS + ASCII sky + beam config through skysim.runit."""

import numpy as np
import pytest
from daskms import xds_from_ms

from simms.apps import skysim
from simms.telescope.generate_ms import create_ms

from . import InitTest, skysim_opts


def _opts(ms, ascii_sky, primary_beam=None, column="DATA"):
    """A minimal skysim opts object with sensible defaults for the ASCII path."""
    return skysim_opts(ms, ascii_sky=ascii_sky, primary_beam=primary_beam, column=column)


class _E2E(InitTest):
    def __init__(self):
        self.test_files = []
        self.ms = self.random_named_directory(suffix=".ms")
        # Heterogeneous subarray spanning the skamid 13.5/15 m boundary.
        create_ms(
            self.ms,
            telescope_name="skamid",
            pointing_direction=["J2000", "0h0m20s", "-30deg"],
            dtime=600,
            ntimes=4,
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
        # A single source ~1 deg south of the phase centre (near the L-band half power).
        self.sky = self.random_named_file(suffix=".txt")
        with open(self.sky, "w") as fh:
            fh.write("#format: name ra dec stokes_i\n")
            fh.write("S 0h0m20s -31d0m0s 4.0\n")
        self.beams = self.random_named_file(suffix=".yaml")
        with open(self.beams, "w") as fh:
            fh.write("MKAT-MA:\n  jimbeam: MKAT-MA-L-JIM-2026\nMKAT-EA:\n  jimbeam: MKAT-EA-L-JIM-2026\n")


@pytest.fixture
def e2e():
    return _E2E()


def test_beam_attenuates_and_differs_from_nobeam(e2e):
    skysim.runit(_opts(e2e.ms, e2e.sky, primary_beam=None, column="NOBEAM"))
    skysim.runit(_opts(e2e.ms, e2e.sky, primary_beam=e2e.beams, column="BEAM"))

    ds = xds_from_ms(e2e.ms)[0]
    nobeam = ds.NOBEAM.data.compute()
    beam = ds.BEAM.data.compute()

    # The beam changes the visibilities and attenuates an off-centre source.
    assert not np.allclose(beam, nobeam)
    assert np.abs(beam).mean() < np.abs(nobeam).mean()
    # Unpolarised source: cross-hand-free 2-corr stays finite and non-zero.
    assert np.all(np.isfinite(beam))
    assert np.abs(beam).max() > 0


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


def test_beam_follows_pointing_direction(e2e):
    # A source sitting exactly on the phase centre.
    from daskms import xds_from_table

    sky = e2e.random_named_file(suffix=".txt")
    with open(sky, "w") as fh:
        fh.write("#format: name ra dec stokes_i\n")
        fh.write("S 0h0m20s -30d0m0s 4.0\n")  # == the fixture's pointing/phase centre

    # Baseline: POINTING.DIRECTION == PHASE_DIR (as simms writes), so the source is on-axis.
    skysim.runit(_opts(e2e.ms, sky, primary_beam=e2e.beams, column="ALIGNED"))

    # Point the dishes 0.8 deg south of the phase centre; the same source is now off-axis.
    cur = np.asarray(xds_from_table(f"{e2e.ms}::POINTING")[0].DIRECTION.data[0, 0].compute())
    _set_pointing_direction(e2e.ms, float(cur[0]), float(cur[1]) - np.radians(0.8))
    skysim.runit(_opts(e2e.ms, sky, primary_beam=e2e.beams, column="OFFSET"))

    ds = xds_from_ms(e2e.ms)[0]
    aligned = np.abs(ds.ALIGNED.data.compute()).mean()
    offset = np.abs(ds.OFFSET.data.compute()).mean()
    # The on-phase-centre source is attenuated more once the dishes point elsewhere. Under the
    # old phase-centre logic the beam would ignore the pointing shift and the two would match.
    assert offset < 0.9 * aligned


def test_beam_rejects_nonlinear_correlations(e2e):
    # A circular-correlation MS must be refused: the beam's feed mapping is linear-only.
    ms = e2e.random_named_directory(suffix=".ms")
    create_ms(
        ms,
        telescope_name="skamid",
        pointing_direction=["J2000", "0h0m20s", "-30deg"],
        dtime=600,
        ntimes=2,
        start_freq="1420MHz",
        dfreq="4MHz",
        nchan=2,
        correlations=["RR", "LL"],
        row_chunks=100000,
        sefd=None,
        column="DATA",
        start_time="2025-03-06T20:00:00",
        smooth=None,
        fit_order=None,
        subarray_range=[60, 68],
    )
    with pytest.raises(RuntimeError, match="linear correlations"):
        skysim.runit(_opts(ms, e2e.sky, primary_beam=e2e.beams))


def test_primary_beam_ignored_without_sky_model(e2e):
    # --primary-beam with a noise-only run (no sky model) must be a no-op, not a crash.
    opts = _opts(e2e.ms, ascii_sky=None, primary_beam=e2e.beams, column="NOISEONLY")
    opts.sefd = 500.0
    skysim.runit(opts)  # must not raise (beam ignored, noise written)
    ds = xds_from_ms(e2e.ms)[0]
    assert np.any(ds.NOISEONLY.data.compute() != 0)


def test_full_jones_requires_four_correlations(e2e):
    # The fixture MS is 2-corr (XX, YY); full Jones needs all four.
    opts = _opts(e2e.ms, e2e.sky, primary_beam=e2e.beams)
    opts.beam_jones = "full"
    with pytest.raises(RuntimeError, match="4 correlations"):
        skysim.runit(opts)


def test_full_jones_circular_ms(e2e):
    # Full 2x2 Jones on a circular-correlation (RR,RL,LR,LL) MS: runs and attenuates.
    ms = e2e.random_named_directory(suffix=".ms")
    create_ms(
        ms,
        telescope_name="skamid",
        pointing_direction=["J2000", "0h0m20s", "-30deg"],
        dtime=600,
        ntimes=3,
        start_freq="1420MHz",
        dfreq="4MHz",
        nchan=2,
        correlations=["RR", "RL", "LR", "LL"],
        row_chunks=100000,
        sefd=None,
        column="DATA",
        start_time="2025-03-06T20:00:00",
        smooth=None,
        fit_order=None,
        subarray_range=[60, 68],
    )
    nb = _opts(ms, e2e.sky, primary_beam=None, column="NOBEAM")
    nb.pol_basis = "circular"
    skysim.runit(nb)
    bm = _opts(ms, e2e.sky, primary_beam=e2e.beams, column="BEAM")
    bm.beam_jones = "full"
    skysim.runit(bm)

    ds = xds_from_ms(ms)[0]
    nobeam = ds.NOBEAM.data.compute()
    beam = ds.BEAM.data.compute()
    assert np.all(np.isfinite(beam))
    assert not np.allclose(beam, nobeam)
    assert np.abs(beam).mean() < np.abs(nobeam).mean()


def test_fits_image_beam_accepts_circular_ms(e2e):
    # The FITS-image scalar power beam is basis-independent, so an RR/LL MS is accepted.
    from astropy.io import fits

    from .predict_fits_tests import make_header

    ms = e2e.random_named_directory(suffix=".ms")
    create_ms(
        ms,
        telescope_name="skamid",
        pointing_direction=["J2000", "1h0m0s", "-31deg"],  # matches make_header RA0/DEC0
        dtime=600,
        ntimes=3,
        start_freq="1420MHz",
        dfreq="4MHz",
        nchan=2,
        correlations=["RR", "LL"],
        row_chunks=100000,
        sefd=None,
        column="DATA",
        start_time="2025-03-06T20:00:00",
        smooth=None,
        fit_order=None,
        subarray_range=[60, 68],
    )
    npix = 256
    data = np.zeros((npix, npix), dtype=np.float32)
    data[npix // 2 - 90, npix // 2] = 1.0  # ~0.5 deg south of centre (near half power)
    img = e2e.random_named_file(suffix=".fits")
    fits.PrimaryHDU(data=data, header=make_header(npix, nstokes=1, nchan=1)).writeto(img)

    def run(column, primary_beam):
        opts = _opts(ms, ascii_sky=None, primary_beam=primary_beam, column=column)
        opts.fits_sky = img
        opts.pol_basis = "circular"
        skysim.runit(opts)

    run("NOBEAM", None)
    run("BEAM", e2e.beams)  # must not raise on a circular MS
    ds = xds_from_ms(ms)[0]
    nobeam = ds.NOBEAM.data.compute()
    beam = ds.BEAM.data.compute()
    assert np.all(np.isfinite(beam))
    assert not np.allclose(beam, nobeam)
    assert np.abs(beam).mean() <= np.abs(nobeam).mean()


def test_primary_beam_accepts_cattery_json(e2e):
    # A Cattery/DDFacet heterogeneous-beam json (typed by raw ANTENNA.NAME via a regex rule, not
    # TELESCOPE_NAME) must attenuate the same way as the equivalent simms beam-config YAML.
    import json

    from simms.skymodel.beams import CosineTaperBeam, write_beam_fits_cattery

    npix = 33
    grid = (np.arange(npix) - npix // 2) * np.radians(2.0 / 60.0)
    freqs = np.array([1.412e9, 1.424e9])  # spans the MS's 1420MHz +/- 4MHz channels
    meerkat_prefix = e2e.random_named_file(suffix="")
    ska_prefix = e2e.random_named_file(suffix="")
    write_beam_fits_cattery(
        CosineTaperBeam.from_builtin("MKAT-MA-L-JIM-2026"), grid, grid, freqs, meerkat_prefix, pol_basis="linear"
    )
    write_beam_fits_cattery(
        CosineTaperBeam.from_builtin("MKAT-EA-L-JIM-2026"), grid, grid, freqs, ska_prefix, pol_basis="linear"
    )
    for prefix in (meerkat_prefix, ska_prefix):
        for corr in ("xx", "xy", "yx", "yy"):
            for reim in ("re", "im"):
                e2e.test_files.append(f"{prefix}_{corr}_{reim}.fits")

    # define-stationtypes maps antenna NAME (here the skamid fixture's M060.. / SKA00.. split,
    # via a regex) to a $(stype) value; patterns substitutes it straight into the file prefix,
    # so the type value can just be the absolute beam-set prefix itself.
    cfg = {
        "lband": {
            "patterns": {"cmd::default": ["$(stype)_$(corr)_$(reim).fits"]},
            "define-stationtypes": {"cmd::default": meerkat_prefix, "~SKA[0-9]{3}": ska_prefix},
        }
    }
    cfg_json = e2e.random_named_file(suffix=".json")
    with open(cfg_json, "w") as fh:
        json.dump(cfg, fh)

    skysim.runit(_opts(e2e.ms, e2e.sky, primary_beam=cfg_json, column="CATTERYJSON"))
    skysim.runit(_opts(e2e.ms, e2e.sky, primary_beam=None, column="NOBEAM"))

    ds = xds_from_ms(e2e.ms)[0]
    beam = ds.CATTERYJSON.data.compute()
    nobeam = ds.NOBEAM.data.compute()
    assert np.all(np.isfinite(beam))
    assert not np.allclose(beam, nobeam)
    assert np.abs(beam).mean() < np.abs(nobeam).mean()


def test_configurable_telescope_name_column(e2e):
    # telsim writes the label to a custom column; skysim must read the same name, and
    # the default name (absent here) must fail clearly rather than infer the metadata.
    ms = e2e.random_named_directory(suffix=".ms")
    create_ms(
        ms,
        telescope_name="skamid",
        pointing_direction=["J2000", "0h0m20s", "-30deg"],
        dtime=600,
        ntimes=2,
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
        telescope_name_column="DISH_TYPE",
    )
    ok = _opts(ms, e2e.sky, primary_beam=e2e.beams, column="BEAM")
    ok.telescope_name_column = "DISH_TYPE"
    skysim.runit(ok)  # reads the custom column
    assert np.all(np.isfinite(xds_from_ms(ms)[0].BEAM.data.compute()))

    bad = _opts(ms, e2e.sky, primary_beam=e2e.beams, column="BEAM2")  # default TELESCOPE_NAME, absent
    with pytest.raises(RuntimeError, match="TELESCOPE_NAME"):
        skysim.runit(bad)


def test_missing_mount_column_fails_clearly(e2e):
    # Same rule as the telescope-name column: beam metadata is read, never inferred.
    pytest.importorskip("casacore")
    from casacore.tables import table

    with table(f"{e2e.ms}::ANTENNA", readonly=False, ack=False) as tab:
        tab.removecols("MOUNT")
    opts = _opts(e2e.ms, e2e.sky, primary_beam=e2e.beams, column="BEAM")
    with pytest.raises(RuntimeError, match="MOUNT"):
        skysim.runit(opts)
