"""Primary-beam models for skysim.

The cosine-taper ("JimBeam") voltage model in this module is re-implemented from
katbeam (https://github.com/ska-sa/katbeam), Copyright (c) 2020, National Research
Foundation (SARAO), released under the BSD 3-Clause License, originally authored by
Mattieu de Villiers. We vendor the maths and the bundled L/UHF coefficient tables
(``simms/skymodel/beam_data/``) rather than depend on the package so that we can
(a) drop an unmaintained dependency and (b) load arbitrary coefficient CSVs -- e.g.
the MeerKAT-Extension ``MKAT-EA-*`` tables -- which upstream katbeam cannot.

A cosine aperture taper (Essential Radio Astronomy, Condon & Ransom, 2016) models
the co-polarised primary beam. It is parameterised, per frequency and per feed
(H/V), by a pointing offset ("squint", ``l0``/``m0``) and a full-width-half-maximum
in each of the two feed axes. ``HH``/``VV`` are *voltage* patterns: they peak at 1
on axis and equal ``sqrt(0.5)`` at the half-power point, so ``|HH|**2`` is the power
beam and Stokes I is ``0.5*(|HH|**2 + |VV|**2)``.
"""

from __future__ import annotations

import csv
import logging
from importlib import resources
from pathlib import Path

import numpy as np

from simms import BIN
from simms.utilities import load_yaml

log = logging.getLogger(BIN.skysim)

# Seconds per day; MS TIME is in seconds (MJD * 86400), casacore convention.
SECONDS_PER_DAY = 86400.0


def local_sidereal_time(times: np.ndarray, lon: float) -> np.ndarray:
    """Mean local sidereal time (radians) for MS ``TIME`` seconds at longitude ``lon`` (rad).

    Used to form the *local* hour angle ``LST - RA`` that sets the sky orientation
    (parallactic angle, elevation) at the site. This is a distinct quantity from the
    Greenwich hour angle that :func:`simms.telescope.array_utilities.Array.uvgen` uses
    for the UVW rotation of its ECEF baselines -- the two differ by the array longitude
    and both are correct in their own frame.
    """
    from astropy import units as u
    from astropy.time import Time

    t = Time(np.atleast_1d(np.asarray(times, dtype=np.float64)) / SECONDS_PER_DAY, format="mjd", scale="utc")
    return t.sidereal_time("mean", longitude=lon * u.rad).to_value("rad")


def parallactic_angle(times: np.ndarray, ra0: float, dec0: float, lon: float, lat: float) -> np.ndarray:
    """Parallactic angle (radians) at the field centre for each timestamp.

    Parameters
    ----------
    times : numpy.ndarray
        MS ``TIME`` values in seconds, shape ``(ntime,)``.
    ra0, dec0 : float
        Field-centre right ascension and declination (radians).
    lon, lat : float
        Array-reference geodetic longitude and latitude (radians).

    Returns
    -------
    numpy.ndarray
        Parallactic angle per timestamp, shape ``(ntime,)``. This is the position
        angle of the zenith at the field centre; alt-az feeds rotate the beam on the
        sky by this angle.
    """
    hour_angle = local_sidereal_time(times, lon) - ra0
    return np.arctan2(
        np.sin(hour_angle),
        np.tan(lat) * np.cos(dec0) - np.sin(dec0) * np.cos(hour_angle),
    )


# r_FWHM: normalises the taper so the half-power point falls at r = 0.5, i.e.
# |taper(0.5)|**2 = 0.5. katbeam truncates this to 1.18896478.
R_FWHM = 1.1889647809329453

# Bundled coefficient tables (see beam_data/NOTICE). The MKAT-AA-* models are
# vendored from katbeam (BSD-3-Clause, (c) 2020 SARAO); the rest ship as package data.
BUILTIN_BEAMS = {
    "MKAT-AA-L-JIM-2020": "MKAT-AA-L-JIM-2020.csv",
    "MKAT-AA-UHF-JIM-2020": "MKAT-AA-UHF-JIM-2020.csv",
    "MKAT-MA-L-JIM-2026": "MKAT-MA-L-JIM-2026.csv",  # MeerKAT, L band
    "MKAT-EA-L-JIM-2026": "MKAT-EA-L-JIM-2026.csv",  # MeerKAT Extension, band 2
    "MKAT-MA-S-JIM-2020": "MKAT-MA-S-JIM-2020.csv",  # MeerKAT, S band
    "MKAT-EA-S3-JIM-2026": "MKAT-EA-S3-JIM-2026.csv",  # MeerKAT Extension, S band (S3)
}


def cosine_taper(r: np.ndarray) -> np.ndarray:
    """Cosine-taper voltage pattern ``cos(pi*rr)/(1 - 4*rr**2)`` with ``rr = r*R_FWHM``.

    ``r`` is normalised so the half-power point is at ``r = 0.5``. The ``rr = 0.5``
    singularity of ``1 - 4*rr**2`` is removable (limit ``pi/4``) and handled here.
    """
    rr = np.asarray(r, dtype=np.float64) * R_FWHM
    denom = 1.0 - 4.0 * rr * rr
    near = np.abs(denom) < 1e-12
    safe = np.where(near, 1.0, denom)
    out = np.cos(np.pi * rr) / safe
    return np.where(near, np.pi / 4.0, out)


class CosineTaperBeam:
    """Frequency-interpolated cosine-taper voltage beam for the H and V feeds.

    Parameters
    ----------
    freqs_mhz : numpy.ndarray
        Tabulated frequencies (MHz), shape ``(nfreq,)``.
    squint_deg : numpy.ndarray
        Per-feed pointing offsets in degrees, shape ``(4, nfreq)`` ordered
        ``Hx, Hy, Vx, Vy``.
    fwhm_deg : numpy.ndarray
        Per-feed FWHMs in degrees, same shape/ordering as ``squint_deg``.
    name : str, optional
        Label for diagnostics.
    """

    def __init__(self, freqs_mhz, squint_deg, fwhm_deg, name: str = ""):
        self.freqs_mhz = np.ascontiguousarray(freqs_mhz, dtype=np.float64)
        self.squint_deg = np.ascontiguousarray(squint_deg, dtype=np.float64)
        self.fwhm_deg = np.ascontiguousarray(fwhm_deg, dtype=np.float64)
        if self.squint_deg.shape != (4, self.freqs_mhz.size):
            raise ValueError("squint_deg must have shape (4, nfreq)")
        if self.fwhm_deg.shape != (4, self.freqs_mhz.size):
            raise ValueError("fwhm_deg must have shape (4, nfreq)")
        self.name = name
        self._warned_out_of_band = False

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_csv(cls, path, name: str = "") -> CosineTaperBeam:
        """Load a katbeam-format coefficient CSV.

        The file has two header rows (column names, then units) followed by rows of
        ``freq[MHz], Hx, Hy, Vx, Vy squint[arcmin], Hx, Hy, Vx, Vy fwhm[arcmin]``.
        """
        path = Path(path)
        with open(path, newline="") as fh:
            rows = [row for row in csv.reader(fh) if row and row[0].strip()]
        # Drop the two header rows (name row, units row).
        table = np.array([[float(c) for c in row] for row in rows[2:]], dtype=np.float64)
        freqs_mhz = table[:, 0]
        squint_deg = table[:, 1:5].T / 60.0  # arcmin -> degrees
        fwhm_deg = table[:, 5:9].T / 60.0
        return cls(freqs_mhz, squint_deg, fwhm_deg, name=name or path.stem)

    @classmethod
    def from_builtin(cls, name: str) -> CosineTaperBeam:
        """Load one of the bundled models in :data:`BUILTIN_BEAMS`."""
        if name not in BUILTIN_BEAMS:
            raise ValueError(f"Unknown built-in beam {name!r}; available: {list(BUILTIN_BEAMS)}")
        with resources.as_file(resources.files("simms.skymodel.beam_data") / BUILTIN_BEAMS[name]) as p:
            return cls.from_csv(p, name=name)

    @classmethod
    def load(cls, spec: str) -> CosineTaperBeam:
        """Load by built-in name if known, otherwise treat ``spec`` as a CSV path."""
        if spec in BUILTIN_BEAMS:
            return cls.from_builtin(spec)
        return cls.from_csv(spec)

    # -- evaluation ---------------------------------------------------------

    def _warn_out_of_band(self, freqs_mhz: np.ndarray) -> None:
        """Warn once if ``freqs_mhz`` reaches outside the tabulated band.

        :func:`numpy.interp` clamps rather than extrapolating, so a frequency past either
        end silently returns the edge row's squint and FWHM -- an L-band-width beam on UHF
        data, say, with nothing in the output to show for it. The clamp is the right
        behaviour (extrapolating a fitted FWHM is worse); the silence is not.
        """
        if self._warned_out_of_band:
            return
        lo, hi = float(self.freqs_mhz.min()), float(self.freqs_mhz.max())
        atol = 1e-6 * hi  # a channel sitting on the edge is not extrapolation
        want_lo, want_hi = float(freqs_mhz.min()), float(freqs_mhz.max())
        if want_lo < lo - atol or want_hi > hi + atol:
            self._warned_out_of_band = True
            log.warning(
                "Beam model %r is tabulated over %.1f-%.1f MHz but is being evaluated over "
                "%.1f-%.1f MHz; coefficients outside the table are held at its edge values, so "
                "the beam width there is the edge width, not the true one. Use a beam table "
                "covering this band (--beam-band, or a matching CSV).",
                self.name or "<unnamed>",
                lo,
                hi,
                want_lo,
                want_hi,
            )

    def _interp(self, freqs_mhz: np.ndarray):
        """Linearly interpolate squint/FWHM to ``freqs_mhz`` -> two ``(4, nchan)`` arrays.

        Clamped outside the tabulated band; :meth:`_warn_out_of_band` says so when it happens.
        """
        freqs_mhz = np.atleast_1d(np.asarray(freqs_mhz, dtype=np.float64))
        self._warn_out_of_band(freqs_mhz)
        squint = np.stack([np.interp(freqs_mhz, self.freqs_mhz, s) for s in self.squint_deg])
        fwhm = np.stack([np.interp(freqs_mhz, self.freqs_mhz, f) for f in self.fwhm_deg])
        return squint, fwhm

    def voltages(self, x_deg: np.ndarray, y_deg: np.ndarray, freqs_mhz: np.ndarray) -> np.ndarray:
        """Voltage patterns for both feeds.

        Parameters
        ----------
        x_deg, y_deg : numpy.ndarray
            Feed-frame coordinates in degrees (``x`` toward +AZ, ``y`` toward +EL),
            shape ``(nsrc,)``.
        freqs_mhz : numpy.ndarray
            Frequencies in MHz, shape ``(nchan,)``.

        Returns
        -------
        numpy.ndarray
            Real voltage patterns of shape ``(nsrc, nchan, 2)``; the last axis is
            feed 0 = H/X, 1 = V/Y.
        """
        x = np.asarray(x_deg, dtype=np.float64)[:, None]
        y = np.asarray(y_deg, dtype=np.float64)[:, None]
        squint, fwhm = self._interp(freqs_mhz)  # (4, nchan)
        sq = squint[:, None, :]  # (4, 1, nchan)
        fw = fwhm[:, None, :]
        rh = np.sqrt(((x - sq[0]) / fw[0]) ** 2 + ((y - sq[1]) / fw[1]) ** 2)
        rv = np.sqrt(((x - sq[2]) / fw[2]) ** 2 + ((y - sq[3]) / fw[3]) ** 2)
        return np.stack([cosine_taper(rh), cosine_taper(rv)], axis=-1)


# Built-in analytic models selectable by band shorthand in a beam-config entry.
BAND_BUILTINS = {
    "L": "MKAT-AA-L-JIM-2020",
    "UHF": "MKAT-AA-UHF-JIM-2020",
}


class BeamProvider:
    """Evaluate a per-feed voltage beam, de-rotating the sky into the feed frame.

    Subclasses implement :meth:`_eval` in the feed frame. The base handles the
    parallactic-angle rotation so a source at sky direction cosines ``(l, m)`` is
    evaluated at the feed-frame coordinates ``R(-chi) . (l, m)``.
    """

    def _eval(self, l_feed: np.ndarray, m_feed: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Feed-frame voltages ``(nsrc, nchan, 2)`` (feed 0 = H/X, 1 = V/Y)."""
        raise NotImplementedError

    def _eval_jones(self, l_feed: np.ndarray, m_feed: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Feed-frame 2x2 voltage Jones ``(nsrc, nchan, 2, 2)``.

        Default is the leakage-free diagonal ``diag(g^H, g^V)`` from :meth:`_eval`;
        providers with cross-polarisation (measured FITS cubes) override this.
        """
        g = self._eval(l_feed, m_feed, freqs)  # (nsrc, nchan, 2)
        jones = np.zeros((*g.shape[:2], 2, 2), dtype=np.complex128)
        jones[..., 0, 0] = g[..., 0]
        jones[..., 1, 1] = g[..., 1]
        return jones

    @staticmethod
    def _rotate_to_feed(ell, emm, angle):
        """Rotate sky cosines ``(l, m)`` into the feed frame by ``-angle`` (identity at 0)."""
        if angle:
            c, s = np.cos(angle), np.sin(angle)
            return ell * c + emm * s, -ell * s + emm * c
        return ell, emm

    def jones(self, ell, emm, freqs, chi) -> np.ndarray:
        """2x2 voltage Jones per parallactic-angle sample.

        Same inputs as :meth:`voltage`; returns ``(ntime, nsrc, nchan, 2, 2)`` complex.
        """
        ell = np.asarray(ell, dtype=np.float64)
        emm = np.asarray(emm, dtype=np.float64)
        freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
        chi = np.atleast_1d(np.asarray(chi, dtype=np.float64))
        out = np.empty((chi.size, ell.size, freqs.size, 2, 2), dtype=np.complex128)
        for ti, angle in enumerate(chi):
            l_feed, m_feed = self._rotate_to_feed(ell, emm, angle)
            out[ti] = self._eval_jones(l_feed, m_feed, freqs)
        return out

    def voltage(self, ell, emm, freqs, chi) -> np.ndarray:
        """Voltage beam for each parallactic angle.

        Parameters
        ----------
        ell, emm : numpy.ndarray
            Sky-frame direction cosines ``(l, m)``, shape ``(nsrc,)``.
        freqs : numpy.ndarray
            Frequencies in Hz, shape ``(nchan,)``.
        chi : numpy.ndarray
            Parallactic angle per sample (radians), shape ``(ntime,)``. Pass zeros
            for a non-rotating (e.g. equatorial-mount) beam.

        Returns
        -------
        numpy.ndarray
            Complex voltages of shape ``(ntime, nsrc, nchan, 2)``.
        """
        ell = np.asarray(ell, dtype=np.float64)
        emm = np.asarray(emm, dtype=np.float64)
        freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
        chi = np.atleast_1d(np.asarray(chi, dtype=np.float64))
        out = np.empty((chi.size, ell.size, freqs.size, 2), dtype=np.complex128)
        for ti, angle in enumerate(chi):
            l_feed, m_feed = self._rotate_to_feed(ell, emm, angle)
            out[ti] = self._eval(l_feed, m_feed, freqs)
        return out


class UnityBeamProvider(BeamProvider):
    """A flat, unity beam (no attenuation). Used for antennas with no configured beam."""

    def _eval(self, l_feed, m_feed, freqs):
        return np.ones((l_feed.size, freqs.size, 2), dtype=np.float64)


class JimBeamProvider(BeamProvider):
    """Analytic cosine-taper ("JimBeam") voltage provider wrapping :class:`CosineTaperBeam`."""

    def __init__(self, beam: CosineTaperBeam):
        self.beam = beam

    def _eval(self, l_feed, m_feed, freqs):
        # The model grids the beam in SIN direction cosines scaled to degrees
        # (l_deg = 180/pi * l), and katbeam consumes MHz.
        x_deg = np.degrees(l_feed)
        y_deg = np.degrees(m_feed)
        return self.beam.voltages(x_deg, y_deg, freqs / 1e6)


def _ascending(grid, values, axis):
    """Return ``(grid, values)`` with ``grid`` ascending, flipping ``values`` along ``axis`` if not."""
    if grid.size > 1 and grid[0] > grid[-1]:
        return grid[::-1].copy(), np.flip(values, axis=axis)
    return grid, values


def _cattery_substitute(
    pattern: str, *, stype: str | None = None, corr: str | None = None, reim: str | None = None
) -> str:
    """Apply the Cattery/DDFacet ``$(...)`` beam-filename substitutions (and uppercase forms).

    Mirrors DDFacet's ``--Beam-FITSFile``/heterogeneous-beam-json substitution rules (the
    same convention MeqTrees' Cattery beam interpolator uses): ``$(stype)``, ``$(corr)``/
    ``$(xy)``, ``$(reim)``, ``$(realimag)``; an uppercase variable name (e.g. ``$(REIM)``) is
    replaced by the uppercase value.
    """
    subs = {}
    if stype is not None:
        subs["$(stype)"] = stype
        subs["$(STYPE)"] = stype.upper()
    if corr is not None:
        subs["$(corr)"] = subs["$(xy)"] = corr
        subs["$(CORR)"] = subs["$(XY)"] = corr.upper()
    if reim is not None:
        subs["$(reim)"] = reim
        subs["$(REIM)"] = reim.upper()
        realimag = {"re": "real", "im": "imag"}[reim]
        subs["$(realimag)"] = realimag
        subs["$(REALIMAG)"] = realimag.upper()
    for key, val in subs.items():
        pattern = pattern.replace(key, val)
    return pattern


def _beam_axis(hdr, fits_axis, npix, unit, path=""):
    """World values along a linear FITS beam axis, converted to ``unit``.

    ``CUNIT`` is honoured rather than assumed: an ``L``/``M`` axis written in radians, or a
    ``FREQ`` axis in MHz, would otherwise load scaled by 180/pi or 1e-6 with nothing in the
    output to show for it -- the beam simply comes out the wrong size.

    An absent ``CUNIT`` means the axis is already in ``unit``. That is the FITS default and
    what :func:`write_beam_fits`/:func:`write_beam_fits_cattery` emit, so simms' own cubes
    and third-party Cattery sets that omit the keyword both keep working. A ``CUNIT`` that
    is present but unparseable, or that measures the wrong quantity, is an error: guessing
    is how the silent-scaling bug happens in the first place.
    """
    from astropy import units as u

    crpix = hdr[f"CRPIX{fits_axis}"]
    crval = hdr[f"CRVAL{fits_axis}"]
    cdelt = hdr[f"CDELT{fits_axis}"]
    values = (np.arange(npix) - (crpix - 1)) * cdelt + crval

    cunit = str(hdr.get(f"CUNIT{fits_axis}", "")).strip()
    if not cunit:
        return values
    try:
        scale = u.Unit(cunit).to(unit)
    except (ValueError, TypeError, u.UnitsError) as exc:
        where = f" of {path!r}" if path else ""
        raise ValueError(
            f"CUNIT{fits_axis}={cunit!r} on axis {fits_axis}{where} is not convertible to {unit!r}."
        ) from exc
    return values * scale


class FitsBeamProvider(BeamProvider):
    """Per-feed voltage beam interpolated from a gridded FITS cube (measured/eidos beams).

    The cube is a regular grid of complex per-feed voltages over ``(l, m)`` direction
    cosines and frequency; it is defined in the *feed frame* (as holography/eidos beams
    are), so the :class:`BeamProvider` base rotates the sky ``(l, m)`` into that frame by
    ``-chi`` before this interpolates. Bilinear in ``(l, m)``, linear in frequency, zero
    outside the grid.

    Use :meth:`from_fits` for the on-disk layout, or :meth:`from_arrays` to build one
    directly (e.g. in tests).
    """

    def __init__(self, l_grid, m_grid, freqs_hz, values, name: str = ""):
        from scipy.interpolate import RegularGridInterpolator

        l_grid = np.ascontiguousarray(l_grid, dtype=np.float64)
        m_grid = np.ascontiguousarray(m_grid, dtype=np.float64)
        freqs_hz = np.ascontiguousarray(freqs_hz, dtype=np.float64)
        values = np.asarray(values, dtype=np.complex128)  # (nl, nm, nfreq, K); K=2 [HH,VV] or 4 [HH,HV,VH,VV]
        nk = values.shape[-1]
        if nk not in (2, 4) or values.shape != (l_grid.size, m_grid.size, freqs_hz.size, nk):
            raise ValueError("values must have shape (nl, nm, nfreq, 2) or (nl, nm, nfreq, 4)")
        # RegularGridInterpolator requires strictly ascending axes; FITS L/M axes often
        # descend (negative CDELT), so flip any descending axis and its data together.
        l_grid, values = _ascending(l_grid, values, 0)
        m_grid, values = _ascending(m_grid, values, 1)
        freqs_hz, values = _ascending(freqs_hz, values, 2)
        self.name = name
        self.has_leakage = nk == 4
        self.l_grid, self.m_grid, self.freqs_hz = l_grid, m_grid, freqs_hz
        # One interpolator per stored entry; a single-channel cube can't interpolate in
        # frequency, so drop that axis and interpolate in (l, m) only.
        self._single_freq = freqs_hz.size == 1
        grid = (l_grid, m_grid) if self._single_freq else (l_grid, m_grid, freqs_hz)
        self._interp = [
            RegularGridInterpolator(
                grid,
                values[:, :, 0, k] if self._single_freq else values[..., k],
                bounds_error=False,
                fill_value=0.0,
            )
            for k in range(nk)
        ]

    def _interp_entries(self, l_feed, m_feed, freqs):
        """Interpolate every stored entry -> ``(nsrc, nchan, nk)``."""
        nsrc, nchan, nk = l_feed.size, freqs.size, len(self._interp)
        out = np.empty((nsrc, nchan, nk), dtype=np.complex128)
        if self._single_freq:
            pts = np.stack([l_feed, m_feed], axis=-1)  # (nsrc, 2)
            for k in range(nk):
                out[:, :, k] = self._interp[k](pts)[:, None]
        else:
            ll = np.repeat(l_feed, nchan)
            mm = np.repeat(m_feed, nchan)
            ff = np.tile(np.asarray(freqs, dtype=np.float64), nsrc)
            pts = np.stack([ll, mm, ff], axis=-1)
            for k in range(nk):
                out[:, :, k] = self._interp[k](pts).reshape(nsrc, nchan)
        return out

    def _eval(self, l_feed, m_feed, freqs):
        # Diagonal feed voltages (H, V): the co-pol entries HH, VV.
        entries = self._interp_entries(l_feed, m_feed, freqs)
        return np.stack([entries[..., 0], entries[..., -1]], axis=-1)  # [HH, VV]

    def _eval_jones(self, l_feed, m_feed, freqs):
        entries = self._interp_entries(l_feed, m_feed, freqs)
        jones = np.zeros((*entries.shape[:2], 2, 2), dtype=np.complex128)
        if self.has_leakage:  # [HH, HV, VH, VV]
            jones[..., 0, 0] = entries[..., 0]
            jones[..., 0, 1] = entries[..., 1]
            jones[..., 1, 0] = entries[..., 2]
            jones[..., 1, 1] = entries[..., 3]
        else:  # [HH, VV] diagonal
            jones[..., 0, 0] = entries[..., 0]
            jones[..., 1, 1] = entries[..., 1]
        return jones

    @classmethod
    def from_arrays(cls, l_grid, m_grid, freqs_hz, values, name: str = "") -> FitsBeamProvider:
        """Build from explicit grids and a ``(nl, nm, nfreq, K)`` cube (K=2 [HH,VV] or 4 [HH,HV,VH,VV])."""
        return cls(l_grid, m_grid, freqs_hz, values, name=name)

    @classmethod
    def from_fits(cls, path, name: str = "") -> FitsBeamProvider:
        """Load a per-feed voltage beam cube.

        The primary HDU leading axis holds the feed voltages as real/imag pairs:
        **4 planes** ``[HH, VV] x (real, imag)`` for a diagonal beam, or **8 planes**
        ``[HH, HV, VH, VV] x (real, imag)`` for a full 2x2 Jones (leakage). A linear WCS
        gives the ``L``/``M`` axes in degrees (SIN direction cosines scaled to degrees,
        ``l_deg = 180/pi * l``) and the ``FREQ`` axis in Hz -- or in any other units the
        axes' ``CUNIT`` keywords declare, which :func:`_beam_axis` converts.
        """
        from astropy.io import fits

        with fits.open(path) as hdul:
            hdr = hdul[0].header
            data = np.asarray(hdul[0].data, dtype=np.float64)  # (4 or 8, nfreq, nm, nl)
        if data.ndim != 4 or data.shape[0] not in (4, 8):
            raise ValueError("FITS beam cube must have shape (4 or 8, nfreq, nm, nl): feed voltages real/imag.")
        _nplane, nfreq, nm, nl = data.shape

        # FITS axis 1 <-> last numpy axis (nl), axis 2 <-> nm, axis 3 <-> nfreq.
        l_grid = np.radians(_beam_axis(hdr, 1, nl, "deg", path))  # degrees -> direction cosine (small-angle)
        m_grid = np.radians(_beam_axis(hdr, 2, nm, "deg", path))
        freqs_hz = _beam_axis(hdr, 3, nfreq, "Hz", path)

        # Even planes are real parts, odd planes imaginary: (nk, nfreq, nm, nl), nk in {2, 4}.
        complex_planes = data[0::2] + 1j * data[1::2]
        values = np.ascontiguousarray(complex_planes.transpose(3, 2, 1, 0))  # (nl, nm, nfreq, nk)
        return cls(l_grid, m_grid, freqs_hz, values, name=name or str(path))

    @classmethod
    def from_cattery(
        cls,
        pattern: str,
        pol_basis: str = "linear",
        l_axis: str = "-X",
        m_axis: str = "Y",
        name: str = "",
    ) -> FitsBeamProvider:
        """Load a Cattery-schema (MeqTrees Cattery / DDFacet ``--Beam-Model FITS``) 8-file beam set.

        ``pattern`` is either a bare prefix (as written by :func:`write_beam_fits_cattery`,
        e.g. ``"beam"`` -> ``beam_xx_re.fits`` etc.) or a full DDFacet-style ``--Beam-FITSFile``
        pattern containing ``$(corr)``/``$(xy)`` and ``$(reim)``/``$(realimag)`` placeholders
        (see :func:`_cattery_substitute`; any ``$(stype)`` must already be resolved by the
        caller). ``pol_basis``/``l_axis``/``m_axis`` must match what the files were written
        with (see :func:`write_beam_fits_cattery`).

        The returned provider always stores **feed-frame (linear H/V) voltages**, like every
        other provider in this module: a ``pol_basis="circular"`` file set is rotated back
        with the inverse of :func:`corr_basis_transform` (a unitary matrix, so its inverse is
        its conjugate transpose) immediately after loading, undoing the single left-multiply
        the writer applied.
        """
        from astropy.io import fits

        if pol_basis not in ("linear", "circular"):
            raise ValueError(f"pol_basis must be 'linear' or 'circular', got {pol_basis!r}.")
        has_placeholder = any(
            tok in pattern
            for tok in (
                "$(corr)",
                "$(xy)",
                "$(CORR)",
                "$(XY)",
                "$(reim)",
                "$(REIM)",
                "$(realimag)",
                "$(REALIMAG)",
            )
        )
        if not has_placeholder:
            pattern = f"{pattern}_$(corr)_$(reim).fits"

        labels = ["xx", "xy", "yx", "yy"] if pol_basis == "linear" else ["rr", "rl", "lr", "ll"]
        sign_l, sign_m = _axis_sign(l_axis), _axis_sign(m_axis)

        l_grid = m_grid = freqs_hz = None
        planes = []
        for corr in labels:
            reim_data = {}
            for reim in ("re", "im"):
                path = _cattery_substitute(pattern, corr=corr, reim=reim)
                if not Path(path).exists():
                    raise FileNotFoundError(
                        f"Cattery beam file {path!r} does not exist (resolved from pattern {pattern!r})."
                    )
                with fits.open(path) as hdul:
                    hdr = hdul[0].header
                    data = np.asarray(hdul[0].data, dtype=np.float64)
                if data.ndim != 3:
                    raise ValueError(
                        f"Cattery beam file {path!r} must be a (nfreq, nm, nl) cube, got shape {data.shape}."
                    )
                nf, nm, nl = data.shape
                this_l = sign_l * np.radians(_beam_axis(hdr, 1, nl, "deg", path))
                this_m = sign_m * np.radians(_beam_axis(hdr, 2, nm, "deg", path))
                this_freqs = _beam_axis(hdr, 3, nf, "Hz", path)
                if l_grid is None:
                    l_grid, m_grid, freqs_hz = this_l, this_m, this_freqs
                elif not (
                    np.allclose(l_grid, this_l) and np.allclose(m_grid, this_m) and np.allclose(freqs_hz, this_freqs)
                ):
                    raise ValueError(f"Cattery beam file {path!r} has a different (l, m, freq) grid than the rest.")
                reim_data[reim] = data
            planes.append(reim_data["re"] + 1j * reim_data["im"])  # (nf, nm, nl)

        complex_planes = np.stack(planes, axis=0)  # (4, nf, nm, nl) = [pp, pq, qp, qq]
        values = np.ascontiguousarray(complex_planes.transpose(3, 2, 1, 0))  # (nl, nm, nf, 4)

        if pol_basis == "circular":
            s_inv = corr_basis_transform(True).conj().T
            jones = np.einsum("ij,...jk->...ik", s_inv, values.reshape((*values.shape[:3], 2, 2)))
            values = jones.reshape((*values.shape[:3], 4))

        return cls(l_grid, m_grid, freqs_hz, values, name=name or pattern)


def _build_jimbeam(spec, beam_band: str) -> JimBeamProvider:
    """Build a JimBeam provider from a beam-config spec (band shorthand, built-in name, or CSV path)."""
    if spec in (True, None, ""):
        spec = beam_band
    spec = str(spec)
    if spec in BUILTIN_BEAMS:
        beam = CosineTaperBeam.from_builtin(spec)
    elif spec in BAND_BUILTINS:
        beam = CosineTaperBeam.from_builtin(BAND_BUILTINS[spec])
    else:
        beam = CosineTaperBeam.from_csv(spec)
    return JimBeamProvider(beam)


def _build_provider(label: str, beam_config, beam_band: str) -> BeamProvider:
    entry = beam_config.get(label) if beam_config else None
    if not entry:
        log.warning("No beam configured for telescope_name %r; using a unity (flat) beam.", label)
        return UnityBeamProvider()
    if "jimbeam" in entry:
        return _build_jimbeam(entry["jimbeam"], beam_band)
    if "fits" in entry:
        return FitsBeamProvider.from_fits(entry["fits"])
    if "cattery" in entry:
        return FitsBeamProvider.from_cattery(
            entry["cattery"],
            pol_basis=entry.get("pol_basis", "linear"),
            l_axis=entry.get("l_axis", "-X"),
            m_axis=entry.get("m_axis", "Y"),
        )
    log.warning("Beam entry for %r has no 'jimbeam', 'fits' or 'cattery' key; using a unity beam.", label)
    return UnityBeamProvider()


def load_beam_config(path) -> dict:
    """Load a beam-config YAML mapping each ``TELESCOPE_NAME`` to a provider spec."""
    return load_yaml(path)


# ``ANTENNA.MOUNT`` values naming a mount that rotates the beam with parallactic angle.
# MSv2 spells it "alt-az", but writers in the wild use these variants interchangeably.
ALTAZ_MOUNTS = frozenset({"ALT-AZ", "ALTAZ", "AZ-EL", "AZEL"})

# Mounts that hold the feed frame still against the sky's rotation. Listed not because
# anything branches on them, but so an unrecognised value can be told apart from a
# deliberate one and reported -- the whole failure mode here is silence.
FIXED_MOUNTS = frozenset({"EQUATORIAL", "X-Y", "XY", "ORBITING", "SPACE-HALCA", "BIZARRE"})


def _normalise_mount(mount: str) -> str:
    """An ``ANTENNA.MOUNT`` value reduced to the mount name alone, upper-cased.

    Trailing focus designations ("ALT-AZ+NASMYTH-R", as ALMA writes) name where the
    receiver sits, not how the mount moves, so they are dropped before the comparison.
    """
    return str(mount).strip().upper().split("+")[0].strip()


def is_altaz_mount(mount: str) -> bool:
    """Whether an ``ANTENNA.MOUNT`` value names a mount that rotates the beam on the sky.

    An alt-az dish carries the sky's parallactic rotation into the feed frame; an
    equatorial one does not. Single definition, so every path that has to make this call
    -- the DFT grid, the a-terms, and the standalone ``primary-beam`` tool -- agrees.

    Matched against :data:`ALTAZ_MOUNTS` by name rather than by substring: a substring
    test quietly answers False for every spelling it does not contain, and False is the
    answer that produces a beam frozen on the sky with nothing to show for it. Anything
    unrecognised still lands on False -- there is no safe guess -- but
    :func:`warn_unknown_mounts` makes that visible at the point the array is resolved.
    """
    return _normalise_mount(mount) in ALTAZ_MOUNTS


def warn_unknown_mounts(mounts) -> None:
    """Warn about ``ANTENNA.MOUNT`` values that name neither a known rotating nor fixed mount.

    Such a value is treated as non-rotating, which for an alt-az array spelled some way
    this does not know about means the beam silently never rotates. Called once where an
    array is resolved, not per antenna, so a whole table costs one message.
    """
    unknown = sorted({str(m) for m in mounts if _normalise_mount(m) not in ALTAZ_MOUNTS | FIXED_MOUNTS})
    if unknown:
        log.warning(
            "Unrecognised ANTENNA.MOUNT value(s) %s; treating those antennas as non-rotating, "
            "so their beam will not follow the parallactic angle. Known rotating mounts are %s.",
            unknown,
            sorted(ALTAZ_MOUNTS),
        )


def resolve_antenna_beams(telescope_names, mount, beam_config, beam_band: str = "L"):
    """Map antennas to beam types and build one provider per type.

    Parameters
    ----------
    telescope_names : sequence of str
        Per-antenna ``ANTENNA.TELESCOPE_NAME`` values, length ``nant``.
    mount : sequence of str
        Per-antenna ``ANTENNA.MOUNT`` values, length ``nant``.
    beam_config : dict or None
        Mapping of telescope_name -> provider spec (see :func:`load_beam_config`).
    beam_band : str
        Default band for JimBeam entries that omit an explicit model.

    Returns
    -------
    ant_type : numpy.ndarray
        Per-antenna type index, shape ``(nant,)``.
    providers : list of BeamProvider
        One provider per type, indexed by ``ant_type``.
    type_is_altaz : numpy.ndarray
        Whether each type's mount rotates the beam (ALT-AZ), shape ``(ntype,)``.
    """
    telescope_names = [str(t) for t in np.asarray(telescope_names).astype(str)]
    mount = [str(m) for m in np.asarray(mount).astype(str)]
    warn_unknown_mounts(mount)
    labels = list(dict.fromkeys(telescope_names))  # unique, insertion-ordered

    providers = []
    type_is_altaz = []
    for label in labels:
        providers.append(_build_provider(label, beam_config, beam_band))
        first = telescope_names.index(label)
        type_is_altaz.append(is_altaz_mount(mount[first]))

    index = {label: i for i, label in enumerate(labels)}
    ant_type = np.array([index[t] for t in telescope_names], dtype=np.int64)
    return ant_type, providers, np.array(type_is_altaz, dtype=bool)


def load_cattery_beam_json(path) -> dict:
    """Load a Cattery/DDFacet heterogeneous-beam json config (the ``--Beam-FITSFile`` json form).

    This json-based multi-antenna-type config is DDFacet's own convention layered on top of
    the underlying Cattery-format beam FITS files it points at. The schema is a dict of
    arbitrarily-named, chained top-level blocks, each holding a ``"define-stationtypes"``
    mapping (antenna ``NAME`` -> station-type label; a key prefixed with ``~`` is a regex
    over antenna names instead of an exact match; the key ``"cmd::default"`` is the fallback
    type for any antenna no other rule matches) and a ``"patterns"`` mapping (arbitrary key
    -> list of ``$(stype)``-templated file patterns, normally one list spanning different
    frequency ranges).

    Returns ``{"stationtypes": [(matcher, is_regex, stype), ...], "pattern": pattern}``:
    ``stationtypes`` is flattened across every block in file order, with the *first* rule
    for a given exact name/`"cmd::default"` winning (blocks only *add* rules -- "once a
    station is type-specialized the type applies to ALL chained blocks"); resolution
    priority within that list is exact name > regex > ``"cmd::default"``, not strict
    block order (this loader does not replicate DDFacet's own block-by-block precedence
    beyond that -- an edge case not documented precisely enough to reproduce without
    reading DDFacet's source, which is out of scope here).

    Scope: DDFacet allows a station type to list several patterns for different frequency
    ranges (selected by proximity to the MS's SPW coverage); this loader supports only a
    single resolved pattern and raises a clear error if the file defines more than one
    distinct pattern, rather than silently guessing.
    """
    import json

    with open(path) as fh:
        cfg = json.load(fh)

    if not isinstance(cfg, dict):
        raise ValueError(f"Cattery beam json {path!r} must contain a top-level JSON object, got {type(cfg).__name__}.")

    stationtypes = []
    patterns = []
    for block in cfg.values():
        for matcher, stype in block.get("define-stationtypes", {}).items():
            is_regex = matcher.startswith("~")
            stationtypes.append((matcher[1:] if is_regex else matcher, is_regex, stype))
        for pattern_list in block.get("patterns", {}).values():
            patterns.extend(pattern_list)

    distinct = list(dict.fromkeys(patterns))
    if not distinct:
        raise ValueError(f"Cattery beam json {path!r} has no 'patterns' entries.")
    if len(distinct) > 1:
        raise ValueError(
            f"Cattery beam json {path!r} defines {len(distinct)} distinct file patterns "
            "(multiple frequency-range patterns per station type); only a single pattern is supported."
        )
    return {"stationtypes": stationtypes, "pattern": distinct[0]}


def resolve_cattery_antenna_beams(
    antenna_names, mount, cattery_cfg: dict, pol_basis: str = "linear", l_axis: str = "-X", m_axis: str = "Y"
):
    """Map antennas to beam types from a Cattery/DDFacet heterogeneous-beam json config.

    Like :func:`resolve_antenna_beams`, but keys antenna typing by the raw ``ANTENNA.NAME``
    (DDFacet's own convention: exact match, ``~regex``, or the ``"cmd::default"`` fallback --
    see :func:`load_cattery_beam_json`) rather than by the ``TELESCOPE_NAME`` column, and
    loads each type's beam via :meth:`FitsBeamProvider.from_cattery` with ``$(stype)``
    substituted into ``cattery_cfg["pattern"]``.

    Parameters
    ----------
    antenna_names : sequence of str
        Per-antenna ``ANTENNA.NAME`` values, length ``nant``.
    mount : sequence of str
        Per-antenna ``ANTENNA.MOUNT`` values, length ``nant``.
    cattery_cfg : dict
        As returned by :func:`load_cattery_beam_json`.
    pol_basis, l_axis, m_axis
        Passed through to :meth:`FitsBeamProvider.from_cattery` (must match how the
        referenced FITS files were written).

    Returns
    -------
    ant_type, providers, type_is_altaz
        Same shapes/meaning as :func:`resolve_antenna_beams`.
    """
    import re

    antenna_names = [str(a) for a in np.asarray(antenna_names).astype(str)]
    mount = [str(m) for m in np.asarray(mount).astype(str)]
    warn_unknown_mounts(mount)

    exact, regexes, default = {}, [], None
    for matcher, is_regex, stype in cattery_cfg["stationtypes"]:
        if is_regex:
            regexes.append((matcher, stype))
        elif matcher == "cmd::default":
            if default is None:
                default = stype
        else:
            exact.setdefault(matcher, stype)

    def _stype(name):
        if name in exact:
            return exact[name]
        for pat, stype in regexes:
            if re.fullmatch(pat, name):
                return stype
        if default is not None:
            return default
        raise RuntimeError(
            f"No Cattery station type matches antenna {name!r} (no exact/regex rule and no "
            "'cmd::default' fallback in the beam json)."
        )

    types = [_stype(name) for name in antenna_names]
    labels = list(dict.fromkeys(types))  # unique, insertion-ordered

    providers = []
    type_is_altaz = []
    for label in labels:
        pattern = _cattery_substitute(cattery_cfg["pattern"], stype=label)
        providers.append(FitsBeamProvider.from_cattery(pattern, pol_basis=pol_basis, l_axis=l_axis, m_axis=m_axis))
        first = types.index(label)
        type_is_altaz.append(is_altaz_mount(mount[first]))

    index = {label: i for i, label in enumerate(labels)}
    ant_type = np.array([index[t] for t in types], dtype=np.int64)
    return ant_type, providers, np.array(type_is_altaz, dtype=bool)


# Hard cap on the parallactic-angle grid. Sizing by the peak PA rate diverges as a
# source approaches the zenith (rate -> infinity at transit through zenith), which would
# otherwise blow up the beam grid; cap it and warn that the fast region is under-resolved.
MAX_PA_SAMPLES = 2048


def pa_sample_grid(t_start, duration, ra0, dec0, lon, lat, pa_step_deg):
    """A parallactic-angle sample grid spanning the observation, uniform in time.

    The grid is uniform in *time* (so a row is indexed by its timestamp, monotonically
    and without ``arctan2`` wrap issues) and sized by the *maximum* PA rate over the span
    so a fast (near-zenith) transit is resolved to ``pa_step_deg``, up to
    :data:`MAX_PA_SAMPLES` samples.

    Returns
    -------
    tgrid : numpy.ndarray
        Sample times (MS seconds), shape ``(n_pa,)``.
    chi_grid : numpy.ndarray
        Unwrapped parallactic angle at each sample (radians), shape ``(n_pa,)``.
    """
    if duration <= 0:
        n_pa = 2
    else:
        fine_t = t_start + np.linspace(0.0, duration, 512)
        chi_fine = np.unwrap(parallactic_angle(fine_t, ra0, dec0, lon, lat))
        rate = np.abs(np.gradient(chi_fine, fine_t)).max()  # rad/s
        span_deg = np.degrees(rate * duration)
        n_pa = max(2, int(np.ceil(span_deg / max(pa_step_deg, 1e-6))) + 1)
        if n_pa > MAX_PA_SAMPLES:
            log.warning(
                "Parallactic-angle sampling wants %d grid points (fast/near-zenith "
                "transit); capping at %d. The beam near transit is under-resolved -- "
                "raise --beam-pa-step or avoid a near-zenith field for full accuracy.",
                n_pa,
                MAX_PA_SAMPLES,
            )
            n_pa = MAX_PA_SAMPLES
    tgrid = t_start + np.linspace(0.0, duration, n_pa)
    chi_grid = np.unwrap(parallactic_angle(tgrid, ra0, dec0, lon, lat))
    return tgrid, chi_grid


# Default hard ceiling (GiB) on the sampled beam grid. The grid is built once at full
# size and only sliced per channel-chunk downstream, so its whole footprint lives in
# memory for the run; --chan-chunks does not reduce it.
BEAM_GRID_MAX_GIB_DEFAULT = 4.0


def _beam_grid_gib(ntype, n_pa, nsrc, nchan, fold):
    """Footprint (GiB) of a ``(ntype, n_pa, nsrc, nchan, fold)`` complex64 beam grid.

    ``fold`` is 2 for the diagonal per-feed grid, 4 for the 2x2 Jones grid; complex64 is
    8 bytes per element.
    """
    return ntype * n_pa * nsrc * nchan * fold * 8 / 2**30


def _check_beam_grid_footprint(ntype, n_pa, nsrc, nchan, fold, max_gib):
    """Warn/raise on the beam-grid footprint before it is allocated.

    Warns above half the ceiling and raises :class:`MemoryError` above ``max_gib``, with an
    actionable message naming the levers (the grid scales with PA samples x sources x
    channels). See :data:`BEAM_GRID_MAX_GIB_DEFAULT`.
    """
    gib = _beam_grid_gib(ntype, n_pa, nsrc, nchan, fold)
    dims = f"{ntype} type(s) x {n_pa} PA x {nsrc} src x {nchan} chan x {fold}"
    if gib > max_gib:
        raise MemoryError(
            f"Primary-beam grid would need {gib:.2f} GiB ({dims}), above the "
            f"{max_gib:.2f} GiB ceiling. Reduce it with a coarser --beam-pa-step (fewer PA "
            f"samples), fewer sky components, or raise --beam-grid-max-gib. --chan-chunks "
            f"does not help: the grid is built once for all channels."
        )
    if gib > 0.5 * max_gib:
        log.warning(
            "Primary-beam grid needs %.2f GiB (%s), over half the %.2f GiB ceiling; a "
            "coarser --beam-pa-step or fewer components lowers it.",
            gib,
            dims,
            max_gib,
        )


def build_beam_grid(providers, type_is_altaz, ell, emm, freqs, chi_grid, max_gib=BEAM_GRID_MAX_GIB_DEFAULT):
    """Sample every type's voltage beam on the PA grid.

    Returns an array of shape ``(ntype, n_pa, nsrc, nchan, 2)``. Stored as ``complex64``
    (halving this large array vs ``complex128``); a beam voltage is O(1), so single
    precision is ample and visibilities still accumulate in double. Alt-az types use
    ``chi_grid``; others are evaluated at zero parallactic angle (no rotation). Raises
    :class:`MemoryError` if the grid would exceed ``max_gib`` (see
    :func:`_check_beam_grid_footprint`).
    """
    ntype = len(providers)
    _check_beam_grid_footprint(ntype, chi_grid.size, ell.size, freqs.size, 2, max_gib)
    grid = np.empty((ntype, chi_grid.size, ell.size, freqs.size, 2), dtype=np.complex64)
    zeros = np.zeros_like(chi_grid)
    for ti, prov in enumerate(providers):
        use_chi = chi_grid if type_is_altaz[ti] else zeros
        grid[ti] = prov.voltage(ell, emm, freqs, use_chi)  # complex128 -> complex64 on assign
    return grid


# Linear -> circular feed transform, consistent with simms' circular brightness
# (R=(X+iY)/sqrt2, L=(X-iY)/sqrt2): verified S @ B_linear @ S^H == B_circular.
_S_LIN2CIRC = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0j], [1.0, -1.0j]], dtype=np.complex128)


def corr_basis_transform(is_circular: bool) -> np.ndarray:
    """2x2 transform folded into the beam Jones so ``V`` lands in the MS correlation basis.

    Identity for a linear-correlation MS; the constant linear->circular feed matrix for a
    circular one (the feeds are physically linear, so circular correlations are a fixed
    rotation of the linear ones).
    """
    return _S_LIN2CIRC.copy() if is_circular else np.eye(2, dtype=np.complex128)


def build_beam_grid_jones(
    providers, type_is_altaz, ell, emm, freqs, chi_grid, basis_transform, max_gib=BEAM_GRID_MAX_GIB_DEFAULT
):
    """Sample every type's 2x2 voltage Jones on the PA grid, folding the basis transform.

    Returns ``(ntype, n_pa, nsrc, nchan, 2, 2)`` complex64 holding ``E' = S . E``, so the
    kernel's ``V = E'_p B_feed E'_q^H = S (E_p B E_q^H) S^H`` lands in the MS correlation
    basis (``B_feed`` is the linear-feed coherency). Alt-az types use ``chi_grid``. Raises
    :class:`MemoryError` if the grid would exceed ``max_gib`` (see
    :func:`_check_beam_grid_footprint`).
    """
    ntype = len(providers)
    _check_beam_grid_footprint(ntype, chi_grid.size, ell.size, freqs.size, 4, max_gib)
    grid = np.empty((ntype, chi_grid.size, ell.size, freqs.size, 2, 2), dtype=np.complex64)
    zeros = np.zeros_like(chi_grid)
    for ti, prov in enumerate(providers):
        use_chi = chi_grid if type_is_altaz[ti] else zeros
        jones = prov.jones(ell, emm, freqs, use_chi)  # (n_pa, nsrc, nchan, 2, 2)
        grid[ti] = np.einsum("ij,tsfjk->tsfik", basis_transform, jones)
    return grid


# Points evaluated per beam-provider call. A provider returns complex128 voltages (32 B
# per point per feed pair) and the analytic models build several float64 temporaries on
# top, so evaluating an 8k x 8k image in one call costs GiB. Slabbing bounds that scratch
# to tens of MiB regardless of how many points are asked for.
EVAL_SLAB_PIXELS = 1 << 20


def image_power_beam(provider, is_altaz, ell, emm, freqs, chi_grid):
    """Parallactic-angle-averaged power beam ``<0.5(|g^X|^2 + |g^V|^2)>`` at each point.

    For the FITS-*image* path, which grids one apparent sky for all baselines and times:
    there is no per-baseline beam, so the beam is averaged over the observation's
    parallactic-angle range (a single sample when ``is_altaz`` is False). Loops over
    point slabs, frequency and PA so the scratch a provider allocates stays bounded by
    :data:`EVAL_SLAB_PIXELS` rather than growing with the image -- this is the path the
    FITS a-term mode degrades into when its cache will not fit, so it must not need more
    memory than the mode it is rescuing.

    Parameters
    ----------
    provider : BeamProvider
        A single representative antenna beam.
    is_altaz : bool
        Whether to average over ``chi_grid`` (True) or evaluate once at chi = 0.
    ell, emm : numpy.ndarray
        Direction cosines of the points (pixels/components), shape ``(npts,)``.
    freqs : numpy.ndarray
        Frequencies (Hz), shape ``(nchan,)``.
    chi_grid : numpy.ndarray
        Parallactic-angle samples (radians).

    Returns
    -------
    numpy.ndarray
        Real power beam of shape ``(npts, nchan)`` in ``[0, ~1]``.
    """
    ell = np.asarray(ell, dtype=np.float64)
    emm = np.asarray(emm, dtype=np.float64)
    freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
    chis = chi_grid if is_altaz else np.zeros(1)
    npts = ell.size
    power = np.empty((npts, freqs.size))
    for start in range(0, npts, EVAL_SLAB_PIXELS):
        sl = slice(start, min(start + EVAL_SLAB_PIXELS, npts))
        l_slab, m_slab = ell[sl], emm[sl]
        for k in range(freqs.size):
            fk = freqs[k : k + 1]
            acc = np.zeros(l_slab.size)
            for chi in chis:
                g = provider.voltage(l_slab, m_slab, fk, np.array([chi]))  # (1, nslab, 1, 2)
                acc += 0.5 * (np.abs(g[0, :, 0, 0]) ** 2 + np.abs(g[0, :, 0, 1]) ** 2)
            power[sl, k] = acc / chis.size
    return power


def corr_feed_maps(ncorr):
    """Feed indices for each correlation in the linear basis.

    Returns ``(corr_feed_p, corr_feed_q)`` giving, per correlation, the feed (0=H/X,
    1=V/Y) of the first and second antenna. 4-corr is ``[XX, XY, YX, YY]``; 2-corr is
    ``[XX, YY]``. Circular basis is unsupported by the diagonal per-feed model.
    """
    if ncorr == 4:
        return np.array([0, 0, 1, 1], dtype=np.int64), np.array([0, 1, 0, 1], dtype=np.int64)
    if ncorr == 2:
        return np.array([0, 1], dtype=np.int64), np.array([0, 1], dtype=np.int64)
    raise ValueError(f"Primary beam supports 2 or 4 correlations, got {ncorr}.")


def array_lonlat(positions):
    """Geodetic (lon, lat) in radians from the mean of ITRF/ECEF antenna positions."""
    from astropy import units as u
    from astropy.coordinates import EarthLocation

    x, y, z = np.asarray(positions, dtype=np.float64).mean(axis=0)
    loc = EarthLocation.from_geocentric(x * u.m, y * u.m, z * u.m)
    return loc.lon.to_value(u.rad), loc.lat.to_value(u.rad)


# Direction measure frames that name a horizontal (az/el-type) direction. These map to a
# *time-dependent* sky position, so a single fixed beam centre is meaningless.
_HORIZONTAL_POINTING_FRAMES = frozenset({"AZEL", "AZELGEO", "AZELSW", "AZELNE", "HADEC"})
# Equatorial frames handled silently on the fast path (a constant RA/Dec direction).
_EQUATORIAL_POINTING_FRAMES = frozenset({"J2000", "ICRS"})
# FIELD columns tried, in order, when POINTING.DIRECTION is unusable. Phase-rotation tools
# (chgcentre, phaseshift) move PHASE_DIR only and leave these at the pointing centre, so they
# are the last record of where the dishes pointed once POINTING is gone.
_FIELD_POINTING_COLUMNS = ("REFERENCE_DIR", "DELAY_DIR")


def _direction_frame(ms, subtable, column):
    """Measure reference of a direction column (upper-case), or ``""`` if unavailable.

    Reads the column's ``MEASINFO``/``Ref`` keyword via casacore. A missing table or keyword
    yields ``""`` (treated as equatorial by the caller, preserving the legacy fast path).
    """
    try:
        from casacore.tables import table

        with table(f"{ms}::{subtable}", ack=False) as tab:
            measinfo = tab.getcolkeywords(column).get("MEASINFO", {})
        return str(measinfo.get("Ref", "")).upper()
    except Exception:
        return ""


def _warn_nonstandard_frame(frame, label):
    """Warn when a direction column's frame is equatorial-but-unusual (e.g. ``B1950``)."""
    if frame and frame not in _EQUATORIAL_POINTING_FRAMES:
        log.warning(
            "%s frame %r is treated as a fixed RA/Dec beam centre; if it is not equatorial the "
            "primary beam may be mis-centred.",
            label,
            frame,
        )


def _read_direction(ms, subtable, column, row):
    """Order-0 poly term of ``row`` of a ``(nrow, npoly, 2)`` direction column, or ``None``.

    ``None`` means "not usable as a beam centre": the subtable or column is absent, the row is
    missing, or the value is not a finite ``(ra, dec)`` pair. ``(0, 0)`` is *not* treated as
    unset -- it is a real sky position.
    """
    from daskms import xds_from_table

    label = f"{subtable}.{column}"
    try:
        ds = xds_from_table(f"{ms}::{subtable}")[0]
        if column not in ds:
            raise ValueError(f"no {column} column")
        data = ds[column].data
        if data.shape[0] <= row:
            raise ValueError(f"only {data.shape[0]} row(s), wanted row {row}")
        radec = np.asarray(data[row, 0].compute(), dtype=np.float64)
    except Exception as exc:
        log.debug("No usable %s (%s).", label, exc)
        return None
    if radec.shape != (2,) or not np.all(np.isfinite(radec)):
        log.debug("No usable %s (direction %r is not a finite (ra, dec) pair).", label, radec)
        return None
    return float(radec[0]), float(radec[1])


def _pointing_row_for_times(ms, time_range):
    """Index of a ``POINTING`` row recorded while ``time_range`` was being observed.

    ``POINTING`` is keyed by ``(ANTENNA_ID, TIME)`` and carries no ``FIELD_ID``, so the
    only handle on which field a pointing belongs to is *when* it was recorded. Row 0 is
    whichever field the MS observed first, which is the wrong beam centre for every other
    field of a mosaic; ``time_range`` is the ``TIME`` span of the rows the run actually
    selected, so a row inside it points where those rows were looking.

    Falls back to the row nearest the span's midpoint when nothing lies strictly inside
    (a ``POINTING`` cadence coarser than the field's dwell), and to row 0 when there is no
    usable ``TIME`` column at all -- both strictly better than, or the same as, the
    legacy unconditional row 0.
    """
    if time_range is None:
        return 0
    from daskms import xds_from_table

    t0, t1 = (float(t) for t in time_range)
    try:
        ds = xds_from_table(f"{ms}::POINTING")[0]
        if "TIME" not in ds:
            raise ValueError("no TIME column")
        times = np.asarray(ds.TIME.data.compute(), dtype=np.float64)
    except Exception as exc:
        log.debug("No usable POINTING.TIME (%s); reading the beam centre from row 0.", exc)
        return 0
    if times.size == 0:
        return 0
    within = np.nonzero((times >= t0) & (times <= t1))[0]
    if within.size:
        return int(within[0])
    nearest = int(np.argmin(np.abs(times - 0.5 * (t0 + t1))))
    log.debug(
        "No POINTING row inside the selected time span [%.3f, %.3f]; using row %d at TIME %.3f.",
        t0,
        t1,
        nearest,
        times[nearest],
    )
    return nearest


def read_pointing_centre(ms, fallback_ra0, fallback_dec0, field_id=0, time_range=None):
    """Antenna pointing centre (radians), tried in order of how faithfully it records the pointing.

    This is where the dishes point, and hence where the primary beam is centred -- distinct
    from ``FIELD.PHASE_DIR`` (the correlator phase centre, a freely-shiftable quantity). The
    chain is

    ``POINTING.DIRECTION`` -> ``FIELD.REFERENCE_DIR`` -> ``FIELD.DELAY_DIR`` -> the given
    phase centre, with a warning.

    The ``FIELD`` fallbacks matter on a rephased MS: ``chgcentre``/``phaseshift`` move
    ``PHASE_DIR`` alone and leave ``REFERENCE_DIR``/``DELAY_DIR`` at the pointing, so dropping
    straight to the phase centre centres the beam on the wrong place. Each candidate is
    validated rather than blindly preferred -- an absent column or a non-finite value is
    skipped -- since some writers leave ``REFERENCE_DIR`` unpopulated.

    ``POINTING`` has no ``FIELD_ID``, so ``field_id`` cannot select its row directly; pass
    ``time_range`` (the ``TIME`` span of the rows the run selected) and the row is chosen
    from that instead -- see :func:`_pointing_row_for_times`. Without it the first row is
    read, which is the pointing of whichever field the MS observed first. The ``FIELD``
    fallbacks are indexed by ``field_id`` directly.

    The direction is interpreted as a fixed sky position, which requires an equatorial measure
    frame. ``J2000``/``ICRS`` (and an absent keyword, as older simms MSs may have) take the
    fast path silently; another equatorial frame is accepted with a warning. A **horizontal**
    frame (``AZEL`` etc.) maps to a time-dependent sky direction that a single beam centre
    cannot represent: on ``POINTING.DIRECTION`` -- the authoritative record, with nothing
    better to fall back to -- it raises :class:`NotImplementedError`; on a ``FIELD`` fallback
    it merely disqualifies that candidate, and the chain continues.
    """
    frame = _direction_frame(ms, "POINTING", "DIRECTION")
    if frame in _HORIZONTAL_POINTING_FRAMES:
        raise NotImplementedError(
            f"POINTING.DIRECTION is in the horizontal frame {frame!r}, which maps to a "
            f"time-dependent sky position; a fixed primary-beam centre is unsupported. Supply "
            f"a J2000/ICRS POINTING table or a target-tracking MS."
        )
    _warn_nonstandard_frame(frame, "POINTING.DIRECTION")
    centre = _read_direction(ms, "POINTING", "DIRECTION", _pointing_row_for_times(ms, time_range))
    if centre is not None:
        return centre

    for column in _FIELD_POINTING_COLUMNS:
        frame = _direction_frame(ms, "FIELD", column)
        if frame in _HORIZONTAL_POINTING_FRAMES:
            log.debug("Skipping FIELD.%s as a beam centre (horizontal frame %r).", column, frame)
            continue
        centre = _read_direction(ms, "FIELD", column, int(field_id))
        if centre is None:
            continue
        _warn_nonstandard_frame(frame, f"FIELD.{column}")
        log.info("No usable POINTING.DIRECTION; centring the primary beam on FIELD.%s.", column)
        return centre

    log.warning(
        "No usable POINTING.DIRECTION, FIELD.REFERENCE_DIR or FIELD.DELAY_DIR; using the phase "
        "centre as the beam centre. If this MS has been rephased, the beam is mis-centred."
    )
    return float(fallback_ra0), float(fallback_dec0)


def reproject_lm(ell, emm, from_ra0, from_dec0, to_ra0, to_dec0):
    """Re-reference direction cosines from one tangent centre to another (a no-op if they match).

    Source ``(l, m)`` prepared for the phase centre are re-expressed relative to the beam
    (pointing) centre, so the beam is sampled at each source's offset from where the dish points.
    """
    if from_ra0 == to_ra0 and from_dec0 == to_dec0:
        return ell, emm
    from simms.skymodel.fits_skies import lm_to_radec
    from simms.utilities import radec2lm

    ra, dec = lm_to_radec(np.asarray(ell), np.asarray(emm), from_ra0, from_dec0)
    return radec2lm(to_ra0, to_dec0, ra, dec)


def resolve_beam(spec, band: str = "L") -> BeamProvider:
    """Build a beam provider from a spec: a ``.fits`` cube, a CSV/built-in JimBeam, or a band."""
    spec = band if spec in (None, "") else str(spec)
    if spec.lower().endswith(".fits"):
        return FitsBeamProvider.from_fits(spec)
    return _build_jimbeam(spec, band)


def averaged_power_beam(provider, is_altaz, ell, emm, freqs, chi_grid):
    """Frequency- and parallactic-angle-averaged Stokes-I power beam ``A(l, m)``, shape ``(npts,)``.

    Averages :func:`image_power_beam` (already PA-averaged) over frequency. Used for the
    image-/component-domain apply/correct, which is a direct multiply/divide by one map.
    ``is_altaz`` must come from the array's ``ANTENNA.MOUNT``: averaging a beam that never
    rotates over the PA track smears an asymmetric or squinted pattern that in truth held
    still (see :func:`image_power_beam`).
    """
    return image_power_beam(provider, is_altaz, ell, emm, freqs, chi_grid).mean(axis=1)


def write_beam_fits(beam: CosineTaperBeam, l_grid, m_grid, freqs_hz, path):
    """Write a cosine-taper beam to a 4-plane FITS cube that :meth:`FitsBeamProvider.from_fits` reads.

    Samples ``beam`` on the ``(l_grid, m_grid)`` direction-cosine grid at ``freqs_hz`` and stores
    ``[HH, VV]`` feed voltages as real/imag planes: ``(4, nfreq, nm, nl)``, with a linear WCS giving
    L/M in degrees and FREQ in Hz.
    """
    from astropy.io import fits

    l_grid = np.ascontiguousarray(l_grid, dtype=np.float64)
    m_grid = np.ascontiguousarray(m_grid, dtype=np.float64)
    freqs_hz = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    nl, nm, nf = l_grid.size, m_grid.size, freqs_hz.size
    # The FITS FREQ axis is linear (CRVAL + CDELT*pixel), so the frequencies must be
    # uniformly spaced. The cosine-taper model is continuous in frequency, so resample
    # onto a uniform grid rather than write a non-uniform one that reloads incorrectly.
    if nf > 2 and not np.allclose(np.diff(freqs_hz), freqs_hz[1] - freqs_hz[0]):
        raise ValueError("write_beam_fits needs uniformly-spaced frequencies for the linear FITS FREQ axis.")

    ll, mm = np.meshgrid(l_grid, m_grid, indexing="ij")
    v = beam.voltages(np.degrees(ll.ravel()), np.degrees(mm.ravel()), freqs_hz / 1e6)  # (nl*nm, nf, 2)
    v = v.reshape(nl, nm, nf, 2)
    hh = v[..., 0].transpose(2, 1, 0)  # (nf, nm, nl)
    vv = v[..., 1].transpose(2, 1, 0)
    data = np.stack([hh.real, hh.imag, vv.real, vv.imag], axis=0).astype(np.float32)  # (4, nf, nm, nl)

    dl = np.degrees(l_grid[1] - l_grid[0]) if nl > 1 else 1.0
    dm = np.degrees(m_grid[1] - m_grid[0]) if nm > 1 else 1.0
    df = float(freqs_hz[1] - freqs_hz[0]) if nf > 1 else 1e6
    hdr = fits.Header()
    hdr["CTYPE1"], hdr["CRPIX1"], hdr["CRVAL1"], hdr["CDELT1"], hdr["CUNIT1"] = "L", 1, np.degrees(l_grid[0]), dl, "deg"
    hdr["CTYPE2"], hdr["CRPIX2"], hdr["CRVAL2"], hdr["CDELT2"], hdr["CUNIT2"] = "M", 1, np.degrees(m_grid[0]), dm, "deg"
    hdr["CTYPE3"], hdr["CRPIX3"], hdr["CRVAL3"], hdr["CDELT3"], hdr["CUNIT3"] = "FREQ", 1, float(freqs_hz[0]), df, "Hz"
    hdr["CTYPE4"], hdr["CRPIX4"], hdr["CRVAL4"], hdr["CDELT4"] = "FEED", 1, 0, 1
    hdr["BUNIT"] = "1"
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


_VALID_AXIS_SIGNS = frozenset({"-X", "X", "-Y", "Y"})


def _axis_sign(spec: str) -> float:
    """``"-X"``/``"-Y"`` -> ``-1.0``, ``"X"``/``"Y"`` -> ``1.0`` (DDFacet's FITSLAxis/FITSMAxis)."""
    spec = str(spec).strip()
    if spec not in _VALID_AXIS_SIGNS:
        raise ValueError(f"Invalid axis sign convention {spec!r}; expected one of {sorted(_VALID_AXIS_SIGNS)}.")
    return -1.0 if spec.startswith("-") else 1.0


def write_beam_fits_cattery(
    beam: CosineTaperBeam,
    l_grid,
    m_grid,
    freqs_hz,
    prefix,
    pol_basis: str = "linear",
    l_axis: str = "-X",
    m_axis: str = "Y",
) -> list:
    """Write a cosine-taper beam as the Cattery 8-file FITS-beam schema.

    This is the beam-FITS layout used by MeqTrees' Cattery beam interpolator, and also
    read by DDFacet (``[Beam]`` parset, ``Beam-Model=FITS``, which inherited the format
    from Cattery): a real/imaginary pair per entry of the 2x2 voltage Jones matrix --
    **8 separate files** -- named by DDFacet's default ``Beam-FITSFile`` pattern
    ``beam_$(corr)_$(reim).fits``: here, ``f"{prefix}_{corr}_{reim}.fits"`` with ``corr``
    in ``xx,xy,yx,yy`` (linear feeds, ``pol_basis="linear"``) or ``rr,rl,lr,ll`` (circular,
    ``pol_basis="circular"``) and ``reim`` in ``re,im``. Each file is a ``(nfreq, nm, nl)``
    cube: CTYPE1/2 are the spatial axes (degrees), CTYPE3 is FREQ (Hz).

    The cosine-taper model has no leakage, so the off-diagonal (xy/yx or rl/lr)
    planes are the diagonal Jones ``diag(HH, VV)`` with no cross terms, optionally
    rotated into the circular basis by :func:`corr_basis_transform` (a single
    left-multiply ``E' = S @ E``, the same transform :func:`build_beam_grid_jones`
    applies -- not the baseline-coherency ``S @ B @ S^H`` form).

    ``l_axis``/``m_axis`` (``"-X"``/``"X"`` and ``"Y"``/``"-Y"``) mirror DDFacet's own
    ``--Beam-FITSLAxis``/``--Beam-FITSMAxis`` options and their documented convention
    ("Minus sign indicates reverse coordinate convention"): passing the same string to
    this writer and to DDFacet round-trips to the identity, so the defaults here match
    DDFacet's own defaults.
    """
    from astropy.io import fits

    if pol_basis not in ("linear", "circular"):
        raise ValueError(f"pol_basis must be 'linear' or 'circular', got {pol_basis!r}.")

    l_grid = np.ascontiguousarray(l_grid, dtype=np.float64)
    m_grid = np.ascontiguousarray(m_grid, dtype=np.float64)
    freqs_hz = np.atleast_1d(np.asarray(freqs_hz, dtype=np.float64))
    nl, nm, nf = l_grid.size, m_grid.size, freqs_hz.size
    if nf > 2 and not np.allclose(np.diff(freqs_hz), freqs_hz[1] - freqs_hz[0]):
        raise ValueError("write_beam_fits_cattery needs uniformly-spaced frequencies for the linear FITS FREQ axis.")

    ll, mm = np.meshgrid(l_grid, m_grid, indexing="ij")
    v = beam.voltages(np.degrees(ll.ravel()), np.degrees(mm.ravel()), freqs_hz / 1e6)  # (nl*nm, nf, 2)
    v = v.reshape(nl, nm, nf, 2).transpose(2, 1, 0, 3)  # (nf, nm, nl, 2) = [H, V] voltages

    jones = np.zeros((*v.shape[:3], 2, 2), dtype=np.complex128)  # (nf, nm, nl, 2, 2)
    jones[..., 0, 0] = v[..., 0]
    jones[..., 1, 1] = v[..., 1]
    if pol_basis == "circular":
        jones = np.einsum("ij,...jk->...ik", corr_basis_transform(True), jones)

    labels = ["xx", "xy", "yx", "yy"] if pol_basis == "linear" else ["rr", "rl", "lr", "ll"]

    sign_l = _axis_sign(l_axis)
    sign_m = _axis_sign(m_axis)
    dl = np.degrees(l_grid[1] - l_grid[0]) if nl > 1 else 1.0
    dm = np.degrees(m_grid[1] - m_grid[0]) if nm > 1 else 1.0
    df = float(freqs_hz[1] - freqs_hz[0]) if nf > 1 else 1e6

    hdr = fits.Header()
    hdr["CTYPE1"], hdr["CRPIX1"], hdr["CUNIT1"] = "X", 1, "deg"
    hdr["CRVAL1"], hdr["CDELT1"] = sign_l * np.degrees(l_grid[0]), sign_l * dl
    hdr["CTYPE2"], hdr["CRPIX2"], hdr["CUNIT2"] = "Y", 1, "deg"
    hdr["CRVAL2"], hdr["CDELT2"] = sign_m * np.degrees(m_grid[0]), sign_m * dm
    hdr["CTYPE3"], hdr["CRPIX3"], hdr["CRVAL3"], hdr["CDELT3"], hdr["CUNIT3"] = "FREQ", 1, float(freqs_hz[0]), df, "Hz"
    hdr["BUNIT"] = "1"

    paths = []
    for (i, j), corr in zip([(0, 0), (0, 1), (1, 0), (1, 1)], labels, strict=True):
        plane = jones[..., i, j]
        for part, suffix in ((plane.real, "re"), (plane.imag, "im")):
            path = f"{prefix}_{corr}_{suffix}.fits"
            fits.PrimaryHDU(data=part.astype(np.float32), header=hdr).writeto(path, overwrite=True)
            paths.append(path)
    return paths
