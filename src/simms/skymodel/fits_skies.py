"""
Visibility prediction from FITS sky models.

Conventions
-----------
Everything here predicts the same quantity, verified in ``tests/predict_fits_tests.py``
against a brute-force transcription of the RIME::

    V(u, v, w, nu) = sum_pix S_pix * exp(2j*pi * (u*l + v*m + w*(n - 1)) * nu / c)

where ``(l, m)`` are the direction cosines of a pixel *relative to the MS phase
centre*, ``n = sqrt(1 - l**2 - m**2)``, and there is no ``1/n`` factor because the
image is in Jy/pixel rather than surface brightness.

``ducc0.wgridder.dirty2vis`` reproduces this exactly when called with
``divide_by_n=False`` and the pixel-to-direction-cosine mapping
``l = -(i - npix//2) * pixsize_x``, ``m = -(j - npix//2) * pixsize_y``. It therefore
requires a *regular* ``(l, m)`` grid, which only an orthographic (SIN) projection
whose tangent point coincides with the phase centre provides. Any other projection,
or a tangent point away from the phase centre, is reprojected onto such a grid first.

The DFT backend needs no regular grid and is exact for every projection, so sparse
component-like images are predicted without resampling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
from astropy import units
from astropy.wcs import WCS
from ducc0.wgridder import dirty2vis
from fitstoolz.reader import FitsData
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

from simms import BIN
from simms.exceptions import FITSSkymodelError
from simms.skymodel.fits_spectrum import (
    DEFAULT_FIT_TOLERANCE,
    FitsSpectrum,
    SpectralKind,
    evaluate_scale,
    fit_log_polynomial,
    interpolate_planes_at,
    read_spi_maps,
)
from simms.skymodel.kernels import is_uniform_grid, predict_vis
from simms.skymodel.mstools import add_noise, smear_kernel_args, stack_unpolarised_vis
from simms.skymodel.smearing import Smearing
from simms.utilities import is_range_in_range, radec2lm

log = logging.getLogger(__name__)

# A pixel grid is treated as regular in (l, m) - and so usable by the gridder -
# when the direction cosines deviate from a linear function of pixel index by
# less than this, in pixels. SIN at the phase centre gives exactly zero.
MAX_GRID_DEVIATION_PIXELS = 1e-2

# Cost-model constants, measured at 1 thread with nchan=16, npix in {1024, 2048}
# and nrow in {2e3, 2e4, 1e5}. The gridder's degridding term grows with the image
# side (wider kernel support, more w planes), hence the per-side factor. These only
# choose between two correct backends, so an imprecise fit costs time, never
# accuracy. Break-even lands at a few hundred components.
DFT_SECONDS_PER_COMPONENT_ROW_CHAN = 6.4e-9
FFT_SECONDS_PER_PIXEL_LOGPIXEL_CHAN = 2.5e-9
FFT_SECONDS_PER_ROW_CHAN_PER_SIDE = 7.0e-10

STOKES_CODES = {1: "I", 2: "Q", 3: "U", 4: "V"}


@dataclass
class FitsGrid:
    """Mapping from image pixel indices to direction cosines, for ``dirty2vis``.

    ``l = l_ref + delta_l * (i - ref_l)`` and likewise for ``m``.
    """

    delta_l: float  # signed, radians per pixel
    delta_m: float
    ref_l: float  # 0-based pixel index of the reference point
    ref_m: float
    l_ref: float  # direction cosine at the reference pixel
    m_ref: float
    deviation_pixels: float

    @property
    def is_regular(self) -> bool:
        return self.deviation_pixels < MAX_GRID_DEVIATION_PIXELS

    def ducc_kwargs(self, npix_l: int, npix_m: int) -> dict:
        """Arguments placing this grid in the frame ``dirty2vis`` assumes."""
        centre_l, centre_m = npix_l // 2, npix_m // 2
        return dict(
            pixsize_x=abs(self.delta_l),
            pixsize_y=abs(self.delta_m),
            flip_u=bool(self.delta_l > 0),
            flip_v=bool(self.delta_m > 0),
            center_x=-(self.l_ref + self.delta_l * (centre_l - self.ref_l)),
            center_y=-(self.m_ref + self.delta_m * (centre_m - self.ref_m)),
            divide_by_n=False,
        )

    def pixel_lmn(self, i_pix: np.ndarray, j_pix: np.ndarray) -> np.ndarray:
        """Direction cosines ``(l, m, n - 1)`` of pixels on this regular grid."""
        el = self.l_ref + self.delta_l * (i_pix - self.ref_l)
        em = self.m_ref + self.delta_m * (j_pix - self.ref_m)
        en = np.sqrt(np.maximum(1 - el * el - em * em, 0.0)) - 1
        return np.stack([el, em, en], axis=-1)


@dataclass
class PreparedFitsSky:
    """A FITS sky model reduced to what a prediction backend needs."""

    chan_freqs: np.ndarray  # MS channel centres (Hz)
    spectrum: FitsSpectrum  # how the model varies with frequency
    backend: str  # "dft", "fft" or "perchan"
    ncorr: int
    polarisation: bool
    linear_basis: bool
    ncomp: int
    npix_l: int
    npix_m: int

    # FFT and perchan backends: real Stokes planes,
    # (nstokes, npix_l, npix_m, nchan_model). FLAT and POLY store one plane, at
    # the reference frequency; CUBE (and hence perchan) stores the whole MS grid.
    planes: np.ndarray | None = None
    stokes_names: list[str] | None = None
    grid: FitsGrid | None = None
    tol: float = 1e-7  # perchan: per-channel component threshold (Jy)

    # DFT backend: brightness per correlation, (ncomp, nspec, nchan)
    lmn: np.ndarray | None = None
    bmat: np.ndarray | None = None
    uniform_freqs: bool = True

    # A-term correction (gridder backends): per-antenna primary beams applied in
    # the image domain (see simms.skymodel.aterms). ``chan_ids`` are the global
    # channel indices of a channel-chunked block, which the a-term model needs
    # to place the block's channels on its frequency-knot grid.
    aterm: object | None = None
    chan_ids: np.ndarray | None = None

    # DFT backend only: time/bandwidth smearing (see simms.skymodel.smearing). The
    # gridder backends integrate whole images and cannot carry a per-visibility factor.
    smearing: Smearing | None = None

    @property
    def nspec(self) -> int:
        return self.ncorr if self.polarisation else 1

    @property
    def flat_spectrum(self) -> bool:
        """True when one image serves every channel, so the gridder runs once."""
        return self.spectrum.kind is SpectralKind.FLAT

    def select_channels(self, chan_ids: np.ndarray) -> PreparedFitsSky:
        """Restrict the model to a subset of channels, for channel-chunked prediction.

        The reference image and the spectral coefficients are channel-independent,
        so only the channel grid moves for FLAT and POLY; a CUBE also slices its
        per-channel planes, and the DFT brightness matrix its channel axis.
        """
        updates = {"chan_freqs": self.chan_freqs[chan_ids]}
        # Track the selected channels' *global* indices across (possibly chained)
        # selections, for the a-term frequency-knot bookkeeping.
        base_ids = np.arange(self.chan_freqs.size) if self.chan_ids is None else self.chan_ids
        updates["chan_ids"] = base_ids[chan_ids]
        if self.backend == "dft":
            updates["bmat"] = self.bmat[:, :, chan_ids]
            updates["uniform_freqs"] = is_uniform_grid(self.chan_freqs[chan_ids])
        elif self.spectrum.kind is SpectralKind.CUBE:
            updates["planes"] = self.planes[..., chan_ids]
        return replace(self, **updates)


# --------------------------------------------------------------------------- geometry


def pixel_lm(cel: WCS, ra0: float, dec0: float, i_pix, j_pix):
    """
    Direction cosines of image pixels relative to the phase centre.

    Exact for every projection: the pixel centres are pushed through the WCS and
    then converted with the same ``radec2lm`` the ASCII path uses.

    Parameters
    ----------
    cel : astropy.wcs.WCS
        Celestial sub-WCS of the image.
    ra0, dec0 : float
        Phase centre (radians).
    i_pix, j_pix : array_like
        0-based pixel indices along the longitude and latitude axes.

    Returns
    -------
    tuple of numpy.ndarray
        ``(l, m)``, each of shape ``i_pix.shape``.
    """
    lng, lat = cel.wcs.lng, cel.wcs.lat
    i_pix = np.atleast_1d(np.asarray(i_pix, dtype=np.float64))
    j_pix = np.atleast_1d(np.asarray(j_pix, dtype=np.float64))

    pixels = np.empty((i_pix.size, 2))
    pixels[:, lng] = i_pix.ravel()
    pixels[:, lat] = j_pix.ravel()
    world = cel.wcs_pix2world(pixels, 0)

    to_rad_lng = units.Unit(cel.wcs.cunit[lng]).to("rad")
    to_rad_lat = units.Unit(cel.wcs.cunit[lat]).to("rad")
    ra = world[:, lng] * to_rad_lng
    dec = world[:, lat] * to_rad_lat

    el, em = radec2lm(ra0, dec0, ra, dec)
    return el.reshape(i_pix.shape), em.reshape(j_pix.shape)


def lm_to_radec(el: np.ndarray, em: np.ndarray, ra0: float, dec0: float):
    """Inverse of :func:`simms.utilities.radec2lm`, in radians."""
    en = np.sqrt(np.maximum(1.0 - el * el - em * em, 0.0))
    dec = np.arcsin(em * np.cos(dec0) + en * np.sin(dec0))
    ra = ra0 + np.arctan2(el, en * np.cos(dec0) - em * np.sin(dec0))
    return ra, dec


def fit_lm_grid(cel: WCS, ra0: float, dec0: float, npix_l: int, npix_m: int, samples: int = 9) -> FitsGrid:
    """
    Fit ``l`` and ``m`` as linear functions of pixel index, and measure the misfit.

    The misfit is zero for a SIN projection tangent at the phase centre and grows
    with field size for any other projection, so it is the criterion for whether
    the gridder may be used at all.
    """
    ref_l, ref_m = npix_l // 2, npix_m // 2
    grid_l = np.unique(np.linspace(0, npix_l - 1, min(samples, npix_l)).astype(int))
    grid_m = np.unique(np.linspace(0, npix_m - 1, min(samples, npix_m)).astype(int))
    i_pix, j_pix = (a.ravel() for a in np.meshgrid(grid_l, grid_m, indexing="ij"))

    el, em = pixel_lm(cel, ra0, dec0, i_pix, j_pix)

    design_l = np.column_stack([np.ones(i_pix.size), i_pix - ref_l])
    design_m = np.column_stack([np.ones(j_pix.size), j_pix - ref_m])
    (l_ref, delta_l), *_ = np.linalg.lstsq(design_l, el, rcond=None)
    (m_ref, delta_m), *_ = np.linalg.lstsq(design_m, em, rcond=None)

    residual = max(
        np.abs(el - design_l @ [l_ref, delta_l]).max(),
        np.abs(em - design_m @ [m_ref, delta_m]).max(),
    )
    cell = min(abs(delta_l), abs(delta_m))
    return FitsGrid(
        delta_l=float(delta_l),
        delta_m=float(delta_m),
        ref_l=float(ref_l),
        ref_m=float(ref_m),
        l_ref=float(l_ref),
        m_ref=float(m_ref),
        deviation_pixels=float(residual / cell) if cell else np.inf,
    )


def attach_image_beam(
    prepared: PreparedFitsSky,
    provider,
    is_altaz: bool,
    ra0: float,
    dec0: float,
    lon: float,
    lat: float,
    t_start: float,
    duration: float,
    pa_step: float,
    mid_freq: float,
    phase_ra0: float | None = None,
    phase_dec0: float | None = None,
) -> PreparedFitsSky:
    """Multiply the apparent sky by a parallactic-angle-averaged power beam (in place).

    Approximate: one representative antenna beam, averaged over parallactic angle, is
    applied to every Stokes plane (or DFT component). This ignores per-baseline beams,
    heterogeneity and cross-hand leakage -- correct only for a homogeneous array. The
    beam is applied per channel where the model has a real channel axis (DFT components,
    a spectral CUBE), otherwise at the band mid-frequency (single FLAT/POLY plane).

    ``ra0``/``dec0`` are the beam (antenna pointing) centre; ``phase_ra0``/``phase_dec0`` are
    the phase centre the image ``l/m`` are referenced to, so the beam is sampled at each pixel's
    offset from where the dish points.
    """
    from simms.skymodel.beams import image_power_beam, pa_sample_grid, reproject_lm

    _, chi_grid = pa_sample_grid(t_start, duration, ra0, dec0, lon, lat, pa_step)

    if prepared.backend == "dft":
        ell, emm = prepared.lmn[:, 0], prepared.lmn[:, 1]
        if phase_ra0 is not None:
            ell, emm = reproject_lm(ell, emm, phase_ra0, phase_dec0, ra0, dec0)
        power = image_power_beam(provider, is_altaz, ell, emm, prepared.chan_freqs, chi_grid)
        prepared.bmat *= power[:, None, :]  # (ncomp, nspec, nchan)
        return prepared

    npix_l, npix_m, nchan_model = prepared.planes.shape[1], prepared.planes.shape[2], prepared.planes.shape[3]
    i_pix, j_pix = (a.ravel() for a in np.meshgrid(np.arange(npix_l), np.arange(npix_m), indexing="ij"))
    lmn = prepared.grid.pixel_lmn(i_pix, j_pix)
    ell, emm = lmn[:, 0], lmn[:, 1]
    if phase_ra0 is not None:
        ell, emm = reproject_lm(ell, emm, phase_ra0, phase_dec0, ra0, dec0)
    per_channel = nchan_model == prepared.chan_freqs.size
    freqs = prepared.chan_freqs if per_channel else np.array([mid_freq])
    power = image_power_beam(provider, is_altaz, ell, emm, freqs, chi_grid)
    prepared.planes *= power.reshape(npix_l, npix_m, freqs.size)[None]
    return prepared


@dataclass
class SinResampler:
    """A fixed mapping from a SIN target grid onto an arbitrary source image.

    Built once so that flux maps and spectral coefficient maps are resampled onto
    exactly the same target pixels, differing only in whether the Jacobian is
    applied.
    """

    coords: np.ndarray  # (2, out_l * out_m) source pixel coordinates
    jacobian: np.ndarray  # (out_l, out_m) |det J| of target -> source pixels
    shape: tuple  # (out_l, out_m)
    grid: FitsGrid
    order: int = 3

    def __call__(self, planes: np.ndarray, conserve_flux: bool = True) -> np.ndarray:
        """
        Resample ``(..., npix_l, npix_m)`` onto the target grid.

        Set ``conserve_flux`` for a density in pixel index, such as Jy/pixel: the
        interpolated value is scaled by ``|det J|``, so the total flux survives a
        change of pixel area. Clear it for an intensive quantity, such as a
        spectral index, which must be carried across unscaled.
        """
        source_shape = planes.shape[-2:]
        flat = planes.reshape(-1, *source_shape)
        out = np.empty((flat.shape[0], *self.shape), dtype=np.float64)
        for index, plane in enumerate(flat):
            resampled = map_coordinates(plane, self.coords, order=self.order, mode="constant", cval=0.0)
            out[index] = resampled.reshape(self.shape)
            if conserve_flux:
                out[index] *= self.jacobian
        return out.reshape(planes.shape[:-2] + self.shape)


def sin_resampler(cel: WCS, ra0: float, dec0: float, cell: float, npix_l: int, npix_m: int, order: int = 3):
    """
    Build a resampler onto a SIN grid tangent at the phase centre.

    Flux is conserved exactly (to interpolation accuracy) without any solid-angle
    bookkeeping: Jy/pixel is a density with respect to pixel index, so scaling the
    interpolated value by the Jacobian determinant of the target-to-source pixel
    map accounts for the change of pixel area.

    Parameters
    ----------
    cel : astropy.wcs.WCS
        Celestial sub-WCS of the source image.
    ra0, dec0 : float
        Phase centre (radians).
    cell : float
        Pixel size of the target grid, in radians.
    npix_l, npix_m : int
        Shape of the source image.
    order : int, optional
        Spline order for :func:`scipy.ndimage.map_coordinates`. Default 3.

    Returns
    -------
    SinResampler
    """
    lng, lat = cel.wcs.lng, cel.wcs.lat

    # Angular extent of the source, sampled on its border.
    edge = np.linspace(0, npix_l - 1, 64).astype(int)
    edge_m = np.linspace(0, npix_m - 1, 64).astype(int)
    border_i = np.concatenate([edge, edge, np.zeros_like(edge_m), np.full_like(edge_m, npix_l - 1)])
    border_j = np.concatenate([np.zeros_like(edge), np.full_like(edge, npix_m - 1), edge_m, edge_m])
    el, em = pixel_lm(cel, ra0, dec0, border_i, border_j)

    half_l = int(np.ceil(max(abs(el).max(), cell) / cell)) + 2
    half_m = int(np.ceil(max(abs(em).max(), cell) / cell)) + 2
    out_l, out_m = 2 * half_l, 2 * half_m

    # Target grid: l decreases with index (as a negative CDELT1 does), m increases.
    delta_l, delta_m = -cell, cell
    ii = (np.arange(out_l) - half_l)[:, None]
    jj = (np.arange(out_m) - half_m)[None, :]
    target_l = np.broadcast_to(delta_l * ii, (out_l, out_m))
    target_m = np.broadcast_to(delta_m * jj, (out_l, out_m))

    ra, dec = lm_to_radec(target_l, target_m, ra0, dec0)
    world = np.empty((ra.size, 2))
    world[:, lng] = np.rad2deg(ra.ravel()) * units.deg.to(cel.wcs.cunit[lng])
    world[:, lat] = np.rad2deg(dec.ravel()) * units.deg.to(cel.wcs.cunit[lat])
    source = cel.wcs_world2pix(world, 0)
    src_i = source[:, lng].reshape(out_l, out_m)
    src_j = source[:, lat].reshape(out_l, out_m)

    # |det J| of the target-pixel -> source-pixel map, by central differences.
    di_dl, di_dm = np.gradient(src_i)
    dj_dl, dj_dm = np.gradient(src_j)
    jacobian = np.abs(di_dl * dj_dm - di_dm * dj_dl)

    grid = FitsGrid(
        delta_l=delta_l,
        delta_m=delta_m,
        ref_l=float(half_l),
        ref_m=float(half_m),
        l_ref=0.0,
        m_ref=0.0,
        deviation_pixels=0.0,
    )
    return SinResampler(
        coords=np.stack([src_i.ravel(), src_j.ravel()]),
        jacobian=jacobian,
        shape=(out_l, out_m),
        grid=grid,
        order=order,
    )


def reproject_to_sin(planes: np.ndarray, cel: WCS, ra0: float, dec0: float, cell: float, order: int = 3):
    """Resample flux planes onto a SIN grid tangent at the phase centre."""
    resampler = sin_resampler(cel, ra0, dec0, cell, planes.shape[-2], planes.shape[-1], order=order)
    return resampler(planes, conserve_flux=True), resampler.grid


# --------------------------------------------------------------------------- polarisation


def stokes_to_correlations(get, ncorr: int, polarisation: bool, linear_basis: bool) -> list:
    """
    Combine Stokes quantities into correlations.

    ``get(name)`` returns the I, Q, U or V quantity, or 0 when the model does not
    carry it. The combination is linear, so it may be applied either to image
    planes or, equivalently, to the visibilities they predict.
    """
    if ncorr not in (2, 4):
        raise ValueError(f"Only two or four correlations allowed, but {ncorr} were requested.")

    stokes_i = get("I")
    if not polarisation:
        zero = np.zeros_like(stokes_i)
        return [stokes_i, stokes_i] if ncorr == 2 else [stokes_i, zero, zero, stokes_i]

    stokes_q, stokes_u, stokes_v = get("Q"), get("U"), get("V")
    if linear_basis:
        corrs = [stokes_i + stokes_q, stokes_u + 1j * stokes_v, stokes_u - 1j * stokes_v, stokes_i - stokes_q]
    else:
        corrs = [stokes_i + stokes_v, stokes_q + 1j * stokes_u, stokes_q - 1j * stokes_u, stokes_i - stokes_v]

    return [corrs[0], corrs[3]] if ncorr == 2 else corrs


def _stokes_getter(planes: np.ndarray, names: list[str]):
    index = {name: k for k, name in enumerate(names)}

    def get(name):
        if name in index:
            return planes[index[name]]
        return np.zeros_like(planes[0])

    return get


# --------------------------------------------------------------------------- backend choice


def choose_backend(ncomp: int, npix_l: int, npix_m: int, nrow: int, nchan: int, nplanes: int) -> str:
    """
    Pick the cheaper of the two exact backends.

    The DFT costs one phasor per component, row and channel. The gridder costs an
    image FFT per plane and channel, plus a degridding term proportional to rows.
    Break-even is therefore a few hundred components, not a fraction of the image:
    a "sparse" CLEAN model with thousands of components still wants the gridder.
    """
    npix = npix_l * npix_m
    side = max(npix_l, npix_m, 2)
    dft = DFT_SECONDS_PER_COMPONENT_ROW_CHAN * ncomp * nrow * nchan
    fft = (
        nplanes
        * nchan
        * (FFT_SECONDS_PER_PIXEL_LOGPIXEL_CHAN * npix * np.log2(side) + FFT_SECONDS_PER_ROW_CHAN_PER_SIDE * side * nrow)
    )
    return "dft" if dft < fft else "fft"


# --------------------------------------------------------------------------- reading


def _ensure_axis(fds: FitsData, name: str, crval: float, cdelt: float, cunit: str):
    if name not in fds.coord_names:
        fds.add_axis(name, fds.ndim + 1, crval=crval, cdelt=cdelt, crpix=0, cunit=cunit)


def _fits_frequencies(fds: FitsData):
    """Channel centres and width of the FITS spectral axis, in Hz."""
    if fds.spectral_coord == "VRAD":
        freqs = fds.get_freq_from_vrad()
        step = fds.coords["VRAD"].pixel_size * getattr(units, fds.coords["VRAD"].units)
        width = step.to(units.Hz, doppler_rest=fds.spectral_restfreq * units.Hz, doppler_convention="radio").value
    elif fds.spectral_coord == "VOPT":
        freqs = fds.get_freq_from_vopt()
        step = fds.coords["VOPT"].pixel_size * getattr(units, fds.coords["VOPT"].units)
        width = step.to(units.Hz, doppler_rest=fds.spectral_restfreq * units.Hz, doppler_convention="optical").value
    else:
        freqs = np.squeeze(np.asarray(fds.coords["FREQ"].data))
        width = fds.coords["FREQ"].pixel_size
    return np.atleast_1d(freqs).astype(np.float64), float(width)


def _beam_axis_radians(column) -> np.ndarray:
    """Beam axis in radians. FITS beam keywords are degrees unless a unit says otherwise."""
    unit = getattr(column, "unit", None) or units.deg
    return np.atleast_1d((np.asarray(column) * unit).to("rad").value)


def _jy_per_pixel(cube: np.ndarray, fds: FitsData, pixel_area: float) -> np.ndarray:
    """Convert the image to Jy/pixel, using the beam table when BUNIT is Jy/beam."""
    if fds.data_units == "Jy/beam":
        if fds.beam_table is None:
            log.warning("FITS sky model is in Jy/beam but carries no beam information. Assuming Jy/pixel.")
            return cube
        bmaj = _beam_axis_radians(fds.beam_table["BMAJ"])
        bmin = _beam_axis_radians(fds.beam_table["BMIN"])
        beam_area = (np.pi * bmaj * bmin) / (4 * np.log(2))
        pixels_per_beam = beam_area / pixel_area
        return cube / pixels_per_beam[np.newaxis, np.newaxis, np.newaxis, :]

    if fds.data_units == "":
        log.warning("FITS sky model has no BUNIT. Assuming Jy/pixel.")
    elif fds.data_units not in ("Jy", "Jy/pixel"):
        log.warning(f"FITS sky model has unknown BUNIT='{fds.data_units}'. Assuming Jy/pixel.")
    return cube


def _interpolate_spectrum(cube, fits_freqs, chan_freqs, method):
    """Resample the spectral axis of ``(nstokes, l, m, nchan_fits)`` onto the MS grid."""
    if fits_freqs.size == len(chan_freqs) and np.allclose(fits_freqs, chan_freqs):
        return cube
    log.warning("Interpolating the FITS sky model onto the MS frequency grid.")
    interp = interp1d(fits_freqs, cube, kind=method, axis=-1, bounds_error=False, fill_value="extrapolate")
    return interp(chan_freqs)


# --------------------------------------------------------------------------- preparation


def prepare_fits_sky(
    input_fitsimages: str | list[str],
    ra0: float,
    dec0: float,
    chan_freqs: np.ndarray,
    ms_delta_nu: float,
    ncorr: int,
    nrow: int,
    linear_basis: bool = True,
    polarisation: bool = True,
    tol: float = 1e-7,
    backend: str = "auto",
    spectrum: str = "auto",
    spi_maps: list[str] | None = None,
    ref_freq: float | None = None,
    spectrum_order: int = 2,
    spectrum_tol: float = DEFAULT_FIT_TOLERANCE,
    interpolation: str = "linear",
    stack_axis: str = "STOKES",
    reproject_order: int = 3,
) -> PreparedFitsSky:
    """
    Read one or more FITS images into a form a prediction backend can consume.

    Parameters
    ----------
    input_fitsimages : File or list of File
        A single FITS image, or an ordered list of per-Stokes images.
    ra0, dec0 : float
        MS phase centre (radians).
    chan_freqs : numpy.ndarray
        MS channel centres (Hz).
    ms_delta_nu : float
        MS channel width (Hz).
    ncorr : int
        Number of correlations to predict (2 or 4).
    nrow : int
        Total rows in the MS. Used only by the backend cost model.
    linear_basis : bool, optional
        Linear (XX/XY/YX/YY) or circular (RR/RL/LR/LL) correlations.
    polarisation : bool, optional
        Predict every correlation. Forced off when the model carries only Stokes I.
    tol : float, optional
        Pixels below this brightness (Jy) are dropped from the component list.
    backend : {"auto", "dft", "fft"}, optional
        Prediction backend. "auto" uses :func:`choose_backend`.
    spectrum : {"auto", "flat", "poly", "cube"}, optional
        How the model varies with frequency. "auto" takes a single plane as flat,
        and for a cube fits a log-polynomial, falling back to the cube when the fit
        residual exceeds `spectrum_tol`.
    spi_maps : list of File, optional
        Spectral-index (and higher-order) coefficient maps, ordered c1, c2, ...
        When given, the spectrum is analytic and nothing is fitted.
    ref_freq : float, optional
        Reference frequency of the analytic spectrum (Hz). Defaults to the MS band
        centre.
    spectrum_order : int, optional
        Order of the fitted log-polynomial. 1 is a plain spectral index. Default 2.
    spectrum_tol : float, optional
        Largest fractional flux residual for which a fitted log-polynomial is
        accepted under `spectrum="auto"`.
    interpolation : str, optional
        Spectral interpolation method when the FITS and MS channel grids differ and
        the cube is kept.
    stack_axis : str, optional
        Axis along which a list of FITS images is stacked.
    reproject_order : int, optional
        Spline order used when reprojecting a non-SIN image.

    Returns
    -------
    PreparedFitsSky

    Raises
    ------
    FITSSkymodelError
        If the MS band lies outside the FITS band, or an unsupported Stokes axis
        or backend is requested.
    """
    log = logging.getLogger(BIN.skysim)
    chan_freqs = np.ascontiguousarray(chan_freqs, dtype=np.float64)
    nchan = chan_freqs.size

    files = [input_fitsimages] if isinstance(input_fitsimages, str) else list(input_fitsimages)

    ms_start_freq = chan_freqs[0] - 0.5 * ms_delta_nu
    ms_end_freq = chan_freqs[-1] + 0.5 * ms_delta_nu

    fds = FitsData(files[0])
    # A flat image is a frequency-independent sky; give it one channel spanning the band.
    _ensure_axis(fds, "FREQ", crval=0.5 * (ms_start_freq + ms_end_freq), cdelt=ms_end_freq - ms_start_freq, cunit="Hz")
    _ensure_axis(fds, stack_axis, crval=1, cdelt=1, cunit="")
    if len(files) > 1:
        fds.expand_along_axis_from_files(stack_axis, files[1:])

    stokes_codes = np.atleast_1d(np.squeeze(np.asarray(fds.coords["STOKES"].data))).astype(int)
    unknown = [int(c) for c in stokes_codes if int(c) not in STOKES_CODES]
    if unknown:
        raise FITSSkymodelError(f"Unsupported FITS STOKES codes {unknown}. Only 1..4 (I, Q, U, V) are supported.")
    stokes_names = [STOKES_CODES[int(c)] for c in stokes_codes]
    if "I" not in stokes_names:
        raise FITSSkymodelError(f"FITS sky model has no Stokes I plane (found {stokes_names}).")

    fits_freqs, fits_delta_nu = _fits_frequencies(fds)
    nchan_fits = fits_freqs.size
    # Widen by a millionth of a channel before the containment test. The most natural
    # case -- a cube written on the MS's own frequency grid -- has the two band edges
    # equal in exact arithmetic but differing in the last ulp, because the MS width
    # comes from CHAN_WIDTH and the FITS width from CDELT3. Without this, such a cube
    # is rejected over a discrepancy of ~1e-7 Hz at 1.4 GHz, with an error message
    # that prints the two ranges identically. A genuine mismatch is off by a
    # meaningful fraction of a channel, far above this.
    edge_tol = 1e-6 * abs(fits_delta_nu)
    fits_range = (fits_freqs[0] - 0.5 * fits_delta_nu - edge_tol, fits_freqs[-1] + 0.5 * fits_delta_nu + edge_tol)
    ms_range = (ms_start_freq, ms_end_freq)
    if nchan_fits > 1 and not is_range_in_range(ms_range, fits_range):
        raise FITSSkymodelError(
            f"MS frequencies [{ms_range[0] / 1e9:.6f} GHz, {ms_range[1] / 1e9:.6f} GHz] are outside the "
            f"FITS image frequencies [{fits_range[0] / 1e9:.6f} GHz, {fits_range[1] / 1e9:.6f} GHz]. "
            "Cannot interpolate the FITS image onto the MS frequency grid."
        )

    cel = fds.wcs.celestial
    npix_l = fds.coords["RA"].size
    npix_m = fds.coords["DEC"].size
    stokes_i = stokes_names.index("I")

    grid = fit_lm_grid(cel, ra0, dec0, npix_l, npix_m)
    pixel_area = abs(grid.delta_l * grid.delta_m)

    # Kept lazy: a POLY fit streams over channel blocks and never holds the cube.
    lazy_cube = fds.get_xds(transpose=["STOKES", "RA", "DEC", "FREQ"]).data
    lazy_cube = _jy_per_pixel(lazy_cube, fds, pixel_area)

    if spi_maps and ref_freq is None:
        raise FITSSkymodelError(
            "Supplied spectral-index maps are defined at a particular frequency, which cannot be guessed. "
            "Set a reference frequency (--fits-ref-freq, in Hz)."
        )
    if ref_freq is None:
        ref_freq = float(0.5 * (ms_start_freq + ms_end_freq))

    planes, spectrum, peak = _resolve_spectrum(
        lazy_cube,
        fits_freqs,
        chan_freqs,
        ref_freq,
        stokes_i,
        spectrum=spectrum,
        spi_maps=spi_maps,
        order=spectrum_order,
        fit_tol=spectrum_tol,
        tol=tol,
        interpolation=interpolation,
        shape=(npix_l, npix_m),
        log=log,
    )
    fds.close()

    polarisation = bool(polarisation) and len(stokes_names) > 1

    support = peak > tol
    ncomp = int(support.sum())
    if ncomp == 0:
        log.warning(f"No FITS pixel exceeds the {tol * 1e6:.2f}-uJy tolerance; the model is empty.")

    nplanes = len(stokes_names) if polarisation else 1
    is_cube = spectrum.kind is SpectralKind.CUBE
    if backend == "auto":
        # A cube may be sparse channel by channel (a spectral line), so it is
        # predicted per channel; a single reference plane is not.
        backend = "perchan" if is_cube else choose_backend(ncomp, npix_l, npix_m, nrow, nchan, nplanes)
    elif backend not in ("dft", "fft", "perchan"):
        raise FITSSkymodelError(f"Unknown predict backend '{backend}'. Choose from auto, dft, fft, perchan.")

    if backend == "perchan" and not is_cube:
        log.info("Per-channel prediction only helps a spectral cube; using the gridder for this single-plane model.")
        backend = "fft"

    if backend in ("fft", "perchan") and (npix_l % 2 or npix_m % 2):
        log.warning(f"Image is {npix_l}x{npix_m}; the gridder needs even dimensions. Using the DFT instead.")
        backend = "dft"

    prepared = PreparedFitsSky(
        chan_freqs=chan_freqs,
        spectrum=spectrum,
        backend=backend,
        ncorr=ncorr,
        polarisation=polarisation,
        linear_basis=linear_basis,
        ncomp=ncomp,
        npix_l=npix_l,
        npix_m=npix_m,
    )

    if backend == "dft":
        log.info(f"Predicting from {ncomp} FITS pixels with the DFT (image is {npix_l}x{npix_m}).")
        i_pix, j_pix = np.nonzero(support)
        el, em = pixel_lm(cel, ra0, dec0, i_pix, j_pix)
        lmn = np.empty((ncomp, 3))
        lmn[:, 0], lmn[:, 1] = el, em
        lmn[:, 2] = np.sqrt(np.maximum(1 - el * el - em * em, 0.0)) - 1

        # (nstokes, ncomp, nchan_model) -> (nstokes, ncomp, nchan) on the MS grid
        comps = planes[:, i_pix, j_pix, :]
        if spectrum.kind is SpectralKind.CUBE:
            comps_at_chan = comps
        else:
            # (nchan, ncomp) for POLY, scalar 1.0 for FLAT
            scale = spectrum.scale_at_pixels(chan_freqs, i_pix, j_pix)
            comps_at_chan = comps[..., 0][..., np.newaxis] * np.atleast_2d(scale).T

        corrs = stokes_to_correlations(_stokes_getter(comps_at_chan, stokes_names), ncorr, polarisation, linear_basis)
        nspec = prepared.nspec
        bmat = np.zeros((ncomp, nspec, nchan), dtype=np.complex128)
        for corr in range(nspec):
            bmat[:, corr, :] = corrs[corr]
        prepared.lmn = lmn
        prepared.bmat = bmat
        prepared.uniform_freqs = is_uniform_grid(chan_freqs)
    else:
        if not grid.is_regular:
            log.info(
                f"Image (l, m) grid departs from a regular grid by {grid.deviation_pixels:.3f} pixels; "
                "reprojecting onto a SIN grid tangent at the phase centre."
            )
            cell = min(abs(grid.delta_l), abs(grid.delta_m))
            resampler = sin_resampler(cel, ra0, dec0, cell, npix_l, npix_m, order=reproject_order)
            planes = np.moveaxis(planes, (1, 2), (-2, -1))  # (nstokes, nchan_model, l, m)
            planes = np.moveaxis(resampler(planes, conserve_flux=True), (-2, -1), (1, 2))
            if spectrum.kind is SpectralKind.POLY:
                # Spectral coefficients are exponents, not densities: no Jacobian.
                spectrum.coeffs = resampler(spectrum.coeffs, conserve_flux=False)
            grid = resampler.grid
            npix_l, npix_m = planes.shape[1], planes.shape[2]
            prepared.npix_l, prepared.npix_m = npix_l, npix_m

        if backend == "perchan":
            log.info(f"Predicting from a {npix_l}x{npix_m} cube per channel (image is {npix_l}x{npix_m}).")
        else:
            log.info(f"Predicting from a {npix_l}x{npix_m} FITS image with the gridder.")
        keep = stokes_names if polarisation else ["I"]
        index = [stokes_names.index(name) for name in keep]
        prepared.planes = np.ascontiguousarray(planes[index])
        prepared.stokes_names = keep
        prepared.grid = grid
        prepared.tol = tol

    return prepared


def _resolve_spectrum(
    lazy_cube,
    fits_freqs,
    chan_freqs,
    ref_freq,
    stokes_i,
    spectrum,
    spi_maps,
    order,
    fit_tol,
    tol,
    interpolation,
    shape,
    log,
):
    """
    Decide how the model varies with frequency, and return the planes it needs.

    Returns ``(planes, spectrum, peak)``, where ``planes`` is
    ``(nstokes, npix_l, npix_m, nchan_model)`` — one plane at ``ref_freq`` for FLAT
    and POLY, the whole MS channel grid for CUBE — and ``peak`` is the peak absolute
    brightness per pixel, used to select components.
    """
    nchan_fits = fits_freqs.size

    if spi_maps:
        # An intensity map plus supplied spectral-index maps: nothing to fit.
        coeffs = read_spi_maps(spi_maps, shape)
        if nchan_fits == 1:
            planes = np.asarray(lazy_cube[..., 0], dtype=np.float64)
        else:
            planes = interpolate_planes_at(lazy_cube, fits_freqs, ref_freq)
        log.info(f"Using {len(spi_maps)} supplied spectral coefficient map(s), reference {ref_freq / 1e9:.6f} GHz.")
        model = FitsSpectrum(kind=SpectralKind.POLY, ref_freq=ref_freq, coeffs=coeffs)
        return planes[..., np.newaxis], model, np.abs(planes).max(axis=0)

    if nchan_fits == 1:
        if spectrum == "poly":
            raise FITSSkymodelError(
                "A 'poly' spectrum needs either a multi-channel FITS cube to fit, or supplied "
                "spectral-index maps. The image has a single frequency plane."
            )
        planes = np.asarray(lazy_cube[..., 0], dtype=np.float64)
        log.info("FITS sky model has one frequency plane; treating the spectrum as flat.")
        return planes[..., np.newaxis], FitsSpectrum(kind=SpectralKind.FLAT, ref_freq=ref_freq), np.abs(planes).max(0)

    if spectrum == "flat":
        raise FITSSkymodelError(
            f"A 'flat' spectrum needs a single frequency plane, but the FITS cube has {nchan_fits}."
        )

    if spectrum in ("auto", "poly"):
        # A polynomial of order k needs k + 1 channels. Under "auto" a short cube
        # simply gets a lower order, or the cube itself.
        fit_order = min(order, nchan_fits - 1)
        if fit_order < 1 and spectrum == "poly":
            raise FITSSkymodelError(
                f"A 'poly' spectrum needs at least two FITS channels to fit, but the cube has {nchan_fits}."
            )
        if fit_order < order:
            log.info(f"The cube has {nchan_fits} channels; fitting an order-{fit_order} spectrum instead of {order}.")

        if fit_order >= 1:
            ref_i, coeffs, residual, coverage, peak = fit_log_polynomial(
                lazy_cube, fits_freqs, ref_freq, stokes_i, order=fit_order, tol=tol
            )
            # A log-polynomial represents only pixels positive at every channel, so
            # under "auto" it is accepted only when it both fits well and covers
            # essentially all the emission - a spectral line fails the coverage test
            # and keeps the cube. An explicit "poly" is honoured regardless.
            well_fit = residual <= fit_tol and coverage >= 0.99
            if spectrum == "poly" or well_fit:
                planes = interpolate_planes_at(lazy_cube, fits_freqs, ref_freq)
                planes[stokes_i] = ref_i  # Stokes I comes from the fit, which is more accurate
                log.info(
                    f"Fitted an order-{fit_order} log-polynomial spectrum, reference {ref_freq / 1e9:.6f} GHz, "
                    f"worst fractional residual {residual:.2e}, covering {coverage * 100:.1f}% of the emission."
                )
                model = FitsSpectrum(kind=SpectralKind.POLY, ref_freq=ref_freq, coeffs=coeffs, residual=residual)
                return planes[..., np.newaxis], model, peak
            log.info(
                f"An order-{fit_order} log-polynomial fits {coverage * 100:.1f}% of the emission to residual "
                f"{residual:.2e} (tolerance {fit_tol:.1e}); keeping the cube."
            )

    cube = np.asarray(lazy_cube, dtype=np.float64)
    cube = _interpolate_spectrum(cube, fits_freqs, chan_freqs, interpolation)
    return cube, FitsSpectrum(kind=SpectralKind.CUBE, ref_freq=ref_freq), np.abs(cube).max(axis=(0, 3))


def component_sky_from_fits_dft(prepared: PreparedFitsSky):
    """Re-express a DFT-backend FITS model as a component :class:`~simms.skymodel.mstools.PreparedSky`.

    The DFT backend already holds per-component ``lmn`` and a brightness matrix,
    which is exactly what the component-sky kernels consume. Bridging to a
    ``PreparedSky`` lets a FITS model with few bright pixels use the *exact*
    per-antenna beam kernels (``predict_vis_beam``/``predict_vis_jones``) via
    :func:`~simms.skymodel.mstools.attach_beam`, instead of an approximate
    image-domain beam.
    """
    from simms.skymodel.mstools import PreparedSky

    if prepared.backend != "dft":
        raise ValueError(f"component-sky bridging needs the dft backend, not {prepared.backend!r}")
    if not prepared.linear_basis:
        # The bridge exists to attach a beam, and the beam kernels consume a
        # linear-feed-basis coherency (to_full_corr: "beams refuse circular").
        # A circular-basis model would silently produce wrong cross-hands.
        raise ValueError(
            "component-sky bridging is for attaching a primary beam, which needs a linear-feed-basis "
            "model; prepare the FITS sky with linear_basis=True."
        )
    ncomp = prepared.ncomp
    return PreparedSky(
        lmn=prepared.lmn if ncomp else np.zeros((0, 3)),
        gauss_shape=np.zeros((ncomp, 3)),
        is_gauss=np.zeros(ncomp, dtype=np.bool_),
        bmat=prepared.bmat if ncomp else np.zeros((0, prepared.nspec, prepared.chan_freqs.size), dtype=np.complex128),
        lightcurve=np.ones((ncomp, 1)),
        unique_times=None,
        freqs=prepared.chan_freqs,
        uniform_freqs=prepared.uniform_freqs,
        ncorr=prepared.ncorr,
        polarisation=prepared.polarisation,
        smearing=prepared.smearing,
    )


# --------------------------------------------------------------------------- prediction


def _fft_stokes_visibilities(plane, spectrum, grid, uvw, chan_freqs, npix_l, npix_m, epsilon, wgridding, nthreads):
    """
    Visibilities of one real Stokes plane stack.

    ``plane`` is ``(npix_l, npix_m, nchan_model)``: a single reference-frequency plane
    for FLAT and POLY, the whole channel grid for CUBE. A POLY image is synthesised
    channel by channel from the reference plane and the coefficient maps, so the
    cube is never held.
    """
    nrow, nchan = uvw.shape[0], chan_freqs.size
    vis = np.zeros((nrow, nchan), dtype=np.complex128)
    kwargs = dict(
        uvw=uvw, epsilon=epsilon, do_wgridding=wgridding, nthreads=nthreads, **grid.ducc_kwargs(npix_l, npix_m)
    )

    # One frequency-independent image: the gridder handles every channel in one pass.
    if spectrum.kind is SpectralKind.FLAT:
        image = np.ascontiguousarray(plane[:, :, 0])
        if np.any(image):
            vis[:] = dirty2vis(freq=chan_freqs, dirty=image, **kwargs)
        return vis

    for chan in range(nchan):
        if spectrum.kind is SpectralKind.POLY:
            scale = evaluate_scale(spectrum.coeffs, chan_freqs[chan : chan + 1], spectrum.ref_freq)[0]
            image = np.ascontiguousarray(plane[:, :, 0] * scale)
        else:
            image = np.ascontiguousarray(plane[:, :, chan])
        # A channel with no emission needs no gridding pass at all.
        if not np.any(image):
            continue
        vis[:, chan] = dirty2vis(freq=chan_freqs[chan : chan + 1], dirty=image, **kwargs)[:, 0]
    return vis


def _perchan_visibilities(prepared, uvw, epsilon, wgridding, nthreads):
    """
    Predict a spectral cube one channel at a time, choosing a backend per channel.

    Each channel of a line cube may hold anything from no emission to a full
    image, so the global backend choice is wrong for most of them. Here an empty
    channel costs nothing, a channel with few bright pixels is predicted by DFT,
    and a dense channel is gridded.
    """
    nrow, nchan, ncorr = uvw.shape[0], prepared.chan_freqs.size, prepared.ncorr
    nspec = prepared.nspec
    npix_l, npix_m = prepared.npix_l, prepared.npix_m
    nplanes = len(prepared.stokes_names)
    planes = prepared.planes  # (nplanes, npix_l, npix_m, nchan)

    vis = np.zeros((nrow, nchan, ncorr), dtype=np.complex128)
    ducc = dict(
        uvw=uvw, epsilon=epsilon, do_wgridding=wgridding, nthreads=nthreads, **prepared.grid.ducc_kwargs(npix_l, npix_m)
    )

    for chan in range(nchan):
        plane = planes[:, :, :, chan]  # (nplanes, npix_l, npix_m)
        active = np.abs(plane).max(axis=0) > prepared.tol
        ncomp = int(active.sum())
        if ncomp == 0:
            continue

        if choose_backend(ncomp, npix_l, npix_m, nrow, 1, nplanes) == "dft":
            i_pix, j_pix = np.nonzero(active)
            lmn = prepared.grid.pixel_lmn(i_pix.astype(np.float64), j_pix.astype(np.float64))
            comps = plane[:, i_pix, j_pix]  # (nplanes, ncomp)
            corrs = stokes_to_correlations(
                _stokes_getter(comps, prepared.stokes_names), ncorr, prepared.polarisation, prepared.linear_basis
            )
            bmat = np.zeros((ncomp, nspec, 1), dtype=np.complex128)
            for corr in range(nspec):
                bmat[:, corr, 0] = corrs[corr]
            chan_vis = np.zeros((nrow, 1, nspec), dtype=np.complex128)
            predict_vis(
                uvw,
                prepared.chan_freqs[chan : chan + 1],
                True,
                lmn,
                np.zeros((ncomp, 3)),
                np.zeros(ncomp, dtype=np.bool_),
                bmat,
                np.ones((ncomp, 1)),
                np.zeros(nrow, dtype=np.int64),
                chan_vis,
                # The gridder serves the dense channels of a perchan model, so
                # smearing is refused for the backend as a whole (see skysim).
                *smear_kernel_args(None, uvw, None),
            )
            if nspec != ncorr:
                chan_vis = stack_unpolarised_vis(chan_vis[..., 0], ncorr)
            vis[:, chan, :] = chan_vis[:, 0, :]
        else:
            stokes_vis = np.empty((nplanes, nrow), dtype=np.complex128)
            for s in range(nplanes):
                stokes_vis[s] = dirty2vis(
                    freq=prepared.chan_freqs[chan : chan + 1], dirty=np.ascontiguousarray(plane[s]), **ducc
                )[:, 0]
            corrs = stokes_to_correlations(
                _stokes_getter(stokes_vis, prepared.stokes_names), ncorr, prepared.polarisation, prepared.linear_basis
            )
            vis[:, chan, :] = np.stack(corrs, axis=-1)

    return vis


def predict_fits_channel_block(
    prepared: PreparedFitsSky,
    uvw: np.ndarray,
    chan_ids: np.ndarray,
    times: np.ndarray = None,
    antenna1: np.ndarray = None,
    antenna2: np.ndarray = None,
    exposure: np.ndarray = None,
    out_dtype: np.dtype = None,
    epsilon: float = 1e-7,
    do_wgridding: bool = True,
    nthreads: int = 1,
) -> np.ndarray:
    """Predict one (row, channel) block, restricting the model to ``chan_ids``."""
    return predict_fits_block(
        prepared.select_channels(chan_ids),
        uvw,
        times=times,
        antenna1=antenna1,
        antenna2=antenna2,
        exposure=exposure,
        out_dtype=out_dtype,
        epsilon=epsilon,
        do_wgridding=do_wgridding,
        nthreads=nthreads,
    )


def predict_fits_block(
    prepared: PreparedFitsSky,
    uvw: np.ndarray,
    times: np.ndarray = None,
    antenna1: np.ndarray = None,
    antenna2: np.ndarray = None,
    exposure: np.ndarray = None,
    noise_vis: float | None = None,
    out_dtype: np.dtype = None,
    epsilon: float = 1e-7,
    do_wgridding: bool = True,
    nthreads: int = 1,
    seed=None,
) -> np.ndarray:
    """
    Predict visibilities for one block of rows.

    Parameters
    ----------
    prepared : PreparedFitsSky
        Sky model from :func:`prepare_fits_sky`.
    uvw : numpy.ndarray
        UVW coordinates of shape ``(nrow, 3)``, in metres.
    times, antenna1, antenna2 : numpy.ndarray, optional
        Per-row timestamp and antenna indices. Required when the model carries
        an a-term (``prepared.aterm``); ignored otherwise.
    exposure : numpy.ndarray, optional
        Integration time per row (the MS ``EXPOSURE`` column, seconds). Required
        when the model carries a :class:`~simms.skymodel.smearing.Smearing`,
        which only the DFT backend supports.
    noise_vis : float, optional
        RMS noise per visibility (Jy).
    out_dtype : numpy.dtype, optional
        Complex dtype to cast to. Named ``out_dtype`` because ``da.blockwise``
        consumes any ``dtype`` kwarg itself and never forwards it.
    epsilon : float, optional
        Gridder accuracy.
    do_wgridding : bool, optional
        Apply the w term with w-gridding.
    nthreads : int, optional
        Threads for the gridder. Default 1; dask supplies row parallelism.
    seed : optional
        Seed for the noise; see :func:`simms.skymodel.mstools.sim_noise`. ``None`` draws
        fresh entropy. When looping over blocks, pass one shared
        ``numpy.random.Generator`` rather than a repeated integer, or every block gets
        the same realisation.

    Returns
    -------
    numpy.ndarray
        Visibilities of shape ``(nrow, nchan, ncorr)``.
    """
    uvw = np.ascontiguousarray(uvw, dtype=np.float64)
    nrow, nchan, ncorr = uvw.shape[0], prepared.chan_freqs.size, prepared.ncorr

    if prepared.aterm is not None:
        from simms.skymodel.aterms import predict_aterm_block

        vis = predict_aterm_block(prepared, uvw, times, antenna1, antenna2, epsilon, do_wgridding, nthreads)
    elif prepared.backend == "dft":
        vis = np.zeros((nrow, nchan, prepared.nspec), dtype=np.complex128)
        if prepared.ncomp:
            ncomp = prepared.ncomp
            predict_vis(
                uvw,
                prepared.chan_freqs,
                prepared.uniform_freqs,
                prepared.lmn,
                np.zeros((ncomp, 3)),
                np.zeros(ncomp, dtype=np.bool_),
                prepared.bmat,
                np.ones((ncomp, 1)),
                np.zeros(nrow, dtype=np.int64),
                vis,
                *smear_kernel_args(prepared.smearing, uvw, exposure),
            )
        if prepared.nspec != ncorr:
            vis = stack_unpolarised_vis(vis[..., 0], ncorr)
    elif prepared.backend == "perchan":
        vis = _perchan_visibilities(prepared, uvw, epsilon, do_wgridding, nthreads)
    else:
        stokes_vis = np.stack(
            [
                _fft_stokes_visibilities(
                    plane,
                    prepared.spectrum,
                    prepared.grid,
                    uvw,
                    prepared.chan_freqs,
                    prepared.npix_l,
                    prepared.npix_m,
                    epsilon,
                    do_wgridding,
                    nthreads,
                )
                for plane in prepared.planes
            ]
        )
        corrs = stokes_to_correlations(
            _stokes_getter(stokes_vis, prepared.stokes_names),
            ncorr,
            prepared.polarisation,
            prepared.linear_basis,
        )
        vis = np.stack(corrs, axis=-1)

    if noise_vis:
        vis = add_noise(vis, noise_vis, seed=seed)
    if out_dtype is not None:
        vis = vis.astype(out_dtype, copy=False)
    return vis
