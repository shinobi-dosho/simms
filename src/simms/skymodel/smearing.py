"""Analytic time and bandwidth smearing for the DFT prediction kernels.

A visibility in an averaged MS is not the monochromatic, instantaneous
coherence the RIME writes down: the correlator averages over a channel of
width ``dnu`` and over an integration of length ``dt``. Both averages reduce
the amplitude of a source away from the phase centre, by more the longer the
baseline -- the familiar bandwidth and time smearing. A model predicted
without them over-predicts exactly where the real data are decorrelated,
which is a modelling choice for a simulation from scratch but a bug when the
model is differenced against real data (``skysim --mode subtract``).

The residual phase of source ``s`` on baseline ``pq`` is

    phi = base * nu,    base = 2*pi*(u.l + v.m + w.(n - 1)) / c

and both averages are over a top-hat, so each contributes a real factor
``sinc(x) = sin(x)/x`` evaluated at half the phase swing:

**Bandwidth.** ``phi`` is linear in ``nu`` with ``base`` fixed, so averaging
over a channel of width ``dnu`` is exact:

    x_nu = 0.5 * dnu * base

It does not depend on ``nu``, so it is one ``sinc`` per row and source.

**Time.** ``phi`` varies over the integration because the baseline turns with
the Earth. For a tracking interferometer at declination ``dec0``,

    du/dt = w0 * (w.cos(dec0) - v.sin(dec0))
    dv/dt = w0 * u.sin(dec0)
    dw/dt = -w0 * u.cos(dec0)

with ``w0`` the sidereal rotation rate (:data:`~simms.constants.OMEGA_EARTH`).
The fringe rate is then ``nu * d(base)/dt``, which is ``base`` again with
``(u, v, w)`` replaced by their derivatives, so

    x_t = 0.5 * dt * nu * d(base)/dt

This one *is* proportional to ``nu`` and so is evaluated per channel. The
combined factor ``sinc(x_nu) * sinc(x_t)`` multiplies the phasor; it is 1 at
the phase centre (``base`` and its derivative both vanish) and falls off with
baseline length and offset.

Both are the standard first-order treatment (Bridle & Schwab, in *Synthesis
Imaging in Radio Astronomy II*): the average is taken about the channel and
interval centres, so the phase is unshifted and only the amplitude changes.
That holds while the fringe rate is constant across one integration, which it
is for any sane ``dt``.

**Sub-sampling** (:class:`SubsampleSmearing`) is the second route, for the
gridder backends, which transform whole images and cannot carry a
per-visibility factor: predict the whole backend at several sub-times and
sub-frequencies and average. Sub-times need the ``uvw`` a fraction of a dump
away from the row's dump-centre value, which is an *exact* rotation: with
``A(dec)`` the fixed declination tilt and ``R_z`` the plain rotation about the
polar axis,

    uvw(t + dt) = A(dec) . R_z(w0*dt) . A(dec)^T . uvw(t)

(differentiating at ``dt = 0`` reproduces the derivative triplet above).
Sub-frequencies shift each channel by fractions of its own (signed) width.
Midpoint offsets make one sub-sample per axis the unsmeared prediction, and the
per-axis counts are frozen once per run from MS-global worst-case phase swings
(:func:`worst_case_swings`/:func:`subsample_counts`), so the model cannot
depend on how the run was chunked. The average carries a one-sided midpoint
bias of up to ``1/sinc(PHASE_TOL/4) - 1`` (about 1%) -- a ~1% method, unlike
the analytic factor's ~1e-4 on the DFT path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np

from simms.constants import OMEGA_EARTH, PI, C

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Smearing:
    """The MS-derived quantities the kernels need to decorrelate a phasor.

    Built once per run from the spectral window and the phase centre; the
    per-row half-interval part is folded into :meth:`row_uvw` because
    ``EXPOSURE`` arrives block by block with the ``UVW``.

    Attributes
    ----------
    bw_half : float
        Half the channel width (Hz). Channel-independent, so the bandwidth
        factor costs one ``sinc`` per row and source.
    sin_dec0, cos_dec0 : float
        Sine and cosine of the phase-centre declination, which set how the
        baseline turns (see the module docstring).
    """

    bw_half: float
    sin_dec0: float
    cos_dec0: float

    @classmethod
    def from_ms(cls, chan_width, dec0: float) -> Smearing:
        """Build from a ``SPECTRAL_WINDOW.CHAN_WIDTH`` row and the phase-centre dec (radians).

        ``chan_width`` may be a scalar or a per-channel array; widths are taken
        in absolute value, so a descending channel grid needs no special case.
        A window whose channels differ in width is reduced to their median, with
        a warning -- the factor is one number per row and source, and no real MS
        mixes widths inside one spectral window.
        """
        widths = np.abs(np.atleast_1d(np.asarray(chan_width, dtype=np.float64)))
        width = float(np.median(widths))
        if widths.size > 1 and not np.allclose(widths, widths[0], rtol=1e-6, atol=0.0):
            log.warning(
                "SPECTRAL_WINDOW.CHAN_WIDTH is not uniform (%.6g to %.6g Hz); bandwidth "
                "smearing uses the median width, %.6g Hz.",
                widths.min(),
                widths.max(),
                width,
            )
        return cls(bw_half=0.5 * width, sin_dec0=float(np.sin(dec0)), cos_dec0=float(np.cos(dec0)))

    def row_uvw(self, uvw: np.ndarray, exposure) -> np.ndarray:
        """``0.5 * dt * d(u, v, w)/dt`` per row, the kernels' time-smearing coefficients.

        Shaped like ``uvw`` so the kernel forms the fringe rate with the same
        dot product against ``(l, m, n - 1)`` that it uses for the phase.

        Parameters
        ----------
        uvw : numpy.ndarray
            ``(nrow, 3)`` baseline coordinates in metres.
        exposure : numpy.ndarray or float
            Per-row integration time (seconds), i.e. the MS ``EXPOSURE`` column.

        Returns
        -------
        numpy.ndarray
            ``(nrow, 3)`` float64, in metres (the ``0.5 * dt * omega`` factor is
            dimensionless).
        """
        uvw = np.asarray(uvw, dtype=np.float64)
        half = 0.5 * OMEGA_EARTH * np.broadcast_to(np.asarray(exposure, dtype=np.float64), uvw.shape[:1])
        u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]
        out = np.empty_like(uvw)
        out[:, 0] = half * (w * self.cos_dec0 - v * self.sin_dec0)
        out[:, 1] = half * self.sin_dec0 * u
        out[:, 2] = -half * self.cos_dec0 * u
        return out


PHASE_TOL = 0.5
"""Largest full phase swing (radians) allowed across one sub-interval.

The midpoint average of a linear phasor over-estimates the top-hat average by
``1/sinc(swing/4)`` per sub-interval half-swing; at 0.5 rad that bias is ~1%,
one-sided (it does not average out across sub-samples).
"""


def midpoint_fractions(n: int) -> np.ndarray:
    """Centres of ``n`` equal sub-intervals of ``[-1/2, 1/2]``.

    ``n == 1`` gives ``[0.0]``: the single sub-sample *is* the unsmeared
    evaluation, which is what makes auto-chosen counts free when smearing is
    negligible.
    """
    return (np.arange(int(n)) + 0.5) / int(n) - 0.5


def worst_case_swings(
    uvw_max: float, exposure_max: float, chan_width_max: float, nu_max: float, theta_max: float
) -> tuple[float, float]:
    """Full phase swings across one dump and one channel for the worst-placed source.

    ``theta_max`` is the largest ``||(l, m, n - 1)||`` the model can reach (for an
    image, its corners); ``base = 2*pi*|uvw|*theta/c`` then bounds the residual
    phase per unit frequency, ``|d(uvw)/dt| <= w0*|uvw|`` bounds the fringe rate,
    and the time swing is evaluated at the highest sub-frequency
    (``nu_max + chan_width_max/2``). Plain floats in, plain floats out -- the
    caller derives the bounds (this module knows nothing about images or MSs).

    Returns
    -------
    (swing_t, swing_nu) : tuple of float
        Full swings in radians over one integration and one channel.
    """
    base_max = 2.0 * PI * abs(uvw_max) * abs(theta_max) / C
    swing_nu = abs(chan_width_max) * base_max
    swing_t = abs(exposure_max) * OMEGA_EARTH * (abs(nu_max) + 0.5 * abs(chan_width_max)) * base_max
    return swing_t, swing_nu


def subsample_counts(swing_t: float, swing_nu: float, cap: int) -> tuple[int, int]:
    """Smallest per-axis counts keeping each sub-interval's swing below ``PHASE_TOL``.

    Clipped to ``[1, cap]``; the caller warns when the cap truncates, since a
    capped run over-predicts the residual amplitude systematically.

    Returns
    -------
    (n_time, n_freq) : tuple of int
    """
    cap = max(int(cap), 1)

    def count(swing: float) -> int:
        return int(np.clip(np.ceil(swing / PHASE_TOL), 1, cap))

    return count(swing_t), count(swing_nu)


@dataclass(frozen=True)
class SubsampleSmearing:
    """Sub-sampled time/bandwidth smearing for the gridder prediction backends.

    Carries what :func:`simms.skymodel.fits_skies.predict_fits_block` needs to
    average whole-backend predictions over the dump and the channel: the signed
    per-channel widths (sliced alongside the channel grid by
    :meth:`select_channels`), the phase-centre declination that sets how a
    baseline turns, and the per-axis counts -- frozen at build time from
    MS-global bounds so the result cannot depend on row or channel chunking.

    Attributes
    ----------
    chan_width : numpy.ndarray
        ``(nchan,)`` signed channel widths (Hz); a descending grid's negative
        widths shift the sub-frequencies the right way with no special case.
    sin_dec0, cos_dec0 : float
        Sine and cosine of the phase-centre declination.
    n_time, n_freq : int
        Sub-samples per dump and per channel (1 means no sub-sampling on that
        axis; (1, 1) is bypassed entirely by the caller).
    """

    chan_width: np.ndarray
    sin_dec0: float
    cos_dec0: float
    n_time: int
    n_freq: int

    @classmethod
    def from_ms(cls, chan_width, dec0: float, n_time: int, n_freq: int) -> SubsampleSmearing:
        """Build from a ``SPECTRAL_WINDOW.CHAN_WIDTH`` row, the phase-centre dec (radians), and frozen counts."""
        widths = np.atleast_1d(np.asarray(chan_width, dtype=np.float64))
        return cls(
            chan_width=widths,
            sin_dec0=float(np.sin(dec0)),
            cos_dec0=float(np.cos(dec0)),
            n_time=int(n_time),
            n_freq=int(n_freq),
        )

    def select_channels(self, chan_ids: np.ndarray) -> SubsampleSmearing:
        """Restrict to a subset of channels, alongside the prepared sky's own slicing."""
        return replace(self, chan_width=self.chan_width[chan_ids])

    def rotate_uvw(self, uvw: np.ndarray, dt) -> np.ndarray:
        """``uvw`` as it will stand ``dt`` seconds later, exactly.

        Applies ``A(dec) . R_z(w0*dt) . A(dec)^T`` per row (see the module
        docstring); ``dt`` may be a scalar or per-row array, since ``EXPOSURE``
        can vary by row. ``dt`` of all zeros returns the input array unchanged --
        the rotation's ``sin^2 + cos^2`` terms are only ulp-exact, and callers
        rely on the zero-offset sub-sample being *identical* to no smearing.
        """
        uvw = np.asarray(uvw, dtype=np.float64)
        phi = OMEGA_EARTH * np.broadcast_to(np.asarray(dt, dtype=np.float64), uvw.shape[:1])
        if not np.any(phi):
            return uvw
        ch, sh = np.cos(phi), np.sin(phi)
        s, c = self.sin_dec0, self.cos_dec0
        u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]
        out = np.empty_like(uvw)
        out[:, 0] = ch * u - s * sh * v + c * sh * w
        out[:, 1] = s * sh * u + (s * s * ch + c * c) * v + s * c * (1.0 - ch) * w
        out[:, 2] = -c * sh * u + s * c * (1.0 - ch) * v + (c * c * ch + s * s) * w
        return out
