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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from simms.constants import OMEGA_EARTH

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
