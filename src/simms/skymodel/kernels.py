"""Numba kernels for predicting visibilities from a discrete sky model.

The kernels accumulate directly into the output visibility buffer, so a
prediction over ``nsrc`` sources allocates no per-source temporaries.

On a uniformly spaced channel grid the phase ``2*pi*(u.l + v.m + w.n)*nu/c``
is linear in ``nu``, so the per-channel phasor is a fixed rotation of the
previous one. That replaces one ``sincos`` per channel with one complex
multiply. The rotation is renormalised every ``RENORM_INTERVAL`` channels to
stop the modulus drifting away from unity.

Every kernel optionally applies analytic time and bandwidth smearing, a real
factor ``sinc(x_nu) * sinc(x_t)`` on the phasor that reproduces the
correlator's averaging over a channel and over an integration; see
:mod:`simms.skymodel.smearing` for the maths and for how ``bw_half`` and
``smear_uvw`` are built. ``smear=False`` skips it, and the hot
point-source path keeps a separate loop for that case, so a run that does not
ask for smearing runs the kernel it always ran.
"""

import numpy as np
from numba import njit

from simms.constants import PI, C

TWO_PI = 2.0 * PI

# Channels between renormalisations of the recurrence phasor.
RENORM_INTERVAL = 256

_JIT = dict(cache=True, nogil=True, fastmath=True)

# Placeholder for the per-row smearing coefficients when ``smear`` is False. Numba
# still needs an array of the right type, but never reads it.
NO_SMEAR_UVW = np.zeros((1, 3))


@njit(inline="always", **_JIT)
def _sinc(x):
    """``sin(x)/x``: the mean of ``exp(1j*phi)`` over ``phi`` uniform on ``[-x, x]``."""
    if -1e-8 < x < 1e-8:
        return 1.0
    return np.sin(x) / x


@njit(inline="always", **_JIT)
def _smear_row(smear_uvw, r, smear):
    """Row ``r``'s ``0.5 * dt * d(u, v, w)/dt``, or zeros when smearing is off."""
    if smear:
        return smear_uvw[r, 0], smear_uvw[r, 1], smear_uvw[r, 2]
    return 0.0, 0.0, 0.0


@njit(inline="always", **_JIT)
def _smear_source(smear, bw_half, base, su, sv, sw, lmn, s):
    """Source ``s``'s bandwidth factor and half-fringe-rate (per unit frequency).

    The bandwidth average is over ``phi = base * nu`` at fixed ``base``, so its
    ``sinc`` does not depend on the channel and is taken once here. The time
    average is over ``nu * d(base)/dt``, which is ``base`` again with the
    baseline replaced by its time derivative; only the ``nu`` factor is left to
    the per-channel loop.
    """
    if not smear:
        return 1.0, 0.0
    rate_half = (su * lmn[s, 0] + sv * lmn[s, 1] + sw * lmn[s, 2]) * TWO_PI / C
    return _sinc(bw_half * base), rate_half


@njit(inline="always", **_JIT)
def _accumulate_point_uniform(vis_row, bmat_s, base, f0, df, nchan, nspec, amp, smear, bw_fac, rate_half):
    """Point source, uniform channel grid: one sincos, then a rotation per channel.

    Smearing gets its own loop rather than a branch inside this one: it carries a
    second recurrence, and keeping that state out of here leaves the unsmeared
    loop exactly as tight as it was before smearing existed.
    """
    if smear:
        _accumulate_point_uniform_smeared(vis_row, bmat_s, base, f0, df, nchan, nspec, amp, bw_fac, rate_half)
        return

    re = np.cos(base * f0)
    im = np.sin(base * f0)
    cos_d = np.cos(base * df)
    sin_d = np.sin(base * df)

    for f in range(nchan):
        phasor = amp * (re + 1j * im)
        for c in range(nspec):
            vis_row[f, c] += bmat_s[c, f] * phasor

        re, im = re * cos_d - im * sin_d, re * sin_d + im * cos_d
        if f % RENORM_INTERVAL == RENORM_INTERVAL - 1:
            inv = 1.0 / np.sqrt(re * re + im * im)
            re *= inv
            im *= inv


@njit(inline="always", **_JIT)
def _accumulate_point_uniform_smeared(vis_row, bmat_s, base, f0, df, nchan, nspec, amp, bw_fac, rate_half):
    """As :func:`_accumulate_point_uniform`, with the smearing factor applied.

    The bandwidth factor does not depend on the channel and is already folded into
    ``bw_fac``. The time factor is ``sin(x)/x`` at ``x = rate_half*nu``, and ``x`` is
    linear in ``nu`` on this grid, so its sine rides a second rotation rather than
    costing a ``sincos`` per channel -- the same trick, and the same renormalisation,
    as the phasor itself.
    """
    re = np.cos(base * f0)
    im = np.sin(base * f0)
    cos_d = np.cos(base * df)
    sin_d = np.sin(base * df)

    x = rate_half * f0
    dx = rate_half * df
    s_re = np.cos(x)
    s_im = np.sin(x)
    s_cos_d = np.cos(dx)
    s_sin_d = np.sin(dx)

    for f in range(nchan):
        phasor = (amp * bw_fac * (1.0 if -1e-8 < x < 1e-8 else s_im / x)) * (re + 1j * im)
        for c in range(nspec):
            vis_row[f, c] += bmat_s[c, f] * phasor

        re, im = re * cos_d - im * sin_d, re * sin_d + im * cos_d
        s_re, s_im = s_re * s_cos_d - s_im * s_sin_d, s_re * s_sin_d + s_im * s_cos_d
        x += dx
        if f % RENORM_INTERVAL == RENORM_INTERVAL - 1:
            inv = 1.0 / np.sqrt(re * re + im * im)
            re *= inv
            im *= inv
            inv = 1.0 / np.sqrt(s_re * s_re + s_im * s_im)
            s_re *= inv
            s_im *= inv


@njit(inline="always", **_JIT)
def _accumulate_point_general(vis_row, bmat_s, base, freqs, nchan, nspec, amp, smear, bw_fac, rate_half):
    """Point source, arbitrary channel grid."""
    for f in range(nchan):
        a = amp * bw_fac * _sinc(rate_half * freqs[f]) if smear else amp
        phase = base * freqs[f]
        phasor = a * (np.cos(phase) + 1j * np.sin(phase))
        for c in range(nspec):
            vis_row[f, c] += bmat_s[c, f] * phasor


@njit(inline="always", **_JIT)
def _accumulate_gaussian(vis_row, bmat_s, base, freqs, nchan, nspec, amp, gauss_arg, smear, bw_fac, rate_half):
    """Gaussian source. The envelope exp(-gauss_arg * (nu/c)**2) needs a real
    exponential per channel, which dominates, so no rotation trick here."""
    for f in range(nchan):
        a = amp * bw_fac * _sinc(rate_half * freqs[f]) if smear else amp
        scale = freqs[f] / C
        envelope = a * np.exp(-gauss_arg * scale * scale)
        phase = base * freqs[f]
        phasor = envelope * (np.cos(phase) + 1j * np.sin(phase))
        for c in range(nspec):
            vis_row[f, c] += bmat_s[c, f] * phasor


@njit(**_JIT)
def predict_vis(
    uvw, freqs, uniform, lmn, gauss_shape, is_gauss, bmat, lightcurve, time_index, vis, smear, bw_half, smear_uvw
):
    """
    Accumulate model visibilities into ``vis``.

    Parameters
    ----------
    uvw : (nrow, 3) float64
        Baseline coordinates in metres.
    freqs : (nchan,) float64
        Channel centre frequencies in Hz.
    uniform : bool
        True if ``freqs`` is uniformly spaced, enabling the rotation recurrence.
    lmn : (nsrc, 3) float64
        Direction cosines ``l``, ``m`` and ``n - 1`` per source.
    gauss_shape : (nsrc, 3) float64
        Per-source ``ell``, ``emm``, ``ecc`` describing the Gaussian envelope.
    is_gauss : (nsrc,) bool
        Whether each source has a non-zero extent.
    bmat : (nsrc, nspec, nchan) complex
        Brightness matrix per source and correlation, excluding any lightcurve.
    lightcurve : (nsrc, ntime) float64
        Time-dependent flux scaling. All ones (and ``ntime == 1``) when the model
        holds no transients.
    time_index : (nrow,) int64
        Index into the time axis of ``lightcurve`` for each row.
    vis : (nrow, nchan, nspec) complex
        Output buffer, accumulated into.
    smear : bool
        Apply analytic time and bandwidth smearing. When False the remaining two
        arguments are ignored.
    bw_half : float
        Half the channel width in Hz (:attr:`simms.skymodel.smearing.Smearing.bw_half`).
    smear_uvw : (nrow, 3) float64
        Per-row ``0.5 * dt * d(u, v, w)/dt`` from
        :meth:`simms.skymodel.smearing.Smearing.row_uvw`.
    """
    nrow = uvw.shape[0]
    nchan = freqs.shape[0]
    nsrc = lmn.shape[0]
    nspec = vis.shape[2]

    f0 = freqs[0]
    df = freqs[1] - freqs[0] if nchan > 1 else 0.0

    for r in range(nrow):
        u = uvw[r, 0]
        v = uvw[r, 1]
        w = uvw[r, 2]
        vis_row = vis[r]
        tidx = time_index[r]
        su, sv, sw = _smear_row(smear_uvw, r, smear)

        for s in range(nsrc):
            amp = lightcurve[s, tidx]
            base = (u * lmn[s, 0] + v * lmn[s, 1] + w * lmn[s, 2]) * TWO_PI / C
            bw_fac, rate_half = _smear_source(smear, bw_half, base, su, sv, sw, lmn, s)

            if is_gauss[s]:
                ell = gauss_shape[s, 0]
                emm = gauss_shape[s, 1]
                ecc = gauss_shape[s, 2]
                fu1 = (u * emm - v * ell) * ecc
                fv1 = u * ell + v * emm
                _accumulate_gaussian(
                    vis_row, bmat[s], base, freqs, nchan, nspec, amp, fu1 * fu1 + fv1 * fv1, smear, bw_fac, rate_half
                )
            elif uniform:
                _accumulate_point_uniform(vis_row, bmat[s], base, f0, df, nchan, nspec, amp, smear, bw_fac, rate_half)
            else:
                _accumulate_point_general(vis_row, bmat[s], base, freqs, nchan, nspec, amp, smear, bw_fac, rate_half)

    return vis


@njit(**_JIT)
def predict_vis_beam(
    uvw,
    freqs,
    uniform,
    lmn,
    gauss_shape,
    is_gauss,
    bmat,
    lightcurve,
    time_index,
    vis,
    antenna1,
    antenna2,
    ant_type,
    beam_grid,
    pa_lo,
    pa_wt,
    corr_feed_p,
    corr_feed_q,
    smear,
    bw_half,
    smear_uvw,
):
    """Accumulate model visibilities with a per-antenna primary beam applied.

    Like :func:`predict_vis`, but each correlation ``c`` of source ``s`` is scaled by
    ``g_p[fp(c)] * conj(g_q[fq(c)])``, where ``g_p``/``g_q`` are the interpolated feed
    voltages of the two antennas on the baseline.

    Extra parameters
    ----------------
    antenna1, antenna2 : (nrow,) int
        Antenna indices per row.
    ant_type : (nant,) int
        Beam-type index per antenna, indexing the first axis of ``beam_grid``.
    beam_grid : (ntype, n_pa, nsrc, nchan, 2) complex
        Feed voltages sampled on the parallactic-angle grid (last axis 0=H, 1=V).
    pa_lo : (nrow,) int
        Lower PA-grid index bracketing each row's timestamp.
    pa_wt : (nrow,) float
        Interpolation weight in ``[0, 1]`` toward ``pa_lo + 1``.
    corr_feed_p, corr_feed_q : (ncorr,) int
        Feed index (0=H, 1=V) of the first/second antenna for each correlation.
    smear : bool
        Apply analytic time and bandwidth smearing. When False the remaining two
        arguments are ignored.
    bw_half : float
        Half the channel width in Hz (:attr:`simms.skymodel.smearing.Smearing.bw_half`).
    smear_uvw : (nrow, 3) float64
        Per-row ``0.5 * dt * d(u, v, w)/dt`` from
        :meth:`simms.skymodel.smearing.Smearing.row_uvw`.
    """
    nrow = uvw.shape[0]
    nchan = freqs.shape[0]
    nsrc = lmn.shape[0]
    ncorr = vis.shape[2]

    f0 = freqs[0]
    df = freqs[1] - freqs[0] if nchan > 1 else 0.0

    for r in range(nrow):
        u = uvw[r, 0]
        v = uvw[r, 1]
        w = uvw[r, 2]
        vis_row = vis[r]
        tidx = time_index[r]
        tp = ant_type[antenna1[r]]
        tq = ant_type[antenna2[r]]
        k = pa_lo[r]
        wt = pa_wt[r]
        su, sv, sw = _smear_row(smear_uvw, r, smear)

        for s in range(nsrc):
            amp = lightcurve[s, tidx]
            base = (u * lmn[s, 0] + v * lmn[s, 1] + w * lmn[s, 2]) * TWO_PI / C
            bw_fac, rate_half = _smear_source(smear, bw_half, base, su, sv, sw, lmn, s)

            gaussian = is_gauss[s]
            if gaussian:
                ell = gauss_shape[s, 0]
                emm = gauss_shape[s, 1]
                ecc = gauss_shape[s, 2]
                fu1 = (u * emm - v * ell) * ecc
                fv1 = u * ell + v * emm
                gauss_arg = fu1 * fu1 + fv1 * fv1

            re = np.cos(base * f0)
            im = np.sin(base * f0)
            cos_d = np.cos(base * df)
            sin_d = np.sin(base * df)

            for f in range(nchan):
                a = amp * bw_fac * _sinc(rate_half * freqs[f]) if smear else amp
                if gaussian:
                    scale = freqs[f] / C
                    envelope = a * np.exp(-gauss_arg * scale * scale)
                    phase = base * freqs[f]
                    phasor = envelope * (np.cos(phase) + 1j * np.sin(phase))
                elif uniform:
                    phasor = a * (re + 1j * im)
                else:
                    phase = base * freqs[f]
                    phasor = a * (np.cos(phase) + 1j * np.sin(phase))

                # Linearly interpolate the two feed voltages of each antenna in PA.
                gp0 = beam_grid[tp, k, s, f, 0] * (1.0 - wt) + beam_grid[tp, k + 1, s, f, 0] * wt
                gp1 = beam_grid[tp, k, s, f, 1] * (1.0 - wt) + beam_grid[tp, k + 1, s, f, 1] * wt
                gq0 = beam_grid[tq, k, s, f, 0] * (1.0 - wt) + beam_grid[tq, k + 1, s, f, 0] * wt
                gq1 = beam_grid[tq, k, s, f, 1] * (1.0 - wt) + beam_grid[tq, k + 1, s, f, 1] * wt

                for c in range(ncorr):
                    gpc = gp0 if corr_feed_p[c] == 0 else gp1
                    gqc = gq0 if corr_feed_q[c] == 0 else gq1
                    vis_row[f, c] += bmat[s, c, f] * phasor * gpc * np.conj(gqc)

                if uniform and not gaussian:
                    re, im = re * cos_d - im * sin_d, re * sin_d + im * cos_d
                    if f % RENORM_INTERVAL == RENORM_INTERVAL - 1:
                        inv = 1.0 / np.sqrt(re * re + im * im)
                        re *= inv
                        im *= inv

    return vis


@njit(**_JIT)
def predict_vis_jones(
    uvw,
    freqs,
    uniform,
    lmn,
    gauss_shape,
    is_gauss,
    bmat,
    lightcurve,
    time_index,
    vis,
    antenna1,
    antenna2,
    ant_type,
    beam_grid,
    pa_lo,
    pa_wt,
    smear,
    bw_half,
    smear_uvw,
):
    """Accumulate visibilities with a full 2x2 Jones primary beam.

    Like :func:`predict_vis_beam`, but applies the complete E-Jones per source:
    ``V_pq = (E_p · B · E_q^H) · phasor`` with 2x2 complex matrices, where ``B`` is the
    source coherency ``[[XX, XY], [YX, YY]]`` from ``bmat[s, :, f]`` (linear feed basis)
    and ``E_p``/``E_q`` are the PA-interpolated beam Jones. The basis transform to the MS
    correlation frame is already folded into ``beam_grid`` (see ``build_beam_grid_jones``).
    Requires ``ncorr == 4``.

    beam_grid : (ntype, n_pa, nsrc, nchan, 2, 2) complex

    smear : bool
        Apply analytic time and bandwidth smearing. When False the remaining two
        arguments are ignored.
    bw_half : float
        Half the channel width in Hz (:attr:`simms.skymodel.smearing.Smearing.bw_half`).
    smear_uvw : (nrow, 3) float64
        Per-row ``0.5 * dt * d(u, v, w)/dt`` from
        :meth:`simms.skymodel.smearing.Smearing.row_uvw`.
    """
    nrow = uvw.shape[0]
    nchan = freqs.shape[0]
    nsrc = lmn.shape[0]

    f0 = freqs[0]
    df = freqs[1] - freqs[0] if nchan > 1 else 0.0

    for r in range(nrow):
        u = uvw[r, 0]
        v = uvw[r, 1]
        w = uvw[r, 2]
        vis_row = vis[r]
        tidx = time_index[r]
        tp = ant_type[antenna1[r]]
        tq = ant_type[antenna2[r]]
        k = pa_lo[r]
        wt = pa_wt[r]
        wt0 = 1.0 - wt
        su, sv, sw = _smear_row(smear_uvw, r, smear)

        for s in range(nsrc):
            amp = lightcurve[s, tidx]
            base = (u * lmn[s, 0] + v * lmn[s, 1] + w * lmn[s, 2]) * TWO_PI / C
            bw_fac, rate_half = _smear_source(smear, bw_half, base, su, sv, sw, lmn, s)

            gaussian = is_gauss[s]
            if gaussian:
                ell = gauss_shape[s, 0]
                emm = gauss_shape[s, 1]
                ecc = gauss_shape[s, 2]
                fu1 = (u * emm - v * ell) * ecc
                fv1 = u * ell + v * emm
                gauss_arg = fu1 * fu1 + fv1 * fv1

            re = np.cos(base * f0)
            im = np.sin(base * f0)
            cos_d = np.cos(base * df)
            sin_d = np.sin(base * df)

            for f in range(nchan):
                a = amp * bw_fac * _sinc(rate_half * freqs[f]) if smear else amp
                if gaussian:
                    scale = freqs[f] / C
                    envelope = a * np.exp(-gauss_arg * scale * scale)
                    phase = base * freqs[f]
                    phasor = envelope * (np.cos(phase) + 1j * np.sin(phase))
                elif uniform:
                    phasor = a * (re + 1j * im)
                else:
                    phase = base * freqs[f]
                    phasor = a * (np.cos(phase) + 1j * np.sin(phase))

                # PA-interpolated 2x2 Jones for each antenna.
                ep00 = beam_grid[tp, k, s, f, 0, 0] * wt0 + beam_grid[tp, k + 1, s, f, 0, 0] * wt
                ep01 = beam_grid[tp, k, s, f, 0, 1] * wt0 + beam_grid[tp, k + 1, s, f, 0, 1] * wt
                ep10 = beam_grid[tp, k, s, f, 1, 0] * wt0 + beam_grid[tp, k + 1, s, f, 1, 0] * wt
                ep11 = beam_grid[tp, k, s, f, 1, 1] * wt0 + beam_grid[tp, k + 1, s, f, 1, 1] * wt
                eq00 = beam_grid[tq, k, s, f, 0, 0] * wt0 + beam_grid[tq, k + 1, s, f, 0, 0] * wt
                eq01 = beam_grid[tq, k, s, f, 0, 1] * wt0 + beam_grid[tq, k + 1, s, f, 0, 1] * wt
                eq10 = beam_grid[tq, k, s, f, 1, 0] * wt0 + beam_grid[tq, k + 1, s, f, 1, 0] * wt
                eq11 = beam_grid[tq, k, s, f, 1, 1] * wt0 + beam_grid[tq, k + 1, s, f, 1, 1] * wt

                # B (coherency) and M = E_p @ B.
                b00 = bmat[s, 0, f]
                b01 = bmat[s, 1, f]
                b10 = bmat[s, 2, f]
                b11 = bmat[s, 3, f]
                m00 = ep00 * b00 + ep01 * b10
                m01 = ep00 * b01 + ep01 * b11
                m10 = ep10 * b00 + ep11 * b10
                m11 = ep10 * b01 + ep11 * b11

                # V = (M @ E_q^H) * phasor;  E_q^H = conj(transpose(E_q)).
                cq00 = np.conj(eq00)
                cq01 = np.conj(eq01)
                cq10 = np.conj(eq10)
                cq11 = np.conj(eq11)
                vis_row[f, 0] += (m00 * cq00 + m01 * cq01) * phasor
                vis_row[f, 1] += (m00 * cq10 + m01 * cq11) * phasor
                vis_row[f, 2] += (m10 * cq00 + m11 * cq01) * phasor
                vis_row[f, 3] += (m10 * cq10 + m11 * cq11) * phasor

                if uniform and not gaussian:
                    re, im = re * cos_d - im * sin_d, re * sin_d + im * cos_d
                    if f % RENORM_INTERVAL == RENORM_INTERVAL - 1:
                        inv = 1.0 / np.sqrt(re * re + im * im)
                        re *= inv
                        im *= inv

    return vis


def is_uniform_grid(freqs: np.ndarray, rtol: float = 1e-9) -> bool:
    """True if ``freqs`` is uniformly spaced to within ``rtol`` of the channel width."""
    if freqs.size < 3:
        return True
    steps = np.diff(freqs)
    return bool(np.all(np.abs(steps - steps[0]) <= rtol * np.abs(steps[0])))
