"""Tests for YAML-driven RIME corruptions in skysim."""

from __future__ import annotations

import logging
import tracemalloc

import dask.array as da
import numpy as np
import pytest
from daskms import xds_from_ms, xds_from_table

from simms.apps import skysim
from simms.skymodel.corruptions import (
    CorruptionSpec,
    TermSpec,
    apply_corruptions,
    load_corruption_spec,
    validate_spec,
)
from simms.telescope.generate_ms import create_ms

from . import InitTest, skysim_opts


class _GainTest(InitTest):
    def __init__(self, ncorr=2):
        self.test_files = []
        self.ncorr = ncorr
        correlations = ["XX", "YY"] if ncorr == 2 else ["XX", "XY", "YX", "YY"]
        self.ms = self.random_named_directory(suffix=".ms")
        create_ms(
            self.ms,
            telescope_name="meerkat",
            pointing_direction=["J2000", "0deg", "-30deg"],
            dtime=60,
            ntimes=10,
            start_freq="1420MHz",
            dfreq="4MHz",
            nchan=4,
            correlations=correlations,
            row_chunks=100000,
            sefd=None,
            column="DATA",
            start_time="2025-03-06T20:00:00",
            smooth=None,
            fit_order=None,
            subarray_range=[0, 5],
        )
        # Source at the phase centre so the uncorrupted visibility is row-independent.
        self.sky = self.random_named_file(suffix=".txt")
        with open(self.sky, "w") as fh:
            fh.write("#format: name ra dec stokes_i\nS 0h0m0s -30d0m0s 2.0\n")

    def write_yaml(self, content: str) -> str:
        path = self.random_named_file(suffix=".yaml")
        with open(path, "w") as fh:
            fh.write(content)
        return path


@pytest.fixture
def gt2():
    return _GainTest(ncorr=2)


@pytest.fixture
def gt4():
    return _GainTest(ncorr=4)


def read_column(ms, name):
    return getattr(xds_from_ms(ms)[0], name).data.compute()


def read_aux(ms):
    ds = xds_from_ms(ms)[0]
    return (
        ds.TIME.data.compute(),
        ds.ANTENNA1.data.compute(),
        ds.ANTENNA2.data.compute(),
    )


def expected_diagonal_factor(time, ant1, ant2, period, amplitude, random_seed, label="G"):
    """Reference implementation for a single diagonal time-varying term."""
    from simms.skymodel.corruptions import _stable_label_hash

    nant = max(int(ant1.max()), int(ant2.max())) + 1
    seed = (random_seed or 0) + _stable_label_hash(label)
    rng = np.random.default_rng(seed)
    phases = rng.random(nant)
    t0 = float(time.min())
    phase = 2.0 * np.pi * ((time - t0) / period + phases[ant1])
    gp = 1.0 + amplitude * (np.cos(phase) + 1j * np.sin(phase))
    phase = 2.0 * np.pi * ((time - t0) / period + phases[ant2])
    gq = 1.0 + amplitude * (np.cos(phase) + 1j * np.sin(phase))
    return gp * np.conj(gq)


def test_single_diagonal_time_term_matches_reference(gt2):
    period = 120.0
    amplitude = 0.1
    seed = 7
    yaml_path = gt2.write_yaml(
        f"""
gains:
  terms: [G]
  spec:
    - label: G
      diagonal: true
      complex: true
      axes: [time]
      period:
        time: {period}
      amplitude: {amplitude}
"""
    )

    skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA"))
    clean = read_column(gt2.ms, "DATA")

    skysim.runit(
        skysim_opts(
            gt2.ms,
            ascii_sky=gt2.sky,
            column="DATA",
            corruptions=yaml_path,
            seed_gains=seed,
        )
    )
    corrupted = read_column(gt2.ms, "DATA")

    time, ant1, ant2 = read_aux(gt2.ms)
    expected = expected_diagonal_factor(time, ant1, ant2, period, amplitude, seed).astype(corrupted.dtype)
    np.testing.assert_allclose(corrupted[:, 0, 0] / clean[:, 0, 0], expected, rtol=1e-6, atol=1e-7)


def test_zero_amplitude_term_contributes_nothing(gt2):
    """A zero-amplitude term is the identity, so adding one changes nothing.

    Expressed against a real term rather than on its own: a spec where every
    term is zero corrupts nothing and is now rejected as a mistake.
    """
    with_zero = gt2.write_yaml(
        """
gains:
  terms: [Z, G]
  spec:
    - label: Z
      type: scalar
      complex: true
      axes: [time]
      period:
        time: 300.0
      amplitude: 0.0
    - label: G
      type: scalar
      complex: true
      axes: [time]
      period:
        time: 120.0
      amplitude: 0.1
"""
    )
    without_zero = gt2.write_yaml(
        """
gains:
  terms: [G]
  spec:
    - label: G
      type: scalar
      complex: true
      axes: [time]
      period:
        time: 120.0
      amplitude: 0.1
"""
    )
    skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA", corruptions=with_zero, seed_gains=3))
    with_zero_data = read_column(gt2.ms, "DATA")

    skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA", corruptions=without_zero, seed_gains=3))
    without_zero_data = read_column(gt2.ms, "DATA")

    np.testing.assert_array_equal(with_zero_data, without_zero_data)


def test_two_diagonal_terms_are_product(gt2):
    yaml_path = gt2.write_yaml(
        """
gains:
  terms: [G1, G2]
  spec:
    - label: G1
      diagonal: true
      complex: true
      axes: [time]
      period:
        time: 120.0
      amplitude: 0.1
    - label: G2
      diagonal: true
      complex: true
      axes: [time]
      period:
        time: 300.0
      amplitude: 0.05
"""
    )
    seed = 5
    skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA"))
    clean = read_column(gt2.ms, "DATA")

    skysim.runit(
        skysim_opts(
            gt2.ms,
            ascii_sky=gt2.sky,
            column="DATA",
            corruptions=yaml_path,
            seed_gains=seed,
        )
    )
    corrupted = read_column(gt2.ms, "DATA")

    time, ant1, ant2 = read_aux(gt2.ms)
    f1 = expected_diagonal_factor(time, ant1, ant2, 120.0, 0.1, seed, label="G1")
    f2 = expected_diagonal_factor(time, ant1, ant2, 300.0, 0.05, seed, label="G2")
    expected = (f1 * f2).astype(corrupted.dtype)
    np.testing.assert_allclose(corrupted[:, 0, 0] / clean[:, 0, 0], expected, rtol=1e-6, atol=1e-7)


def test_frequency_term_varies_with_channel(gt2):
    yaml_path = gt2.write_yaml(
        """
gains:
  terms: [B]
  spec:
    - label: B
      diagonal: true
      complex: true
      axes: [frequency]
      period:
        frequency: "8MHz"
      amplitude: 0.05
"""
    )
    seed = 11
    skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA"))
    clean = read_column(gt2.ms, "DATA")

    skysim.runit(
        skysim_opts(
            gt2.ms,
            ascii_sky=gt2.sky,
            column="DATA",
            corruptions=yaml_path,
            seed_gains=seed,
        )
    )
    corrupted = read_column(gt2.ms, "DATA")

    spw = xds_from_table(f"{gt2.ms}::SPECTRAL_WINDOW")[0]
    freqs = spw.CHAN_FREQ.data[0].compute()
    _time, ant1, ant2 = read_aux(gt2.ms)
    nant = max(int(ant1.max()), int(ant2.max())) + 1
    from simms.skymodel.corruptions import _stable_label_hash

    rng = np.random.default_rng(seed + _stable_label_hash("B"))
    phases = rng.random(nant)
    freq0 = float(freqs[0])
    period = 8e6
    phase_p = 2.0 * np.pi * ((freqs[None, :] - freq0) / period + phases[ant1][:, None])
    phase_q = 2.0 * np.pi * ((freqs[None, :] - freq0) / period + phases[ant2][:, None])
    gp = 1.0 + 0.05 * (np.cos(phase_p) + 1j * np.sin(phase_p))
    gq = 1.0 + 0.05 * (np.cos(phase_q) + 1j * np.sin(phase_q))
    expected = (gp * np.conj(gq)).astype(corrupted.dtype)
    np.testing.assert_allclose(corrupted[:, :, 0] / clean[:, :, 0], expected, rtol=1e-6, atol=1e-7)


def test_time_frequency_term(gt2):
    yaml_path = gt2.write_yaml(
        """
gains:
  terms: [J]
  spec:
    - label: J
      diagonal: true
      complex: true
      axes: [time, frequency]
      period:
        time: "2min"
        frequency: "8MHz"
      amplitude: 0.03
"""
    )
    seed = 13
    skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA"))
    clean = read_column(gt2.ms, "DATA")

    skysim.runit(
        skysim_opts(
            gt2.ms,
            ascii_sky=gt2.sky,
            column="DATA",
            corruptions=yaml_path,
            seed_gains=seed,
        )
    )
    corrupted = read_column(gt2.ms, "DATA")

    spw = xds_from_table(f"{gt2.ms}::SPECTRAL_WINDOW")[0]
    freqs = spw.CHAN_FREQ.data[0].compute()
    time, ant1, ant2 = read_aux(gt2.ms)
    nant = max(int(ant1.max()), int(ant2.max())) + 1
    from simms.skymodel.corruptions import _stable_label_hash

    rng = np.random.default_rng(seed + _stable_label_hash("J"))
    phases_t = rng.random(nant)
    phases_f = rng.random(nant)
    t0 = float(time.min())
    freq0 = float(freqs[0])

    phase_pt = 2.0 * np.pi * ((time - t0) / 120.0 + phases_t[ant1])
    phase_pf = 2.0 * np.pi * ((freqs[None, :] - freq0) / 8e6 + phases_f[ant1][:, None])
    phase_qt = 2.0 * np.pi * ((time - t0) / 120.0 + phases_t[ant2])
    phase_qf = 2.0 * np.pi * ((freqs[None, :] - freq0) / 8e6 + phases_f[ant2][:, None])

    gp = 1.0 + 0.03 * np.exp(1j * (phase_pt[:, None] + phase_pf))
    gq = 1.0 + 0.03 * np.exp(1j * (phase_qt[:, None] + phase_qf))
    expected = (gp * np.conj(gq)).astype(corrupted.dtype)
    np.testing.assert_allclose(corrupted[:, :, 0] / clean[:, :, 0], expected, rtol=1e-6, atol=1e-7)


def test_full_term_requires_four_correlations(gt2):
    yaml_path = gt2.write_yaml(
        """
gains:
  terms: [J]
  spec:
    - label: J
      diagonal: false
      complex: true
      axes: [time]
      period:
        time: 120.0
      amplitude: 0.1
"""
    )
    with pytest.raises(RuntimeError, match="needs 4 correlations"):
        skysim.runit(
            skysim_opts(
                gt2.ms,
                ascii_sky=gt2.sky,
                column="DATA",
                corruptions=yaml_path,
                seed_gains=1,
            )
        )


def test_full_jones_rime_on_four_correlations(gt4):
    yaml_path = gt4.write_yaml(
        """
gains:
  terms: [J]
  spec:
    - label: J
      diagonal: false
      complex: true
      axes: [time]
      period:
        time: 120.0
      amplitude: 0.1
"""
    )
    # Just ensure it runs and produces finite, changed visibilities.
    skysim.runit(skysim_opts(gt4.ms, ascii_sky=gt4.sky, column="DATA"))
    clean = read_column(gt4.ms, "DATA")

    skysim.runit(
        skysim_opts(
            gt4.ms,
            ascii_sky=gt4.sky,
            column="DATA",
            corruptions=yaml_path,
            seed_gains=2,
        )
    )
    corrupted = read_column(gt4.ms, "DATA")
    assert np.all(np.isfinite(corrupted))
    assert not np.allclose(corrupted, clean)


def test_seed_gains_makes_corruptions_reproducible(gt2):
    yaml_path = gt2.write_yaml(
        """
gains:
  terms: [G]
  spec:
    - label: G
      diagonal: true
      complex: true
      axes: [time]
      period:
        time: 120.0
      amplitude: 0.1
"""
    )
    skysim.runit(
        skysim_opts(
            gt2.ms,
            ascii_sky=gt2.sky,
            column="DATA",
            corruptions=yaml_path,
            seed_gains=42,
        )
    )
    first = read_column(gt2.ms, "DATA")
    skysim.runit(
        skysim_opts(
            gt2.ms,
            ascii_sky=gt2.sky,
            column="DATA",
            corruptions=yaml_path,
            seed_gains=42,
        )
    )
    second = read_column(gt2.ms, "DATA")
    np.testing.assert_array_equal(first, second)


def test_different_seed_gains_gives_different_corruption(gt2):
    yaml_path = gt2.write_yaml(
        """
gains:
  terms: [G]
  spec:
    - label: G
      diagonal: true
      complex: true
      axes: [time]
      period:
        time: 120.0
      amplitude: 0.1
"""
    )
    skysim.runit(
        skysim_opts(
            gt2.ms,
            ascii_sky=gt2.sky,
            column="DATA",
            corruptions=yaml_path,
            seed_gains=1,
        )
    )
    first = read_column(gt2.ms, "DATA")
    skysim.runit(
        skysim_opts(
            gt2.ms,
            ascii_sky=gt2.sky,
            column="DATA",
            corruptions=yaml_path,
            seed_gains=2,
        )
    )
    second = read_column(gt2.ms, "DATA")
    assert not np.allclose(first, second)


def test_deprecated_seed_alias_matches_seed_noise(gt2):
    """--seed maps onto --seed-noise, reproducing the pre-rename realisation."""
    sefd = 500.0
    skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA", sefd=sefd, seed=99))
    legacy = read_column(gt2.ms, "DATA")

    skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA", sefd=sefd, seed_noise=99))
    renamed = read_column(gt2.ms, "DATA")
    np.testing.assert_array_equal(legacy, renamed)


@pytest.mark.parametrize(
    "yaml_content,match",
    [
        (
            """
gains:
  terms: [X]
  spec:
    - label: G
      diagonal: true
      axes: [time]
      period:
        time: 120.0
      amplitude: 0.1
""",
            "not described",
        ),
        (
            """
gains:
  terms: [G, G]
  spec:
    - label: G
      diagonal: true
      axes: [time]
      period:
        time: 120.0
      amplitude: 0.1
""",
            "duplicate",
        ),
        (
            """
gains:
  terms: [G]
  spec:
    - label: G
      diagonal: true
      axes: [distance]
      period:
        distance: 120.0
      amplitude: 0.1
""",
            "unknown axis",
        ),
        (
            """
gains:
  terms: [G]
  spec:
    - label: G
      diagonal: true
      axes: [time]
      period:
        time: -10.0
      amplitude: 0.1
""",
            "positive",
        ),
        (
            """
gains:
  terms: [G]
  spec:
    - label: G
      diagonal: true
      axes: [time]
      period:
        time: 120.0
      amplitude: -0.1
""",
            "non-negative",
        ),
    ],
    ids=["missing-label", "duplicate-label", "unknown-axis", "negative-period", "negative-amplitude"],
)
def test_invalid_specs_raise(gt2, yaml_content, match):
    yaml_path = gt2.write_yaml(yaml_content)
    with pytest.raises(RuntimeError, match=match):
        skysim.runit(
            skysim_opts(
                gt2.ms,
                ascii_sky=gt2.sky,
                column="DATA",
                corruptions=yaml_path,
                seed_gains=1,
            )
        )


def test_scalar_period_shorthand_for_single_axis(gt2):
    yaml_path = gt2.write_yaml(
        """
gains:
  terms: [G]
  spec:
    - label: G
      diagonal: true
      complex: true
      axes: [time]
      period: 120.0
      amplitude: 0.1
"""
    )
    # Should run without the mapping-form requirement.
    skysim.runit(
        skysim_opts(
            gt2.ms,
            ascii_sky=gt2.sky,
            column="DATA",
            corruptions=yaml_path,
            seed_gains=1,
        )
    )
    assert np.all(np.isfinite(read_column(gt2.ms, "DATA")))


def test_apply_corruptions_public_api(gt2):
    """The high-level apply_corruptions helper can be called directly."""
    yaml_path = gt2.write_yaml(
        """
gains:
  terms: [G]
  spec:
    - label: G
      diagonal: true
      complex: true
      axes: [time]
      period:
        time: 120.0
      amplitude: 0.1
"""
    )
    skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA"))
    clean = read_column(gt2.ms, "DATA")

    ds = xds_from_ms(gt2.ms)[0]
    spec = load_corruption_spec(yaml_path)
    corrupted = apply_corruptions(
        ds.DATA.data,
        ds.TIME.data,
        ds.ANTENNA1.data,
        ds.ANTENNA2.data,
        xds_from_table(f"{gt2.ms}::SPECTRAL_WINDOW")[0].CHAN_FREQ.data[0].compute(),
        spec,
        random_seed=1,
    ).compute()

    time, ant1, ant2 = read_aux(gt2.ms)
    expected = expected_diagonal_factor(time, ant1, ant2, 120.0, 0.1, 1, label="G").astype(corrupted.dtype)
    np.testing.assert_allclose(corrupted[:, 0, 0] / clean[:, 0, 0], expected, rtol=1e-6, atol=1e-7)


def test_row_chunk_without_highest_antenna():
    """Regression: gains sized off a block's antenna range crash on any block
    that does not contain the highest-numbered antenna."""
    nrow, nchan, ncorr = 16, 2, 2
    # First 8 rows only use antennas 0-2; the next 8 use 0-7.
    ant1 = np.array([0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2, 3, 4, 5, 6, 7])
    ant2 = np.array([1, 2, 0, 1, 2, 0, 1, 2, 7, 6, 5, 4, 3, 2, 1, 0])
    vis = da.from_array(np.ones((nrow, nchan, ncorr), dtype=np.complex128), chunks=(8, nchan, ncorr))
    time = da.from_array(np.arange(nrow, dtype=float) * 60.0, chunks=8)
    a1 = da.from_array(ant1, chunks=8)
    a2 = da.from_array(ant2, chunks=8)
    freqs = np.linspace(1.420e9, 1.428e9, nchan)

    spec = CorruptionSpec(
        terms=["G"],
        spec=[TermSpec(label="G", axes=["time"], period=120.0, amplitude=0.1)],
    )
    result = apply_corruptions(vis, time, a1, a2, freqs, spec, random_seed=1).compute()
    assert np.all(np.isfinite(result))


def test_frequency_term_is_invariant_under_channel_chunking():
    """Regression: the frequency phase reference is the MS's first channel, not
    the block's first channel -- chunk boundaries must not restart the sinusoid."""
    nrow, nchan, ncorr = 4, 8, 2
    ant1 = np.zeros(nrow, dtype=int)
    ant2 = np.ones(nrow, dtype=int)
    freqs = np.linspace(1.420e9, 1.436e9, nchan)

    spec = CorruptionSpec(
        terms=["B"],
        spec=[TermSpec(label="B", axes=["frequency"], period=8e6, amplitude=0.5)],
    )

    def run(chan_chunk):
        vis = da.from_array(np.ones((nrow, nchan, ncorr), dtype=np.complex128), chunks=(nrow, chan_chunk, ncorr))
        time = da.from_array(np.arange(nrow, dtype=float) * 60.0, chunks=nrow)
        a1 = da.from_array(ant1, chunks=nrow)
        a2 = da.from_array(ant2, chunks=nrow)
        return apply_corruptions(vis, time, a1, a2, freqs, spec, random_seed=1).compute()

    np.testing.assert_allclose(run(nchan), run(4), rtol=1e-13, atol=1e-14)


_UNSET_TYPE_YAML = """
gains:
  terms: [J]
  spec:
    - label: J
      complex: true
      axes: [time]
      period:
        time: 120.0
      amplitude: 0.1
"""


def _explicit_diagonal_yaml(diagonal: str) -> str:
    return f"""
gains:
  terms: [J]
  spec:
    - label: J
      diagonal: {diagonal}
      complex: true
      axes: [time]
      period:
        time: 120.0
      amplitude: 0.1
"""


def test_unset_type_is_diagonal_on_four_correlations(gt4):
    """An omitted 'type' means diag(g_x, g_y), not a dense Jones.

    Leakage has to be asked for: the default must not mix the polarisations.
    """
    unset = gt4.write_yaml(_UNSET_TYPE_YAML)
    explicit = gt4.write_yaml(_UNSET_TYPE_YAML.replace("      complex:", "      type: diagonal\n      complex:"))

    skysim.runit(skysim_opts(gt4.ms, ascii_sky=gt4.sky, column="DATA", corruptions=unset, seed_gains=3))
    from_unset = read_column(gt4.ms, "DATA")

    skysim.runit(skysim_opts(gt4.ms, ascii_sky=gt4.sky, column="DATA", corruptions=explicit, seed_gains=3))
    from_explicit = read_column(gt4.ms, "DATA")

    np.testing.assert_array_equal(from_unset, from_explicit)

    # A dense Jones would put flux into the cross-hands of a Stokes-I sky; a
    # diagonal one leaves them zero.
    skysim.runit(skysim_opts(gt4.ms, ascii_sky=gt4.sky, column="DATA"))
    clean = read_column(gt4.ms, "DATA")
    assert np.allclose(clean[..., 1], 0.0) and np.allclose(clean[..., 2], 0.0)
    assert np.allclose(from_unset[..., 1], 0.0) and np.allclose(from_unset[..., 2], 0.0)

    # ...and the two parallel hands must differ, or it is really a scalar term.
    assert not np.allclose(from_unset[..., 0], from_unset[..., 3])


def test_unset_type_is_diagonal_on_two_correlations(gt2):
    """The default does not depend on the correlation count: 2 corrs get
    diag(g_x, g_y) too, one feed per parallel hand."""
    unset = gt2.write_yaml(_UNSET_TYPE_YAML)
    explicit = gt2.write_yaml(_UNSET_TYPE_YAML.replace("      complex:", "      type: diagonal\n      complex:"))

    skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA", corruptions=unset, seed_gains=3))
    from_unset = read_column(gt2.ms, "DATA")

    skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA", corruptions=explicit, seed_gains=3))
    from_explicit = read_column(gt2.ms, "DATA")

    np.testing.assert_array_equal(from_unset, from_explicit)
    assert not np.allclose(from_unset[..., 0], from_unset[..., 1])


def test_unset_type_falls_back_to_scalar_on_one_correlation():
    """A single-correlation MS has no second feed, so the default degrades."""
    from simms.skymodel.corruptions import resolve_type

    term = TermSpec(label="J", axes=["time"], period=120.0, amplitude=0.1)
    assert resolve_type(term, ncorr=1) == "scalar"
    assert resolve_type(term, ncorr=2) == "diagonal"
    assert resolve_type(term, ncorr=4) == "diagonal"


def test_noise_is_added_after_the_gain_chain(gt2):
    """A noisy RIME is J_p V J_q^H + n, so the noise itself is never gain-modulated.

    This exercises the no-model branch: with no sky model there is nothing to
    corrupt, so the visibilities must be the bare noise realisation whether or
    not --corruptions is given. Under the old ordering the gains multiplied the
    noise, so this failed by the gain amplitude.
    """
    sefd = 500.0
    yaml_path = gt2.write_yaml(_explicit_diagonal_yaml("true"))

    skysim.runit(skysim_opts(gt2.ms, column="DATA", sefd=sefd, seed_noise=99))
    noise_only = read_column(gt2.ms, "DATA")

    skysim.runit(
        skysim_opts(
            gt2.ms,
            column="DATA",
            sefd=sefd,
            seed_noise=99,
            seed_gains=5,
            corruptions=yaml_path,
        )
    )
    with_corruptions = read_column(gt2.ms, "DATA")

    assert not np.allclose(noise_only, 0.0)
    np.testing.assert_array_equal(noise_only, with_corruptions)


def test_corruptions_leave_the_noise_component_untouched(gt2):
    """Sky + noise: subtracting the corrupted-model run from the noisy run must
    return the same noise realisation as a noise-only run.

    Under the old ordering the noise was multiplied by the gains, so this
    residual would differ from the bare realisation by the gain amplitude (10%
    here). The tolerance is set by complex64 rounding on the ~2 Jy model that
    cancels in the subtraction, not by the noise.
    """
    sefd = 500.0
    yaml_path = gt2.write_yaml(_explicit_diagonal_yaml("true"))
    opts = dict(corruptions=yaml_path, seed_gains=5)

    skysim.runit(skysim_opts(gt2.ms, column="DATA", sefd=sefd, seed_noise=99))
    noise_only = read_column(gt2.ms, "DATA")

    skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA", **opts))
    corrupted_model = read_column(gt2.ms, "DATA")

    skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA", sefd=sefd, seed_noise=99, **opts))
    both = read_column(gt2.ms, "DATA")

    # atol binds: one float32 ulp at the fixture's 2 Jy model is ~1.2e-7, and the
    # old gain-modulated noise differed by ~|g-1|*|n| ~ 5e-3. Raising the fixture
    # source flux by an order of magnitude would eat this margin.
    np.testing.assert_allclose(both - corrupted_model, noise_only, rtol=1e-4, atol=1e-6)


def test_falsy_diagonal_is_rejected_up_front(gt2):
    """YAML `diagonal: 0` is a full-Jones request and must fail validation.

    It parses as int, not bool, so an identity check against False lets it reach
    the block function and raise from inside a dask task instead of here.
    """
    yaml_path = gt2.write_yaml(
        """
gains:
  terms: [J]
  spec:
    - label: J
      diagonal: 0
      complex: true
      axes: [time]
      period:
        time: 120.0
      amplitude: 0.1
"""
    )
    with pytest.raises(RuntimeError, match="needs 4 correlations"):
        skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA", corruptions=yaml_path, seed_gains=1))


def test_bad_spec_still_fails_on_a_noise_only_run(gt2):
    """A noise-only run has nothing to corrupt, but must not silently accept a
    broken --corruptions file and write a clean-looking noise column."""
    yaml_path = gt2.write_yaml(
        """
gains:
  terms: [G]
  spec:
    - label: NOT_G
      diagonal: true
      complex: true
      axes: [time]
      period:
        time: 120.0
      amplitude: 0.1
"""
    )
    with pytest.raises(RuntimeError, match="not described in the spec"):
        skysim.runit(skysim_opts(gt2.ms, column="DATA", sefd=500.0, seed_noise=1, corruptions=yaml_path))


def test_noise_only_run_warns_that_corruptions_are_ignored(gt2):
    """The user must be told --corruptions did nothing, not left to infer it.

    Captures on skysim's own logger rather than via caplog, and overrides
    log_level: the shared test opts set it to CRITICAL, and runit applies that
    to the skysim logger, so a WARNING is filtered out before any handler.
    """
    yaml_path = gt2.write_yaml(_explicit_diagonal_yaml("true"))

    messages = []

    class _Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = _Capture(level=logging.WARNING)
    skysim.log.addHandler(handler)
    try:
        skysim.runit(
            skysim_opts(
                gt2.ms,
                column="DATA",
                sefd=500.0,
                seed_noise=1,
                corruptions=yaml_path,
                log_level="WARNING",
            )
        )
    finally:
        skysim.log.removeHandler(handler)

    assert any("no effect on a noise-only run" in m for m in messages)


def _corruption_peak_bytes(nant, nrow=400, nchan=64, ncorr=2):
    """Peak allocation while computing one corrupted block with `nant` antennas."""
    rng = np.random.default_rng(0)
    ant1 = rng.integers(0, nant, nrow)
    ant2 = (ant1 + 1) % nant
    vis = da.from_array(np.ones((nrow, nchan, ncorr), np.complex128), chunks=(nrow, nchan, ncorr))
    time = da.from_array(np.arange(nrow, dtype=float) * 8.0, chunks=nrow)
    spec = CorruptionSpec(
        terms=["G"],
        spec=[TermSpec(label="G", diagonal=True, axes=["time"], period=120.0, amplitude=0.1)],
    )
    corrupted = apply_corruptions(
        vis,
        time,
        da.from_array(ant1, chunks=nrow),
        da.from_array(ant2, chunks=nrow),
        np.linspace(1.42e9, 1.45e9, nchan),
        spec,
        random_seed=1,
    )
    tracemalloc.start()
    try:
        corrupted.compute()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_gain_memory_does_not_scale_with_antenna_count():
    """Regression: gains were built as an (nant, nrow, nchan) cube and then
    indexed down to the two (nrow, nchan) slices that are actually used, so peak
    memory scaled with the size of the array -- tens of GiB per block at the
    default chunking on a real MS. Only the per-row gains are built now, so a
    64x larger array must not cost measurably more memory.
    """
    small = _corruption_peak_bytes(4)
    large = _corruption_peak_bytes(256)
    assert large < 2 * small, f"peak grew with antenna count: {small} -> {large} bytes"


def read_nant(ms):
    """Antenna count from the ANTENNA subtable -- the authoritative source."""
    return xds_from_table(f"{ms}::ANTENNA")[0].sizes["row"]


def expected_feed_factor(time, ant1, ant2, period, amplitude, random_seed, feed_p, feed_q, nant, label="J"):
    """Reference for one correlation of a `type: diagonal` time-varying term."""
    from simms.skymodel.corruptions import _stable_label_hash

    rng = np.random.default_rng((random_seed or 0) + _stable_label_hash(label))
    phases = rng.random((nant, 2))
    t0 = float(time.min())

    def gain(ant, feed):
        phase = 2.0 * np.pi * ((time - t0) / period + phases[ant, feed])
        return 1.0 + amplitude * (np.cos(phase) + 1j * np.sin(phase))

    return gain(ant1, feed_p) * np.conj(gain(ant2, feed_q))


_DIAGONAL_TYPE_YAML = """
gains:
  terms: [J]
  spec:
    - label: J
      type: diagonal
      complex: true
      axes: [time]
      period:
        time: 120.0
      amplitude: 0.1
"""


def test_diagonal_type_gives_independent_feed_gains(gt2):
    """`type: diagonal` is diag(g_x, g_y): XX and YY get different gains.

    A scalar term drives both hands identically, so this is the term the boolean
    `diagonal: true` could never express.
    """
    yaml_path = gt2.write_yaml(_DIAGONAL_TYPE_YAML)

    skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA"))
    clean = read_column(gt2.ms, "DATA")

    skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA", corruptions=yaml_path, seed_gains=4))
    corrupted = read_column(gt2.ms, "DATA")

    xx = corrupted[:, 0, 0] / clean[:, 0, 0]
    yy = corrupted[:, 0, 1] / clean[:, 0, 1]
    assert not np.allclose(xx, yy), "both feeds got the same gain; this is a scalar term, not diagonal"

    time, ant1, ant2 = read_aux(gt2.ms)
    nant = read_nant(gt2.ms)
    for corr, feed in ((0, 0), (1, 1)):
        expected = expected_feed_factor(time, ant1, ant2, 120.0, 0.1, 4, feed, feed, nant)
        np.testing.assert_allclose(
            corrupted[:, 0, corr] / clean[:, 0, corr], expected.astype(corrupted.dtype), rtol=1e-6, atol=1e-7
        )


def test_diagonal_type_mixes_feeds_in_the_cross_hands():
    """On 4 correlations XY sees feed x on antenna p and feed y on antenna q.

    Driven at the corruption layer with all four correlations set to 1: the
    fixture sky is Stokes I only, so the cross-hands are zero in a predicted
    column and would carry no signal to check.
    """
    nrow, nchan, nant = 16, 3, 6
    rng = np.random.default_rng(0)
    ant1 = rng.integers(0, nant, nrow)
    ant2 = (ant1 + 1 + rng.integers(0, nant - 1, nrow)) % nant
    time = np.arange(nrow, dtype=float) * 30.0
    freqs = np.linspace(1.420e9, 1.428e9, nchan)

    spec = CorruptionSpec(
        terms=["J"],
        spec=[TermSpec(label="J", type="diagonal", axes=["time"], period=120.0, amplitude=0.1)],
    )
    vis = da.from_array(np.ones((nrow, nchan, 4), np.complex128), chunks=-1)
    corrupted = apply_corruptions(
        vis,
        da.from_array(time, chunks=-1),
        da.from_array(ant1, chunks=-1),
        da.from_array(ant2, chunks=-1),
        freqs,
        spec,
        random_seed=4,
        nant=nant,
    ).compute()

    for corr, (fp, fq) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        expected = expected_feed_factor(time, ant1, ant2, 120.0, 0.1, 4, fp, fq, nant)
        np.testing.assert_allclose(corrupted[:, 0, corr], expected, rtol=1e-11, atol=1e-13)

    # The cross-hands must differ from the parallel hands: that is the mixing.
    assert not np.allclose(corrupted[:, 0, 1], corrupted[:, 0, 0])


def test_scalar_type_matches_deprecated_diagonal_true(gt2):
    scalar = gt2.write_yaml(_UNSET_TYPE_YAML.replace("      complex:", "      type: scalar\n      complex:"))
    deprecated = gt2.write_yaml(_explicit_diagonal_yaml("true"))

    skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA", corruptions=scalar, seed_gains=3))
    from_type = read_column(gt2.ms, "DATA")

    skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA", corruptions=deprecated, seed_gains=3))
    from_bool = read_column(gt2.ms, "DATA")

    np.testing.assert_array_equal(from_type, from_bool)


def test_full_type_matches_deprecated_diagonal_false(gt4):
    full = gt4.write_yaml(_UNSET_TYPE_YAML.replace("      complex:", "      type: full\n      complex:"))
    deprecated = gt4.write_yaml(_explicit_diagonal_yaml("false"))

    skysim.runit(skysim_opts(gt4.ms, ascii_sky=gt4.sky, column="DATA", corruptions=full, seed_gains=3))
    from_type = read_column(gt4.ms, "DATA")

    skysim.runit(skysim_opts(gt4.ms, ascii_sky=gt4.sky, column="DATA", corruptions=deprecated, seed_gains=3))
    from_bool = read_column(gt4.ms, "DATA")

    np.testing.assert_array_equal(from_type, from_bool)


def test_type_and_deprecated_diagonal_together_is_an_error(gt2):
    yaml_path = gt2.write_yaml(
        _UNSET_TYPE_YAML.replace("      complex:", "      type: scalar\n      diagonal: true\n      complex:")
    )
    with pytest.raises(RuntimeError, match="both 'type' and the deprecated 'diagonal'"):
        skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA", corruptions=yaml_path, seed_gains=1))


def test_unknown_type_is_an_error(gt2):
    yaml_path = gt2.write_yaml(_UNSET_TYPE_YAML.replace("      complex:", "      type: leakage\n      complex:"))
    with pytest.raises(RuntimeError, match="unknown type 'leakage'"):
        skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA", corruptions=yaml_path, seed_gains=1))


def test_diagonal_type_needs_at_least_two_correlations():
    """diag(g_x, g_y) has nothing to say about a single-correlation MS."""
    spec = CorruptionSpec(
        terms=["J"],
        spec=[TermSpec(label="J", type="diagonal", axes=["time"], period=120.0, amplitude=0.1)],
    )
    with pytest.raises(RuntimeError, match="needs 2 or 4 correlations"):
        validate_spec(spec, ncorr=1)


def test_explicit_references_make_gains_selection_independent():
    """Regression: t0/freq0/nant taken from the rows in hand made the same
    antenna's gain differ between a per-field or per-SPW run and a whole-MS one.
    Passing MS-wide references pins them."""
    nrow, nchan, ncorr, nant = 24, 6, 2, 8
    rng = np.random.default_rng(0)
    ant1 = rng.integers(0, nant, nrow)
    ant2 = (ant1 + 1 + rng.integers(0, nant - 1, nrow)) % nant
    time = np.arange(nrow, dtype=float) * 30.0
    freqs = np.linspace(1.420e9, 1.440e9, nchan)
    spec = CorruptionSpec(
        terms=["J"],
        spec=[
            TermSpec(
                label="J",
                type="scalar",
                axes=["time", "frequency"],
                period={"time": 200.0, "frequency": 8e6},
                amplitude=0.1,
            )
        ],
    )
    refs = dict(nant=nant, time_ref=float(time[0]), freq_ref=float(freqs[0]))

    def run(rows, chans, **kwargs):
        vis = da.from_array(np.ones((len(rows), len(chans), ncorr), np.complex128), chunks=-1)
        return apply_corruptions(
            vis,
            da.from_array(time[rows], chunks=-1),
            da.from_array(ant1[rows], chunks=-1),
            da.from_array(ant2[rows], chunks=-1),
            freqs[chans],
            spec,
            random_seed=1,
            **kwargs,
        ).compute()

    whole = run(np.arange(nrow), np.arange(nchan), **refs)
    # A later, narrower selection: different first row, different first channel,
    # and it happens to omit the highest-numbered antenna.
    rows = np.array([i for i in range(8, nrow) if ant1[i] != nant - 1 and ant2[i] != nant - 1])
    chans = np.arange(2, nchan)

    with_refs = run(rows, chans, **refs)
    np.testing.assert_array_equal(with_refs, whole[np.ix_(rows, chans)])

    # Without them the same rows get different gains -- the bug being guarded.
    without_refs = run(rows, chans)
    assert not np.allclose(without_refs, whole[np.ix_(rows, chans)])


_REAL_TERM = """
    - label: G
      type: scalar
      complex: true
      axes: [time]
      period:
        time: 120.0
      amplitude: 0.1
"""


def test_spec_with_no_terms_is_an_error(gt2):
    """A spec listing no terms corrupts nothing; say so rather than run silently."""
    yaml_path = gt2.write_yaml(f"gains:\n  terms: []\n  spec:{_REAL_TERM}")
    with pytest.raises(RuntimeError, match="lists no terms"):
        skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA", corruptions=yaml_path))


def test_spec_without_a_gains_block_is_an_error(gt2):
    """A misspelled top-level key used to load as an empty spec and do nothing."""
    yaml_path = gt2.write_yaml(f"corruptions:\n  terms: [G]\n  spec:{_REAL_TERM}")
    with pytest.raises(RuntimeError, match="no top-level 'gains' block"):
        skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA", corruptions=yaml_path))


def test_empty_spec_file_is_an_error(gt2):
    yaml_path = gt2.write_yaml("")
    with pytest.raises(RuntimeError, match="empty or is not a YAML mapping"):
        skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA", corruptions=yaml_path))


def test_all_zero_amplitudes_is_an_error(gt2):
    """One zero term is the identity; a spec of nothing but zeros is a mistake."""
    yaml_path = gt2.write_yaml(
        """
gains:
  terms: [G, H]
  spec:
    - label: G
      type: scalar
      axes: [time]
      period:
        time: 120.0
      amplitude: 0.0
    - label: H
      type: scalar
      axes: [time]
      period:
        time: 300.0
      amplitude: 0.0
"""
    )
    with pytest.raises(RuntimeError, match="amplitude 0"):
        skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA", corruptions=yaml_path))


def test_zero_amplitude_is_allowed_beside_a_real_term(gt2):
    """The all-zero check must not reject a legitimate identity term."""
    yaml_path = gt2.write_yaml(
        """
gains:
  terms: [Z, G]
  spec:
    - label: Z
      type: scalar
      axes: [time]
      period:
        time: 300.0
      amplitude: 0.0
    - label: G
      type: scalar
      axes: [time]
      period:
        time: 120.0
      amplitude: 0.1
"""
    )
    skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA", corruptions=yaml_path, seed_gains=1))
    assert np.all(np.isfinite(read_column(gt2.ms, "DATA")))


def test_misspelled_term_key_names_the_term(gt2):
    """The dataclass raises TypeError; the loader must name the file and term."""
    yaml_path = gt2.write_yaml(
        """
gains:
  terms: [G]
  spec:
    - label: G
      amplitide: 0.1
      axes: [time]
      period:
        time: 120.0
"""
    )
    with pytest.raises(RuntimeError, match=r"term 'G'.*amplitide"):
        skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA", corruptions=yaml_path))


def test_term_without_a_label_is_reported_by_position(gt2):
    yaml_path = gt2.write_yaml(
        """
gains:
  terms: [G]
  spec:
    - axes: [time]
      period:
        time: 120.0
      amplitude: 0.1
"""
    )
    with pytest.raises(RuntimeError, match=r"term 'entry 0'.*label"):
        skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA", corruptions=yaml_path))


def reference_jones(params, time, freqs, ant, t0, freq0):
    """Independent per-term 2x2 Jones for one antenna column: (nrow, nchan, 2, 2).

    Assembles the matrix forms directly from the term parameters -- it does not
    call the module's Jones builder, so the matrix shapes, the feed-to-row/column
    placement and the composition order are checked rather than restated.
    """
    nrow, nchan = time.size, freqs.size
    osc = np.ones((nrow, nchan, params["nfeed"]), dtype=np.complex128)
    for axis in params["axes"]:
        period = params["period"][axis]
        phases = params["phases"][axis][ant][:, None, :]  # (nrow, 1, nfeed)
        x, x0 = (time[:, None, None], t0) if axis == "time" else (freqs[None, :, None], freq0)
        phase = 2.0 * np.pi * ((x - x0) / period + phases)
        # exp(i phi) rather than cos + i sin: the reference should not restate
        # the implementation's spelling of the same thing.
        osc = osc * (np.exp(1j * phase) if params["complex"] else np.cos(phase))
    amp = params["amplitude"]

    out = np.zeros((nrow, nchan, 2, 2), dtype=np.complex128)
    if params["type"] == "scalar":
        g = 1.0 + amp * osc[..., 0]
        out[..., 0, 0] = g
        out[..., 1, 1] = g
    elif params["type"] == "diagonal":
        out[..., 0, 0] = 1.0 + amp * osc[..., 0]
        out[..., 1, 1] = 1.0 + amp * osc[..., 1]
    else:
        m = params["matrix"][ant][:, None, :, :]
        out = np.eye(2) + amp * osc[..., 0][:, :, None, None] * m
    return out


def reference_corrupted(spec, vis, time, freqs, ant1, ant2, nant, seed):
    """V' = J_p V J_q^H with the terms composed left-to-right, by explicit matmul."""
    from simms.skymodel.corruptions import _build_all_term_params

    params = _build_all_term_params(spec, nant, seed, vis.shape[-1])
    t0, freq0 = float(time.min()), float(freqs.min())

    def chain(ant):
        j = np.broadcast_to(np.eye(2), (time.size, freqs.size, 2, 2)).astype(np.complex128)
        for pr in params:
            j = j @ reference_jones(pr, time, freqs, ant, t0, freq0)
        return j

    jp, jq = chain(ant1), chain(ant2)
    vmat = vis.reshape(vis.shape[0], vis.shape[1], 2, 2)
    out = jp @ vmat @ np.conj(np.swapaxes(jq, -2, -1))
    return out.reshape(vis.shape)


def _mixed_case(nrow=12, nchan=3, nant=5):
    rng = np.random.default_rng(3)
    ant1 = rng.integers(0, nant, nrow)
    ant2 = (ant1 + 1 + rng.integers(0, nant - 1, nrow)) % nant
    time = np.arange(nrow, dtype=float) * 45.0
    freqs = np.linspace(1.420e9, 1.428e9, nchan)
    vis = (rng.standard_normal((nrow, nchan, 4)) + 1j * rng.standard_normal((nrow, nchan, 4))).astype(np.complex128)
    return vis, time, freqs, ant1, ant2, nant


def _run_spec(spec, vis, time, freqs, ant1, ant2, nant, seed=7):
    return apply_corruptions(
        da.from_array(vis, chunks=-1),
        da.from_array(time, chunks=-1),
        da.from_array(ant1, chunks=-1),
        da.from_array(ant2, chunks=-1),
        freqs,
        spec,
        random_seed=seed,
        nant=nant,
    ).compute()


@pytest.mark.parametrize("order", [["F", "D"], ["D", "F"], ["F", "S"], ["S", "F"], ["F", "G"]])
def test_full_jones_matches_explicit_matmul_reference(order):
    """The full-Jones path against an independent J_p V J_q^H.

    Every ordering of a full term with a scalar/diagonal one: the terms must
    compose left-to-right, so a diagonal term after a full one is a *right*
    multiplication. Only checking finiteness let an antenna swap, a missing
    conjugate or a reversed product through.
    """
    vis, time, freqs, ant1, ant2, nant = _mixed_case()
    catalogue = {
        "F": TermSpec(label="F", type="full", axes=["time"], period=150.0, amplitude=0.2),
        "D": TermSpec(label="D", type="diagonal", axes=["time"], period=120.0, amplitude=0.3),
        "S": TermSpec(label="S", type="scalar", axes=["time"], period=90.0, amplitude=0.15),
        "G": TermSpec(label="G", type="full", axes=["frequency"], period=6e6, amplitude=0.25),
    }
    spec = CorruptionSpec(terms=list(order), spec=[catalogue[k] for k in order])

    got = _run_spec(spec, vis, time, freqs, ant1, ant2, nant)
    want = reference_corrupted(spec, vis, time, freqs, ant1, ant2, nant, 7)
    np.testing.assert_allclose(got, want, rtol=1e-11, atol=1e-13)


def test_full_and_diagonal_do_not_commute_in_the_spec():
    """Guards the reference itself: if [F, D] and [D, F] agreed, the ordering
    test above would prove nothing."""
    vis, time, freqs, ant1, ant2, nant = _mixed_case()
    F = TermSpec(label="F", type="full", axes=["time"], period=150.0, amplitude=0.2)
    D = TermSpec(label="D", type="diagonal", axes=["time"], period=120.0, amplitude=0.3)
    fd = _run_spec(CorruptionSpec(terms=["F", "D"], spec=[F, D]), vis, time, freqs, ant1, ant2, nant)
    df = _run_spec(CorruptionSpec(terms=["D", "F"], spec=[D, F]), vis, time, freqs, ant1, ant2, nant)
    assert not np.allclose(fd, df)


def test_time_term_is_invariant_under_row_chunking():
    """Regression twin of the frequency test: the time origin is the whole
    selection's, so row-chunk boundaries must not restart the sinusoid."""
    nrow, nchan, ncorr, nant = 24, 2, 2, 4
    rng = np.random.default_rng(1)
    ant1 = rng.integers(0, nant, nrow)
    ant2 = (ant1 + 1) % nant
    time = np.arange(nrow, dtype=float) * 30.0
    freqs = np.linspace(1.420e9, 1.424e9, nchan)
    spec = CorruptionSpec(
        terms=["G"],
        spec=[TermSpec(label="G", type="scalar", axes=["time"], period=180.0, amplitude=0.2)],
    )

    def run(row_chunk):
        vis = da.from_array(np.ones((nrow, nchan, ncorr), np.complex128), chunks=(row_chunk, nchan, ncorr))
        return apply_corruptions(
            vis,
            da.from_array(time, chunks=row_chunk),
            da.from_array(ant1, chunks=row_chunk),
            da.from_array(ant2, chunks=row_chunk),
            freqs,
            spec,
            random_seed=1,
            nant=nant,
        ).compute()

    np.testing.assert_allclose(run(nrow), run(7), rtol=1e-13, atol=1e-14)
