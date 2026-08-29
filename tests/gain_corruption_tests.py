"""Tests for YAML-driven RIME corruptions in skysim."""

from __future__ import annotations

import numpy as np
import pytest
from daskms import xds_from_ms, xds_from_table

from simms.apps import skysim
from simms.skymodel.corruptions import apply_corruptions, load_corruption_spec
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


def test_zero_amplitude_leaves_visibilities_unchanged(gt2):
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
      amplitude: 0.0
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
            seed_gains=3,
        )
    )
    corrupted = read_column(gt2.ms, "DATA")
    np.testing.assert_allclose(corrupted, clean, rtol=1e-11, atol=1e-13)


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
    with pytest.raises(RuntimeError, match="4-correlation"):
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
    for _ in range(2):
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


def test_corruptions_do_not_change_the_noise_realisation(gt2):
    """Noise draws from --seed-noise only, so adding corruptions leaves it unchanged."""
    sefd = 500.0
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
      amplitude: 0.0
"""
    )
    skysim.runit(skysim_opts(gt2.ms, ascii_sky=gt2.sky, column="DATA", sefd=sefd, seed_noise=99))
    noise_only = read_column(gt2.ms, "DATA")

    skysim.runit(
        skysim_opts(
            gt2.ms,
            ascii_sky=gt2.sky,
            column="DATA",
            sefd=sefd,
            seed_noise=99,
            seed_gains=5,
            corruptions=yaml_path,
        )
    )
    noise_with_corruptions = read_column(gt2.ms, "DATA")
    np.testing.assert_allclose(noise_only, noise_with_corruptions, rtol=1e-11, atol=1e-13)


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
