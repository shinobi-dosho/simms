### 3.0.1 -> unreleased

- `skysim`: predicted visibilities now carry time and bandwidth smearing, via
  the new `--smearing` option (`analytic`, the default, or `none` for the
  previous monochromatic, instantaneous prediction). The correlator averages
  each visibility over `CHAN_WIDTH` and over `EXPOSURE`, so an unsmeared model
  over-predicts every source away from the phase centre, by more the longer the
  baseline. That was a modelling choice for a simulation from scratch but a
  silent correctness bug for `--mode subtract` against real averaged data,
  where it left a residual growing with baseline length and with offset from
  the phase centre. Each average is over a top-hat, so each contributes a real
  `sinc` factor on the phasor: the bandwidth term is frequency-independent and
  costs one `sinc` per row and source, the time term follows the fringe rate of
  a baseline turning with the Earth at the phase-centre declination and is
  evaluated per channel. Both are exactly 1 at the phase centre. Worked
  example, MeerKAT L-band (1.28 GHz, 7.7 km baselines) averaged to 128 channels
  and 8 s dumps: a source 51' off axis decorrelates to ~0.11 of its coherent
  amplitude on the longest baselines, so the old model over-predicted it
  ninefold there. Only the
  per-visibility kernels can apply it, so it covers `--ascii-sky`,
  `--wsclean-sky` and `--fits-sky` with `--predict-backend dft` (including the
  component bridge used for a FITS model with few bright pixels and a primary
  beam); the FITS gridder and a-term backends transform whole images, and a run
  that lands on one warns and predicts unsmeared rather than silently claiming
  to have smeared. On a uniform channel grid the time factor's sine rides a
  second phasor recurrence, the way the phase itself already does, and the
  unsmeared loop is kept separate from the smeared one: `--smearing none` runs
  the kernel it always ran, and `analytic` costs ~1.2-1.4x on a 300-source,
  256-channel, 4000-row predict.

- `skysim`: add YAML-driven RIME Jones corruptions via `--corruptions`. Terms are
  described with arbitrary labels, axes (`time` and/or `frequency`), diagonal or
  full 2x2 Jones matrices, and sinusoidal amplitudes/periods. Periods accept
  raw numbers (seconds/Hz) or `astropy`-compatible strings such as `"2min"` and
  `"2MHz"`. The per-baseline corruption is `V' = J_p V J_q^H`, with terms
  multiplied in the order listed. Full (non-diagonal) terms require a
  4-correlation MS. `type` selects the Jones form -- `scalar` (`g I`), `diagonal`
  (`diag(g_x, g_y)`, independent per-feed gains, needs at least 2 correlations)
  or `full` (dense 2x2, needs 4) -- and omitting it gives `diagonal`, falling
  back to `scalar` only on a 1-correlation MS. Leakage is never implied. The boolean
  `diagonal: true/false` is deprecated in favour of `type: scalar/full`; it
  warns and still works, and never meant the new `diagonal` form. Corruptions are applied to
  the model before thermal noise is added (`V' = J_p V J_q^H + n`), so `--sefd`
  noise is not gain-modulated. Phase references and the gain array size are taken
  once over the whole selection rather than per dask block, so a corruption is
  invariant under row and channel chunking; note the references are per field
  and per SPW. `skysim` therefore passes MS-wide references -- the earliest time,
  the lowest frequency across all SPWs, and the `ANTENNA` table's row count -- so
  separate `--field-id`/`--spw-id` runs over one MS give the same antenna the
  same gain. Gains are evaluated per row rather than as
  an `(nant, nrow, nchan)` cube that was then indexed down to two slices, so
  peak memory no longer scales with the size of the array: a 64-antenna,
  1024-channel MS needed ~9.8 GiB per block (~39 GiB for full Jones) at the
  default `--row-chunks`, which made `--corruptions` unusable on a real MS.
- `skysim`: the corruption time/frequency phase references now span every
  `SPECTRAL_WINDOW` dataset, not just the first, so an MS whose SPWs differ in
  channel count no longer references only part of its bandwidth. The deprecated
  `diagonal:` spelling now warns once per term when the spec is loaded rather
  than on every internal type resolution.
- `skysim`: a corruption `spec` entry that `gains.terms` does not list is no
  longer required to fit the MS, so one file can hold a library of terms and
  `terms` select among them; an unused `full` entry no longer aborts a
  scalar-only run on a 2-correlation MS. Structural validity (axes, period,
  amplitude, type declaration) is still required of every entry, listed or not.
- `skysim`: `diagonal` and `full` corruption terms now check
  `POLARIZATION.CORR_TYPE` and require a standard linear (`XX..YY`) or circular
  (`RR..LL`) ordering, as the primary-beam path already did; they map
  correlation index onto feed index by position, so a non-standard ordering
  silently assigned the wrong feed. `scalar` terms are unaffected.
- `skysim`: a `--corruptions` spec that would corrupt nothing is now an error
  rather than a silent no-op: an empty file, one with no top-level `gains` block
  (a misspelled key used to load as an empty spec), an empty `gains.terms` list,
  or a spec where every listed term has `amplitude: 0`. A single zero-amplitude
  term alongside a real one is still the identity and remains valid. A missing
  `label` or a misspelled term key now raises the same `RuntimeError` as the
  rest of the loader, naming the file and the term, instead of a bare
  `TypeError` from the dataclass.
- `skysim`: deprecate `--seed` in favour of `--seed-noise` (the same value gives
  the same noise realisation; a deprecation warning is emitted). Corruption
  terms draw from their own `--seed-gains`, so adding corruptions cannot change
  the noise realisation.
- `skysim`: `--row-chunks` is now an upper bound rather than a fixed chunk size.
  A fixed size tied the task count to the length of the MS, so a short track
  produced fewer chunks than workers and left most of them idle (76608 rows at
  the 10000-row default is 8 chunks, pinning `--nworkers 32` to ~8 cores). The
  size is now reduced so each worker gets several chunks, floored at 256 rows.
  Measured on MeerKAT: 2.4x faster on a 5-min/10k-source predict (67.0s -> 28.0s,
  720% -> 2420% CPU), 1.1-1.4x on longer tracks. Chunk size is never increased,
  so memory per task cannot grow. Note that, as before, the thermal-noise
  realisation for a given `--seed` depends on the chunking, so a `--sefd` run
  reproduces a previous realisation only if `--row-chunks` is set explicitly.
- Fast image-domain a-term (primary-beam) correction for the FITS-image
  prediction path (`simms.skymodel.aterms`), in the spirit of WSClean's IDG and
  DDFacet's facet beams but with no spatial approximation: per-antenna beams are
  applied to the full image per baseline-type class, interpolated linearly in
  time on the parallactic-angle grid (algebraically identical to the exact
  per-component DFT kernels, asserted in the tests) and between adaptively
  chosen frequency knots (`--aterm-freq-tol`; `0` samples every channel). This
  replaces the PA-averaged single-antenna power beam as the default
  (`--fits-beam-mode aterm`); the legacy approximation remains available as
  `--fits-beam-mode average`, and is the automatic fallback for diagonal beams
  on a circular-correlation MS. `--beam-jones full` is now honoured on the
  FITS-image path. When the DFT backend wins the FITS cost model under a beam,
  the model is bridged to the exact per-component beam kernels instead of the
  approximate image beam.
- A-term review follow-ups: an image too large for the voltage-map cache now
  degrades to `--fits-beam-mode average` with a warning instead of raising
  `MemoryError` (the ceiling is still enforced for direct API callers, and its
  message names that escape); apparent-beam products are yielded one correlation
  at a time rather than materialised per pass, and memoised across a channel
  segment in diagonal mode; `attach_fits_aterm` and `component_sky_from_fits_dft`
  reject a circular-basis model rather than silently producing wrong cross-hands;
  the planned gridder-pass count is logged at DEBUG.

### 3.0.0 -> 3.0.1

- gitingore poetry|uv.lock
- Port cabs to shinobi/dosho
- Fix `-pb` duplicate abbrevation. The `primary-beam` option retains it
  and `predict-backend` looses it
- `primary-beam` declares `ms` alongside `output` in its pystep outputs, so
  `tag-ms` (which mutates the MS in place) can be chained from in a recipe

