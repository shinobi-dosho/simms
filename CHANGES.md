### 3.0.1 -> unreleased

- `skysim`: add YAML-driven RIME Jones corruptions via `--corruptions`. Terms are
  described with arbitrary labels, axes (`time` and/or `frequency`), diagonal or
  full 2x2 Jones matrices, and sinusoidal amplitudes/periods. Periods accept
  raw numbers (seconds/Hz) or `astropy`-compatible strings such as `"2min"` and
  `"2MHz"`. The per-baseline corruption is `V' = J_p V J_q^H`, with terms
  multiplied in the order listed. Full (non-diagonal) terms require a
  4-correlation MS; omitting `diagonal` follows the MS, giving a full 2x2 Jones
  on a 4-correlation MS and a scalar gain otherwise. Corruptions are applied to
  the model before thermal noise is added (`V' = J_p V J_q^H + n`), so `--sefd`
  noise is not gain-modulated. Phase references and the gain array size are taken
  once over the whole selection rather than per dask block, so a corruption is
  invariant under row and channel chunking; note the references are per field
  and per SPW, so separate runs over different fields or SPWs of one MS do not
  share a time or frequency origin. Gains are evaluated per row rather than as
  an `(nant, nrow, nchan)` cube that was then indexed down to two slices, so
  peak memory no longer scales with the size of the array: a 64-antenna,
  1024-channel MS needed ~9.8 GiB per block (~39 GiB for full Jones) at the
  default `--row-chunks`, which made `--corruptions` unusable on a real MS.
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

