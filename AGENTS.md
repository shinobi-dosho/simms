# simms -- design conventions

Radio-interferometry simulator. `telsim` builds a Measurement Set from a telescope layout;
`skysim` predicts model visibilities from a sky model into an MS; `primary-beam` provides
standalone beam operations. Single CLI entry point `simms` (`simms.apps.main:cli`), with
subcommands wired in `src/simms/apps/` (one module + one `<name>.yaml` cab per subcommand).
Src layout: the importable package lives under `src/simms/`, not at the repo root.

Organisation-wide conventions live in
[`shinobi-dosho/.github`](https://github.com/shinobi-dosho/.github/blob/main/AGENTS.md) — this file states what is
specific to `simms` and wins where the two disagree.

## Environment & commands

Use `uv` for everything — never call `python`/`pytest`/`ruff` directly.

- Run code: `uv run python ...`, or the CLI via `uv run simms <subcommand> ...`
- Tests: `uv run --group tests python -m pytest` (a specific file: `... python -m pytest tests/<name>_tests.py`)
- Lint/format: `uv run --group ruff ruff check src tests` and `uv run --group ruff ruff format <paths>`

The repo ships a tracked git hook at `.githooks/pre-commit` that runs `ruff check` and
`ruff format --check` over the staged Python files. Enable it once per clone with
`git config core.hooksPath .githooks`. It reports rather than rewrites, so a formatting
failure means running `ruff format` yourself and re-staging.

## Reading dependency source (important)

**Never use a local sibling checkout as the source of truth for a dependency.** Repos such as
`stimela-ninja`, `dosho`, `scabha`, `fitstoolz` and `msutils` may be checked out next to this
one, but they are under active development — uncommitted work, feature branches, detached
HEADs — so their working trees do not reflect `origin/main` or any release.

Clone the dependency fresh from its remote into a scratch directory and read that instead:

```
git clone -q git@github.com:shinobi-dosho/stimela-ninja.git /tmp/<scratch>/ninja-src
```

Reading a local clone's `git remote -v` to find the URL is fine; reading its working tree is
not. To see what changed against what is installed here, diff the fresh clone against the
pinned release tag (`git diff v0.1.0b3..origin/main`), never against a local checkout.

## Tests

- Test files must be named `*_tests.py` (pytest is configured with `python_files = ["*_tests.py"]`);
  a `foo_test.py` or `test_foo.py` will not be collected.
- Temp MSs/files/dirs go through `tests.InitTest` (`random_named_file` / `random_named_directory`),
  which registers them for cleanup — don't hand-roll `tempfile`.
- Heavy or optional dependencies are opt-in dependency groups and guarded with
  `pytest.importorskip`, so the default `tests` run stays light. Example: the CASA round-trip
  test needs the `casa` group — `uv run --group tests --group casa python -m pytest tests/casa_roundtrip_tests.py`.

## MS conventions (load-bearing, easy to get wrong)

- **Metadata has a single authoritative source; never infer it.** The per-antenna telescope/type
  label lives in the `ANTENNA` table column named by `--telescope-name-column` (default
  `TELESCOPE_NAME`). Read it and fail clearly if absent — do not guess from `DISH_DIAMETER` etc.
- **Pointing vs phase centre are different.** `FIELD.PHASE_DIR` is the correlator phase-tracking
  centre (arbitrary, shiftable). The primary beam is centred on the antenna pointing centre in
  `POINTING.DIRECTION`. Use `simms.skymodel.beams.read_pointing_centre` for the beam centre.
- **`SPECTRAL_WINDOW.MEAS_FREQ_REF` must be set** (5 == TOPO). casacore defaults it to 0 (REST),
  which leaves the spectral frame undefined and makes CASA imaging fail ("No MeasFrame specified
  for conversion of Frequency").
- **casacore STRING columns are numpy `object` dtype**, written in one chunk
  (`da.from_array(values, chunks=n)`). Adding a *new* column to a standard subtable needs an
  explicit descriptor, e.g. `xds_to_table(..., "{ms}::ANTENNA", columns=[col], descriptor="mssubtable('ANTENNA')")`.

## Beam data

Cosine-taper (`beams.py`) tables under `src/simms/skymodel/beam_data/`. The `MKAT-AA-*` model and its
tables are vendored from katbeam (BSD-3-Clause) — keep that attribution in `beam_data/NOTICE`. The
other tables ship as ordinary bundled package data.

## Corruptions

`skysim` applies RIME Jones corruptions after prediction when `--corruptions` points to a YAML
spec. The spec lists an ordered `terms` array of arbitrary labels and a `spec` array describing each
term: `axes` (`time` and/or `frequency`), `type`, `complex`, `amplitude`, and `period`.
`type` is `scalar` (`g I`), `diagonal` (`diag(g_x, g_y)`, needs >= 2 correlations) or `full`
(dense 2x2, needs 4). Omitting it gives `diagonal`, falling back to `scalar` only on a
1-correlation MS; leakage is never implied. An explicit type the MS cannot carry is an
error, not a downgrade. The boolean
`diagonal: true/false` is deprecated and maps to `scalar`/`full` (it never meant the `diagonal`
type).
Periods are in seconds/Hz or `astropy`-compatible strings (e.g. `"2min"`, `"2MHz"`). The
per-baseline corruption is `V' = J_p V J_q^H`, with terms multiplied left-to-right in the order
given. `diagonal`/`full` terms map correlation index to feed index positionally, so they
require a standard linear/circular `POLARIZATION.CORR_TYPE` (validated via `_corr_basis`,
shared with the beam path); `scalar` terms do not. Phase origins come from the whole MS (earliest time, lowest frequency) and the gain array
from the `ANTENNA` table, not from the field/SPW a run selects, so per-field runs agree.
Corruptions are applied to the
model *before* thermal noise is added (`V' = J_p V J_q^H + n`), so the noise is never
gain-modulated. `--seed-noise` seeds thermal
noise and `--seed-gains` the corruption terms, so corruptions cannot alter the noise realisation.
`--seed` is a deprecated alias for `--seed-noise` (the same value gives the same noise).

## Git

- Branch off `main` for changes; open PRs against `main` (repo `shinobi-dosho/simms`).
- End commit messages with the agent's attribution trailer, and keep it off the PR body --
  see *Attribution: commit trailers yes, PR trailers no* below.
- `gh pr edit --body` can fail on this repo with a Projects-classic GraphQL error; edit the body
  via `gh api -X PATCH repos/shinobi-dosho/simms/pulls/<n> -F body=@file` instead (capital `-F`;
  lowercase `-f` sets the body to the literal string `@file`).

## Attribution

The organisation-wide rule applies: commits carry an
`Assisted-by: <AGENT> <MODEL>` trailer (no address, so no GitHub
co-authorship), and PR descriptions carry no trailer at all. See the
[org-wide file](https://github.com/shinobi-dosho/.github/blob/main/AGENTS.md)
for the reasoning.
