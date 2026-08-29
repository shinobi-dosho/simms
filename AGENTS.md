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
term: `axes` (`time` and/or `frequency`), `diagonal`/`complex` flags, `amplitude`, and `period`.
Periods are in seconds/Hz or `astropy`-compatible strings (e.g. `"2min"`, `"2MHz"`). The
per-baseline corruption is `V' = J_p V J_q^H`, with terms multiplied left-to-right in the order
given. Non-diagonal (full 2x2) terms require a 4-correlation MS. `--random-seed` seeds both thermal
noise and corruption terms, using isolated sub-seeds so corruptions do not alter the noise
realisation.

## Git

- Branch off `main` for changes; open PRs against `main` (repo `shinobi-dosho/simms`).
- End commit messages with the agent's attribution trailer, and keep it off the PR body --
  see *Attribution: commit trailers yes, PR trailers no* below.
- `gh pr edit --body` can fail on this repo with a Projects-classic GraphQL error; edit the body
  via `gh api -X PATCH repos/shinobi-dosho/simms/pulls/<n> -F body=@file` instead (capital `-F`;
  lowercase `-f` sets the body to the literal string `@file`).

## Reviewing changes: check the tree, not just the diff

A claim that something "doesn't exist" or "is unused" should be verified against
the actual tree before acting on it — a symbol absent from the diff is usually
present in the repo.

## Attribution: commit trailers yes, PR trailers no

A commit made with an assistant's help says so in a trailer on the
**commit message**. Use whatever trailer the agent emits by default --
Claude Code, for instance, ends a commit with

```
Co-Authored-By: Claude <noreply@anthropic.com>
```

An agent with no default of its own uses the same form, naming itself and
the model behind it, with an address:

```
Co-authored-by: <AGENT> <MODEL> <EMAIL>
```

— e.g. `Co-authored-by: Codex GPT-5 <noreply@openai.com>`. One line, last
in the message, after any `Co-authored-by:` for real people. The address
is not decoration: GitHub only renders a trailer as co-authorship when it
carries an `<email>`, so without one the credit stays plain text in the
message body. Credit is the point — these tools do real work here, and
the history should say so.

**Pull request descriptions carry no trailer at all** — no
`Co-authored-by:`, no "Generated with", no tool badge. A PR body is
review material: it exists to tell a reviewer what changed and why, and
what to check. Provenance already lives on every commit the PR contains,
where it is attached to the specific change rather than repeated once
per PR, so a trailer in the description is duplication in the one place
that has no room for it. Agents default to adding one; delete it.

Neither form is a substitute for the message itself. A commit that
explains a decision badly does not improve by naming the model that
helped make it — see the existing history for the standard: what
changed, what it deviates from and why, and what a reviewer should not
assume held still.
