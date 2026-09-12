# Contributing to simms

Thanks for your interest in contributing! **simms** is a radio-interferometry
simulator: `telsim` builds a Measurement Set from a telescope layout, `skysim`
predicts model visibilities from a sky model into it, and `primary-beam`
provides standalone beam operations.

By participating you agree to abide by our
[Code of Conduct](https://github.com/shinobi-dosho/simms/blob/main/CODE_OF_CONDUCT.md).

## Scope and philosophy

simms **simulates**: `telsim` builds a Measurement Set from a telescope layout,
`skysim` predicts model visibilities from a sky model into an MS, and
`primary-beam` provides standalone beam operations. It does not calibrate,
image, or otherwise process real observations — those belong to the tools built
for them. A feature that would turn simms into a processing package is out of
scope, however useful.

Read [`AGENTS.md`](https://github.com/shinobi-dosho/simms/blob/main/AGENTS.md) before touching anything under
*MS conventions* — those are load-bearing and easy to get wrong.

## Ways to contribute

- **Report bugs** and request features via [issues](https://github.com/shinobi-dosho/simms/issues).
- **Improve documentation** under `docs/` or the docstrings that feed the API
  reference.
- **Submit code** -- bug fixes, new telescope layouts, sky-model formats,
  tests.

## Development setup

The project uses [uv](https://docs.astral.sh/uv/) for everything -- avoid
calling `python`/`pytest`/`ruff` directly:

```bash
git clone https://github.com/shinobi-dosho/simms.git
cd simms
uv run --group tests python -m pytest
uv run --group ruff ruff check src tests

# enable the repo's pre-commit hook (once per clone)
git config core.hooksPath .githooks
```

The hook lives at [`.githooks/pre-commit`](https://github.com/shinobi-dosho/simms/blob/main/.githooks/pre-commit),
tracked in the repo so it is reviewable in diffs and identical for everyone. Git
will not let a repository install an executable hook by itself, hence the one
`git config` command. It runs ruff through `uv run --group ruff`, the same way
CI does, so the two cannot disagree about which version is in force.

## Testing

Run the default suite with:

```bash
uv run --group tests python -m pytest
```

- Test files must be named `*_tests.py` (pytest is configured with
  `python_files = ["*_tests.py"]`) -- a `test_foo.py` will not be collected.
- Temp MSs/files/dirs should go through `tests.InitTest`
  (`random_named_file`/`random_named_directory`), which registers them for
  cleanup, rather than hand-rolled `tempfile` usage.
- Heavy or optional dependencies are opt-in dependency groups, guarded with
  `pytest.importorskip`, so the default run stays light. For example, the
  CASA round-trip test needs the `casa` group:

  ```bash
  uv run --group tests --group casa python -m pytest tests/casa_roundtrip_tests.py
  ```

**New features and bug fixes should come with tests.**

## Code style

- **Lint must be clean**: `ruff check src tests` should report no errors.
- `ruff format` is available for autoformatting; the pre-commit hook runs
  `ruff check` and `ruff format --check` over the Python files a commit
  touches. It reports rather than rewrites -- it cannot tell which hunks of a
  file you staged, so reformatting in place would drag unstaged work into the
  commit. On a formatting failure, run `uv run --group ruff ruff format src
  tests`, re-stage, and commit again. `git commit --no-verify` bypasses it.
- Use type hints and docstrings on public API -- they render into the Sphinx
  API reference via autodoc.
- Match the surrounding code's naming, comment density, and idiom.

### MS conventions

The package works with `casacore` Measurement Sets and has a few load-bearing
conventions around metadata (telescope-name column, phase vs. pointing
centre, spectral frame) -- see the *MS conventions* section of
[`CLAUDE.md`](https://github.com/shinobi-dosho/simms/blob/main/CLAUDE.md) (also
readable as `AGENTS.md`) and the
[MS conventions doc page](https://simms.readthedocs.io/en/latest/concepts/ms-conventions.html)
before touching MS I/O code.

## Documentation

Docs are built with Sphinx (Furo theme) and hosted on Read the Docs. Build
them locally with:

```bash
uv sync --group docs
uv run sphinx-build -b html docs docs/_build/html
```

Please update the docs when you change public API. Documentation dependencies
live in the `docs` dependency group in `pyproject.toml` and nowhere else: Read
the Docs builds from `uv.lock` via that group, so adding one there is all that
is needed.

## Pull requests

1. Branch off `main` and keep PRs **small and focused**.
2. Make sure `pytest` and `ruff check src tests` pass locally, and that docs
   build if you touched public API.
3. Push and open a PR against `main`. Reference any related issue
   (e.g. "Closes #12").
4. CI must be green.

### Commit messages

Write clear, descriptive commit messages explaining *why* a change is made.

Provenance for an assistant-assisted commit goes in a commit trailer and never
in the PR description — see
[*Attribution: commit trailers yes, PR trailers no*](https://github.com/shinobi-dosho/simms/blob/main/AGENTS.md#attribution-commit-trailers-yes-pr-trailers-no).

## Versioning and releases

The project follows [Semantic Versioning](https://semver.org/). Releases are
maintainer-driven: the maintainer bumps `version` in `pyproject.toml` and
publishes a GitHub release, which triggers the publish workflow.

## License

By contributing, you agree that your contributions are licensed under the
project's [Apache License 2.0](https://github.com/shinobi-dosho/simms/blob/main/LICENSE).

You are responsible for ensuring that you have the right to submit your
contribution under this licence, including any necessary employer or
third-party permissions. Do not submit confidential, proprietary, or
otherwise restricted material that you are not authorised to release.
The project's open-source terms apply irrespective of contributors'
employment or institutional affiliations. See [NOTICE](https://github.com/shinobi-dosho/simms/blob/main/NOTICE).
