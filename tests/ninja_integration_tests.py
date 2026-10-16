"""Optional workflow adapters: uv run --extra ninja --group tests python -m pytest tests/ninja_integration_tests.py."""

import pytest
from click.testing import CliRunner

pytest.importorskip("shinobi")

from shinobi.cli import main as ninja_cli
from shinobi.clickutil import build_options as ninja_options
from shinobi.steps.dispatch import _dispatch

from simms.apps import primary_beam, skysim, telsim
from simms.apps.schema import build_options

from . import InitTest


@pytest.fixture(autouse=True)
def workflow_directory(monkeypatch):
    files = InitTest()
    monkeypatch.chdir(files.random_named_directory())
    yield files


@pytest.mark.parametrize("module,name", [(telsim, "telsim"), (skysim, "skysim"), (primary_beam, "primary_beam")])
def test_adapter_contract_and_dispatch(module, name, monkeypatch):
    adapter = getattr(module, name)
    assert getattr(module, name) is adapter
    assert set(adapter.step.inputs_model.model_fields) == set(module.inputs_model.model_fields)
    native = build_options(module.inputs_model)
    workflow = ninja_options(adapter.step.inputs_model)
    assert [(opt.name, opt.opts, opt.secondary_opts, opt.multiple) for opt in native] == [
        (opt.name, opt.opts, opt.secondary_opts, opt.multiple) for opt in workflow
    ]
    assert [(opt.type.name, getattr(opt.type, "choices", None), opt.default, opt.required) for opt in native] == [
        (opt.type.name, getattr(opt.type, "choices", None), opt.default, opt.required) for opt in workflow
    ]
    captured = []
    outputs = primary_beam.PrimaryBeamOutputs if module is primary_beam else module.SimmsOutputs
    monkeypatch.setattr(module, "runit", lambda opts: captured.append(opts) or outputs(ms="example.ms"))
    kwargs = {"mode": "tag-ms", "ms": ["example.ms"]} if module is primary_beam else {"ms": "example.ms"}
    if module is telsim:
        kwargs["telescope"] = "meerkat"
    result = _dispatch(adapter.step, adapter.func, cache=False, provenance=False, sandbox=False, **kwargs)
    assert result.success
    assert captured
    assert result.outputs.ms == kwargs["ms"] or module is primary_beam


def test_ninja_run_legacy_target(monkeypatch):
    captured = []
    monkeypatch.setattr(telsim, "runit", lambda opts: captured.append(opts))
    result = CliRunner().invoke(
        ninja_cli,
        [
            "run",
            "simms.apps.telsim:telsim",
            "--no-cache",
            "--no-provenance",
            "--no-sandbox",
            "--ms",
            "example.ms",
            "--telescope",
            "meerkat",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured and captured[0].ms == "example.ms"


def test_real_simulation_recipe_with_provenance(workflow_directory):
    from pathlib import Path

    import numpy as np
    from daskms import xds_from_ms
    from pydantic import BaseModel
    from shinobi import Recipe

    class Inputs(BaseModel):
        ms: str
        sky: str

    files = workflow_directory
    sky = files.random_named_file(suffix=".txt")
    Path(sky).write_text("#format: name ra dec stokes_i\ns0 15.0deg -31.0deg 1.0\n")
    ms = files.random_named_directory(suffix=".ms")
    recipe = Recipe(name="simulate", inputs_model=Inputs, outputs_model=skysim.SimmsOutputs)
    recipe.add_step(
        "create",
        telsim.telsim,
        ms=recipe.inputs.ms,
        telescope="meerkat",
        subarray_range=[0, 3],
        ntime=2,
        nchan=2,
        nworkers=1,
    )
    recipe.add_step(
        "predict",
        skysim.skysim,
        ms=recipe.outputs.create.ms,
        ascii_sky=recipe.inputs.sky,
        nworkers=1,
    )
    recipe.set_output("ms", recipe.outputs.predict.ms)
    result = recipe(ms=ms, sky=sky, cache=False, provenance=True)
    assert result.success
    assert result.outputs.ms == ms
    data = xds_from_ms(ms, columns=["DATA"])[0].DATA.data.compute()
    np.testing.assert_allclose(np.abs(data), 1.0, atol=1e-6)
    assert list(Path.cwd().glob(".shinobi/runs/**/*.json"))


def test_real_primary_beam_cache_and_provenance(workflow_directory):
    from pathlib import Path

    from astropy.io import fits

    files = workflow_directory
    output = files.random_named_file(suffix=".fits")
    kwargs = {
        "mode": "to-fits",
        "beam_pattern": "MKAT-EA-L-JIM-2026",
        "npix": 8,
        "nchan": 1,
        "output": output,
        "nworkers": 1,
        "cache": True,
        "cache_dir": files.random_named_directory(),
        "provenance": True,
    }
    first = primary_beam.primary_beam(**kwargs)
    second = primary_beam.primary_beam(**kwargs)
    assert first.success and second.success
    assert second.cached
    assert first.outputs == second.outputs
    assert first.outputs.output == output
    assert fits.getdata(output).shape[-2:] == (8, 8)
    assert list(Path.cwd().glob(".shinobi/runs/**/*.json"))
