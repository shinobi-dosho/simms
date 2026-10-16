"""Smoke test an installed wheel from outside the source checkout."""

import importlib.metadata
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import simms
from simms.apps import primary_beam, skysim, telsim

kind = sys.argv[1]

assert all(callable(module.runit) for module in (telsim, skysim, primary_beam))
checkout = Path(__file__).resolve().parents[1]
assert not Path(simms.__file__).resolve().is_relative_to(checkout), simms.__file__
if kind == "base":
    assert importlib.util.find_spec("shinobi") is None, "Base install unexpectedly provides shinobi"
    try:
        importlib.metadata.distribution("stimela-ninja")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise AssertionError("Base install unexpectedly requires stimela-ninja (check fitstoolz#30)")
else:
    importlib.metadata.distribution("stimela-ninja")

    for module, name in ((telsim, "telsim"), (skysim, "skysim"), (primary_beam, "primary_beam")):
        assert getattr(module, name).step.inputs_model
    subprocess.run(
        [str(Path(sys.executable).with_name("ninja")), "run", "simms.apps.telsim:telsim", "--help"], check=True
    )

cli = str(Path(sys.executable).with_name("simms"))
for args in (["--version"], ["--help"], ["telsim", "--help"], ["skysim", "--help"], ["primary-beam", "--help"]):
    subprocess.run([cli, *args], check=True)

with tempfile.TemporaryDirectory(prefix="simms-wheel-") as directory:
    root = Path(directory)
    ms = str(root / "example.ms")
    sky = root / "sky.txt"
    sky.write_text("#format: name ra dec stokes_i\ns0 15.0deg -31.0deg 1.0\n")
    for args in (
        ["telsim", ms, "-tel", "meerkat", "-subrange", "0,3", "-nt", "2", "-nc", "2", "--nworkers", "1"],
        ["skysim", ms, "--ascii-sky", str(sky), "--nworkers", "1"],
        ["primary-beam", "tag-ms", "--ms", ms, "--label", "MEERKAT", "--nworkers", "1"],
    ):
        subprocess.run([cli, *args], cwd=root, check=True)
