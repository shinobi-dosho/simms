"""Native imports remain usable even when the workflow package cannot be imported."""

import subprocess
import sys
import textwrap


def test_native_cli_without_shinobi():
    code = """
import importlib.abc
import sys
class NoShinobi(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "shinobi" or fullname.startswith("shinobi."):
            raise ModuleNotFoundError("blocked optional dependency", name="shinobi")
sys.meta_path.insert(0, NoShinobi())
from click.testing import CliRunner
from simms.apps import telsim, skysim, primary_beam
from simms.apps.main import cli
seen = []
telsim.runit = lambda opts: seen.append(("telsim", opts))
skysim.runit = lambda opts: seen.append(("skysim", opts))
primary_beam.runit = lambda opts: seen.append(("primary-beam", opts))
runner = CliRunner()
for args in (["--help"], ["--version"], ["telsim", "--help"], ["skysim", "--help"],
             ["primary-beam", "--help"], ["telsim", "example.ms", "-tel", "meerkat"],
             ["skysim", "example.ms", "--no-do-wstacking"],
             ["primary-beam", "tag-ms", "--ms", "example.ms"],
             ["--chain", "--ms", "chain.ms", "telsim", "-tel", "meerkat", "skysim", "--column", "DATA"]):
    result = runner.invoke(cli, args)
    assert result.exit_code == 0, (args, result.output, result.exception)
assert [name for name, _ in seen] == ["telsim", "skysim", "primary-beam", "telsim", "skysim"]
assert seen[1][1].do_wstacking is False
assert seen[2][1].ms == ["example.ms"]
assert seen[-1][1].ms == "chain.ms"
assert not any(name == "shinobi" or name.startswith("shinobi.") for name in sys.modules)
try:
    telsim.telsim
except ImportError as exc:
    assert "simms[ninja]" in str(exc)
else:
    raise AssertionError("adapter loaded without shinobi")
"""
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
