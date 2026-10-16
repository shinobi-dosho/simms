.. _installation:

Installation
============

Requirements
------------

* Python 3.11 -- 3.13.

From PyPI
---------

The stable release is available on `PyPI <https://pypi.org/project/simms>`_:

.. code-block:: console

    $ pip install "simms>=3.0"

From GitHub
-----------

To install the latest development version:

.. code-block:: console

    $ pip install git+https://github.com/shinobi-dosho/simms.git

Either way, this installs the ``simms`` command-line tool and the importable
``simms`` package.

For development
----------------

The project uses `uv <https://docs.astral.sh/uv/>`_ -- see it for
everything, rather than calling ``python``/``pytest``/``ruff`` directly:

.. code-block:: console

    $ git clone https://github.com/shinobi-dosho/simms.git
    $ cd simms
    $ uv run --group tests python -m pytest
    $ uv run --group ruff ruff check src tests

To build the documentation locally:

.. code-block:: console

    $ uv sync --group docs
    $ uv run sphinx-build -b html docs docs/_build/html
    $ open docs/_build/html/index.html

``uv sync --group docs`` installs the package in the active virtual environment
along with the Sphinx extensions listed in ``pyproject.toml``. Read the Docs
builds the same group, so adding a documentation dependency there is the only
change needed.

Optional workflow integration
-----------------------------

The native ``simms`` CLI and Python simulation code do not import Stimela Ninja.
To use the Python steps in Ninja recipes or dosho, install the ``ninja`` extra:

.. code-block:: console

    $ uv pip install 'simms[ninja]'
    $ ninja run simms.apps.telsim:telsim --help

The existing ``simms.apps.telsim:telsim``, ``simms.apps.skysim:skysim`` and
``simms.apps.primary_beam:primary_beam`` targets remain available. Accessing one
loads its workflow adapter; without Ninja, it reports how to install the extra.
Both command interfaces derive their options from the same typed functions.

FITS support remains part of the base simms installation. The required
``fitstoolz>=0.2.2`` keeps Stimela Ninja out of a base install. CI checks this
in a fresh wheel installation.

To run the optional workflow integration tests:

.. code-block:: console

    $ uv run --extra ninja --group tests python -m pytest tests/ninja_integration_tests.py
