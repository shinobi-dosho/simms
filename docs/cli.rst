Command-line interface
======================

The ``simms`` command is a click group chaining three subcommands:
``telsim``, ``skysim``, and ``primary-beam``. Global options (``--ms``,
``--log-level``, ``--chain``) precede the subcommand:

.. code-block:: console

    $ simms [--log-level LEVEL] [--ms FILE --chain] COMMAND ...

Pass ``--chain`` on the group to run ``telsim`` and ``skysim`` back to back
against one MS given once at the top level (``--ms`` is then dropped from the
subcommands):

.. code-block:: console

    $ simms --ms obs.ms --chain telsim --telescope kat-7 skysim --ascii-sky sky.txt

Options can appear on either side of the positional argument, so the following
are equivalent:

.. code-block:: console

    $ simms telsim obs.ms --telescope kat-7 --startfreq 900MHz
    $ simms telsim --telescope kat-7 --startfreq 900MHz obs.ms

Error handling
--------------

Most user errors below the CLI are reported as a single ``Error:`` line rather
than a full Python traceback. To see the traceback, either pass
``--log-level DEBUG`` or set the environment variable ``SIMMS_TRACEBACK=1``:

.. code-block:: console

    $ simms --log-level DEBUG telsim --telescope no-such-array obs.ms
    $ SIMMS_TRACEBACK=1 simms telsim --telescope no-such-array obs.ms

.. click:: simms.apps.main:cli
   :prog: simms
   :nested: full
