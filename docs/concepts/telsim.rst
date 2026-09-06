.. _telsim:

telsim -- building a Measurement Set
=====================================

``telsim`` builds a simulated `Measurement Set (MS)
<https://casa.nrao.edu/Memos/229.html>`_ from a telescope layout: it lays out
antennas, computes ``uvw`` for the requested time/frequency grid, and writes
the standard MS subtables (``ANTENNA``, ``FIELD``, ``SPECTRAL_WINDOW``,
``POINTING``, ...) with no visibility data yet -- that's what :doc:`skysim`
fills in afterwards.

Required inputs
----------------

- **ms**: the name of the MS to create.
- **telescope**: the telescope array layout (see ``simms telsim --list`` for
  the bundled layouts under ``src/simms/telescope/layouts/``).
- **direction**: the pointing direction, written to ``FIELD.PHASE_DIR`` /
  ``POINTING.DIRECTION`` (see :doc:`ms-conventions` for why these differ).
- **starttime**, **dtime**, **ntimes**: the observation's time grid.
- **startfreq**, **dfreq**, **nchan**: the observation's frequency grid.

Usage
-----

.. code-block:: console

    $ simms telsim --telescope kat-7 --direction "J2000,0h24m20s,-30d12m33s" \
        --starttime 2024-03-14T06:15:10 --dtime 8 --ntime 100 \
        --startfreq 900MHz --dfreq 1MHz --nchan 64 obs.ms

List the available telescope layouts:

.. code-block:: console

    $ simms telsim --list

Time and frequency specification
---------------------------------

``--starttime`` is a UTC ISO timestamp; if omitted the current machine time is
used. ``--dtime`` is the integration (dump) length in seconds and ``--ntime``
is the number of dumps. ``--startha`` lets you give the starting hour angle
instead of a date, which is useful for short test tracks where only the
relative geometry matters.

``--startfreq`` and ``--dfreq`` accept units (``900MHz``, ``1MHz``) or bare
numbers in Hz. ``--nchan`` is the number of channels. Alternatively,
``--freq-range start,end,nchan`` defines the band in one argument, e.g.
``--freq-range 900MHz,1000MHz,64``.

Subarrays
---------

``--subarray-list``, ``--subarray-range`` and ``--subarray-file`` let you pick
a subset of antennas from a bundled layout:

.. code-block:: console

    # select antennas by name
    $ simms telsim --telescope meerkat --subarray-list M000,M005,M010 ... obs.ms

    # select antennas by index (start,end,step; end is inclusive when no step)
    $ simms telsim --telescope meerkat --subarray-range 0,30,5 ... obs.ms

    # read names from a file with an 'antnames' key
    $ simms telsim --telescope meerkat --subarray-file subarray.yaml ... obs.ms

The selected antennas must be a valid subarray of the layout.

Polarisation
------------

``--correlations`` sets the MS polarisation products. The default is ``XX,YY``;
a full-polarisation MS is created with ``--correlations XX,XY,YX,YY`` or
``--correlations RR,RL,LR,LL``.

Noise and sensitivity
-----------------------

``telsim`` can also create an empty ``MODEL_DATA`` column and optionally corrupt
it with thermal noise, so a noise-only MS is ready without running ``skysim``:

.. code-block:: console

    $ simms telsim --telescope kat-7 --sefd 420 --column MODEL_DATA obs.ms

The noise level can be specified in three ways:

- ``--sefd`` -- a single SEFD value used at all frequencies.
- ``--tsys-over-eta`` -- :math:`T_\mathrm{sys}/\eta`, converted to SEFD using
  the dish area from the layout.
- ``--sensitivity-file`` -- a file with per-frequency ``freq``, ``tsys``,
  ``sefd`` or ``tsys_over_eta`` columns. ``--smooth`` (``polyn`` or ``spline``)
  and ``--fit-order`` control how the per-channel SEFDs are interpolated.

Elevation limits
----------------

``--low-source-limit`` and ``--high-source-limit`` flag visibilities where the
phase-centre elevation falls outside the reliable range (in degrees). The data
rows are flagged in the MS rather than removed.

Telescope-name column
---------------------

``--telescope-name-column`` (default ``TELESCOPE_NAME``) names the ``ANTENNA``
table column that holds the per-antenna telescope/type label. ``skysim`` reads
this column to choose a primary beam per antenna; see :doc:`beams` and
:doc:`ms-conventions`.

Where to next
-------------

- :doc:`skysim` -- simulate visibilities into the MS ``telsim`` just created.
- :doc:`beams` -- how ``skysim`` uses the telescope-name column to attach beams.
- :doc:`ms-conventions` -- how simms reads/writes MS metadata, and the
  distinction between phase centre and pointing centre.
- :doc:`../cli` -- full option reference.
