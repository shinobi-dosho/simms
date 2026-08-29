.. _ms-conventions:

Measurement Set conventions
=============================

simms follows a few `Measurement Set (MS)
<https://casa.nrao.edu/Memos/229.html>`_ conventions that are easy to get
wrong if you're scripting around it directly (rather than through
:doc:`telsim`/:doc:`skysim`).

Telescope metadata has a single source of truth
--------------------------------------------------

The per-antenna telescope/type label lives in the ``ANTENNA`` table column
named by ``--telescope-name-column`` (default ``TELESCOPE_NAME``). This is
what :doc:`skysim` reads to select a primary beam per antenna. It is never
inferred from other columns such as ``DISH_DIAMETER`` -- if the column is
absent, tools should fail clearly rather than guess.

Pointing centre vs. phase centre
-----------------------------------

These are two different directions, and conflating them silently mis-centres
the primary beam:

- ``FIELD.PHASE_DIR`` is the correlator's phase-tracking centre. It is
  arbitrary and can be shifted (e.g. by phase rotation) without changing
  where the antennas are actually pointing.
- ``POINTING.DIRECTION`` is the antenna pointing centre, and is what the
  primary beam is centred on.

Use :func:`simms.skymodel.beams.read_pointing_centre` to get the beam centre
rather than reading ``FIELD.PHASE_DIR`` directly. It walks
``POINTING.DIRECTION`` -> ``FIELD.REFERENCE_DIR`` -> ``FIELD.DELAY_DIR`` ->
``PHASE_DIR``, which keeps the beam on the pointing after a ``chgcentre`` or
``phaseshift``: those move ``PHASE_DIR`` only.

``POINTING`` is keyed by ``(ANTENNA_ID, TIME)``
--------------------------------------------------

The ``POINTING`` subtable has one row per antenna per timeslot -- ``nant x
ntime`` rows -- not one row per main-table row. Writing it against main-table
rows repeats each pointing ``nbl/nant`` times and, if ``ANTENNA_ID`` is left to
its default of 0, leaves every antenna but the first with no pointing record at
all. The beam centre above is then only defined for antenna 0, which happens to
work while every antenna points the same way and breaks silently as soon as they
do not.

Spectral frame must be set
-----------------------------

``SPECTRAL_WINDOW.MEAS_FREQ_REF`` must be ``5`` (TOPO). ``casacore`` defaults
this to ``0`` (REST), which leaves the spectral frame undefined and makes CASA
imaging fail with *"No MeasFrame specified for conversion of Frequency"*.
``telsim`` sets this correctly when it creates an MS.

STRING columns
----------------

``casacore`` STRING columns are numpy ``object`` dtype, and simms writes them
in one chunk (``da.from_array(values, chunks=n)``). Adding a *new* column to a
standard subtable needs an explicit descriptor, e.g.:

.. code-block:: python

    xds_to_table(..., "{ms}::ANTENNA", columns=[col], descriptor="mssubtable('ANTENNA')")
