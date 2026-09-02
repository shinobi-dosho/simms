.. _quickstart:

Quickstart
==========

This walkthrough builds a small `Measurement Set (MS)
<https://casa.nrao.edu/Memos/229.html>`_ from scratch and simulates
visibilities into it from an ASCII sky model.

1. Create an MS
----------------

:doc:`concepts/telsim` builds the MS -- antenna layout, time/frequency grid,
and pointing -- with no visibility data yet:

.. code-block:: console

    $ simms telsim --telescope kat-7 \
        --direction "J2000,0h24m20s,-30d12m33s" \
        --starttime 2024-03-14T06:15:10 --dtime 8 --ntime 100 \
        --startfreq 900MHz --dfreq 1MHz --nchan 64 obs.ms

2. Write a sky model
----------------------

An ASCII sky model is a catalogue of sources, one per line. A single point
source looks like:

.. code-block:: text

    #format: name ra dec stokes_i
    src1 0h24m20s -30d12m33s 1.0

3. Predict visibilities
-------------------------

:doc:`concepts/skysim` predicts model visibilities from the sky model into a
data column on the MS created above:

.. code-block:: console

    $ simms skysim --ascii-sky skymodel.txt --column DATA obs.ms

Chain both steps
------------------

``telsim`` and ``skysim`` can be chained into one invocation with ``--chain``
(see :doc:`cli`):

.. code-block:: console

    $ simms --ms obs.ms --chain \
        telsim --telescope kat-7 --startfreq 900MHz --dfreq 1MHz --nchan 64 \
        skysim --ascii-sky skymodel.txt --column DATA

Where to next
-------------

* :doc:`concepts/telsim` -- telescope layouts, time/frequency grid, pointing.
* :doc:`concepts/skysim` -- sky model schemas, FITS models, noise, column
  add/subtract, chunking.
* :doc:`concepts/beams` -- primary beams, a-terms, and the standalone
  ``primary-beam`` tool.
* :doc:`concepts/ms-conventions` -- how simms reads/writes MS metadata.
* :doc:`cli` -- full option reference.
