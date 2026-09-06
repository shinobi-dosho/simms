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

List the bundled telescope layouts with:

.. code-block:: console

    $ simms telsim --list

2. Write a sky model
----------------------

An ASCII sky model is a catalogue of sources, one per line. A single point
source looks like:

.. code-block:: text

    #format: name ra dec stokes_i
    src1 0h24m20s -30d12m33s 1.0

A slightly richer catalogue with a spectral index and one Gaussian is:

.. code-block:: text

    #format: name ra dec stokes_i emaj emin pa cont_reffreq cont_coeff_1
    src1 0h24m20s -30d12m33s 1.0  null     null    null 1.4GHz -0.7
    src2 0h25m10s -30d08m00s 0.35 12arcsec 6arcsec 45   1.4GHz -0.9

3. Predict visibilities
-------------------------

:doc:`concepts/skysim` predicts model visibilities from the sky model into a
data column on the MS created above:

.. code-block:: console

    $ simms skysim --ascii-sky skymodel.txt --column DATA obs.ms

Chain both steps
------------------

``telsim`` and ``skysim`` can be chained into one invocation with ``--chain``
(see :doc:`cli`). The MS is given once at the top level and is passed through
to both subcommands:

.. code-block:: console

    $ simms --ms obs.ms --chain \
        telsim --telescope kat-7 --startfreq 900MHz --dfreq 1MHz --nchan 64 \
        skysim --ascii-sky skymodel.txt --column DATA

The order of options around the positional MS argument is flexible; the
following is equivalent:

.. code-block:: console

    $ simms telsim obs.ms --telescope kat-7 --startfreq 900MHz --dfreq 1MHz --nchan 64

4. Add noise
------------

``--sefd`` adds thermal noise after the sky model has been predicted, so the
noise is not gain-modulated:

.. code-block:: console

    $ simms skysim --ascii-sky skymodel.txt --column DATA --sefd 420 obs.ms

A reproducible noise realisation needs ``--seed-noise`` and the same chunking
(``--row-chunks`` / ``--nworkers``) across runs.

5. Add corruptions
------------------

:doc:`concepts/skysim` can apply RIME Jones corruptions from a YAML file before
adding noise:

.. code-block:: console

    $ simms skysim --ascii-sky skymodel.txt --column DATA \
        --corruptions corruptions.yaml --seed-gains 42 obs.ms

6. Use a subarray
-----------------

:doc:`concepts/telsim` supports selecting a subset of antennas by name, by
index range, or from a file:

.. code-block:: console

    $ simms telsim --telescope meerkat --subarray-list M000,M005,M010 \
        --startfreq 900MHz --dfreq 1MHz --nchan 64 obs.ms

Where to next
-------------

* :doc:`concepts/telsim` -- telescope layouts, time/frequency grid, pointing,
  subarrays, sensitivity.
* :doc:`concepts/skysim` -- sky model schemas, FITS models, noise, column
  add/subtract, chunking, smearing, corruptions.
* :doc:`concepts/beams` -- primary beams, a-terms, and the standalone
  ``primary-beam`` tool.
* :doc:`concepts/ms-conventions` -- how simms reads/writes MS metadata.
* :doc:`cli` -- full option reference.
