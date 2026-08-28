.. _skysim:

skysim -- predicting visibilities
===================================

``skysim`` predicts model visibilities from a sky model (ASCII catalogue,
FITS image(s), or a WSClean component list) into an existing MS, writing them
to a chosen data column. It can also add or subtract an existing column
in-place instead of simulating from a sky model.

Basic usage
-----------

.. code-block:: console

    $ simms skysim --ascii-sky skymodel.txt --column DATA visdata.ms

- ``--ascii-sky skymodel.txt`` -- a catalogue of parametrised sources.
- ``--column DATA`` -- the column where simulated visibilities are written.
- ``visdata.ms`` -- the target MS, which must already exist (from an
  observation, or created with :doc:`telsim`).

ASCII sky model schema
-----------------------

- **Point sources** need only RA, Dec, and intensity (``stokes_i``).
- **Extended sources** are 2D Gaussians, parametrised by FWHM major/minor axes
  (``emaj``/``emin``) and position angle (``pa``); double-horn profiles are
  not supported.
- **Spectral line sources** need the line centre and width (``line_width``,
  an observed-frame FWHM). The centre is either the observed peak frequency
  (``line_peak``), or a rest frequency (``line_restfreq``) optionally shifted
  by a redshift (``line_redshift``, default 0). ``line_peak`` wins when both
  are given.
- **Continuum sources** need a reference frequency (``cont_reffreq``) and at
  least one power-law coefficient (``cont_coeff_1`` = spectral index,
  ``cont_coeff_2`` = curvature, ...).
- **Transient sources** are modulated by a logistic transit lightcurve:
  ``transient_start`` (from the start of the observation), ``transient_period``
  (the whole event, ingress and egress included), ``transient_ingress`` and
  ``transient_absorb`` (peak fractional flux drop). The time fields default to
  seconds; write ``30min`` for anything else. The lightcurve only scales the
  spectrum, so a source may be a spectral line *and* a transient.

A type is claimed only when **all** of its fields are given -- a source with
some but not all of the ``line_*`` or ``transient_*`` fields is an error, not
silently a plain source.

Mixing source types in one catalogue
....................................

A catalogue is a single table, so a file holding more than one source type
carries every type's columns and the rows that are not of a given type mark
those columns unset with ``null`` (or an empty field, with
``--ascii-delimiter ,``):

.. code-block:: text

    #format: name ra dec stokes_i line_peak line_width transient_start transient_period transient_ingress transient_absorb
    aline  0h24m20s -30d12m33s 1.0 1.42GHz 10MHz null null null null
    atrans 0h25m00s -30d10m00s 2.0 null    null  100  500  50   0.5
    both   0h23m40s -30d14m00s 3.0 1.30GHz 5MHz  200  400  40   0.2
    plain  0h24m00s -30d13m00s 4.0 null    null  null null null null

An unset column behaves exactly as if it were absent from the header for that
row. ``null``, ``none``, ``nan`` and an empty field are all accepted, in any
case.

See :doc:`schemas` for the full schema, and use ``--ascii-species`` to select
a non-default catalogue mapping (e.g. ``bdsf_gaul`` for a PyBDSF catalogue).

FITS sky models
----------------

.. code-block:: console

    $ simms skysim --fits-sky skymodel.fits --column DATA visdata.ms

Provide separate FITS files per Stokes when simulating polarised sources.
Tune the prediction with ``--pixel-tol`` (minimum pixel brightness considered,
default ``1e-7``), ``--fft-precision`` (``single``/``double``), and
``--no-do-wstacking`` to disable w-stacking.

Adding or subtracting an existing column
------------------------------------------

Once visibilities are simulated into one column, add or subtract them
against another:

.. code-block:: console

    $ simms skysim --ic DATA --column MODEL_DATA --mode add visdata.ms
    $ simms skysim --ic DATA --column MODEL_DATA --mode subtract visdata.ms

``--ic``/``--input-column`` is the source column; ``--column`` is where the
result is written; ``--mode`` defaults to ``simulate``.

Thermal noise
-------------

.. code-block:: console

    $ simms skysim --ascii-sky skymodel.txt --column SIMULATED_DATA --sefd 421 visdata.ms

Provide either ``--sefd`` (System Equivalent Flux Density, in Jy) or
``--tsys-over-eta`` (:math:`T_\mathrm{sys}/\eta`).

Chunking large MSs
-------------------

.. code-block:: console

    $ simms skysim --ascii-sky skymodel.txt --column SIMULATED_DATA --row-chunks 5000 largevis.ms

``--row-chunks`` controls the row-wise task/memory granularity (default
``10000``).

Where to next
-------------

- :doc:`telsim` -- create the target MS.
- :doc:`ms-conventions` -- how the primary beam centre is read from
  ``POINTING.DIRECTION``.
- :doc:`schemas` -- full sky model and catalogue-mapper schemas.
- :doc:`../cli` -- full option reference, including all abbreviations.
