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

Adding to or subtracting from an existing column
-------------------------------------------------

``--mode`` says how the freshly simulated visibilities reach ``--column``.
The default, ``sim``, overwrites it. ``add`` and ``subtract`` instead combine
the simulation with an existing column, which defaults to ``--column`` itself:

.. code-block:: console

    # add the sky model on top of what is already in DATA
    $ simms skysim --ascii-sky skymodel.txt --column DATA --mode add visdata.ms

    # subtract the sky model from DATA, writing the residual to CORRECTED_DATA
    $ simms skysim --ascii-sky skymodel.txt --ic DATA --column CORRECTED_DATA \
        --mode subtract visdata.ms

``--ic``/``--input-column`` names the column to combine with when it is not
``--column``; it must already exist. Every mode still needs something to
simulate -- a sky model, ``--sefd``, or both.

Thermal noise
-------------

.. code-block:: console

    $ simms skysim --ascii-sky skymodel.txt --column SIMULATED_DATA --sefd 421 visdata.ms

``--sefd`` is the System Equivalent Flux Density, in Jy.  (``telsim`` also
accepts ``--tsys-over-eta``, :math:`T_\mathrm{sys}/\eta`, and derives an SEFD
from it when building the MS; ``skysim`` takes the SEFD directly.)

Use ``--seed-noise`` to make the noise realisation reproducible at a given
chunking. (``--seed`` is the deprecated pre-3.1 name for the same option.)

Corruptions
-----------

``skysim`` can apply RIME Jones corruptions to the predicted visibilities from
a YAML specification:

.. code-block:: console

    $ simms skysim --ascii-sky skymodel.txt --column DATA \
        --corruptions corruptions.yaml --seed-gains 42 visdata.ms

The file describes an ordered list of terms and a specification for each one:

.. code-block:: yaml

    gains:
      terms: [G, B]
      spec:
        - label: G
          type: scalar
          complex: true
          axes: [time]
          period:
            time: "2min"
          amplitude: 0.1
        - label: B
          type: scalar
          complex: true
          axes: [frequency]
          period:
            frequency: "8MHz"
          amplitude: 0.05

Term labels are arbitrary strings. ``terms`` gives the multiplication order;
the per-baseline corruption is :math:`V'_{pq} = J_p(t,f) \, V_{pq}(t,f) \,
J_q(t,f)^H`.

``type`` selects the Jones form:

``scalar``
    :math:`g I` -- one gain per antenna, both polarisations identical.
``diagonal``
    :math:`\mathrm{diag}(g_x, g_y)` -- independent per-feed gains with no
    leakage.  Needs at least 2 correlations; on a 4-correlation MS the
    cross-hands mix feeds, so XY sees :math:`g_x` on antenna p and
    :math:`g_y` on antenna q.
``full``
    a dense 2x2 Jones with leakage.  Requires a 4-correlation MS.

Leaving ``type`` out gives ``diagonal``, falling back to ``scalar`` only on a
single-correlation MS, which has no second feed to give its own gain.  Leakage
is never implied -- ``full`` has to be asked for.  An explicit type the MS
cannot carry is an error rather than a silent downgrade.

``diagonal: true``/``false`` is the deprecated boolean spelling of ``scalar``
and ``full``; it warns and still works.  Note it never meant the ``diagonal``
type -- that form was previously unreachable.

``complex: true`` uses a complex sinusoid, ``false`` a real cosine.

``axes`` selects which dimensions vary (``time`` and/or ``frequency``).  ``period``
is a mapping from axis name to value; values can be raw numbers (seconds for
``time``, Hz for ``frequency``) or ``astropy`` strings such as ``"2min"`` or
``"2MHz"``.  A scalar ``period`` is accepted as shorthand when a term has a
single axis.

The random per-antenna phases (and matrices, for full terms) draw from
``--seed-gains``; omitting it gives a deterministic per-label draw. Corruptions
never touch the thermal-noise stream, which is seeded separately by
``--seed-noise``.

Phases are referenced to the earliest time and lowest frequency in the *MS*,
and gains are sized from the ``ANTENNA`` table, not from the rows a given run
happens to select.  Separate ``--field-id`` or ``--spw-id`` runs over one MS
therefore give the same antenna the same gain, and the result does not depend
on dask chunk boundaries.

Gains corrupt the sky signal only.  Receiver noise enters the signal chain
after the antenna gains, so a noisy run computes :math:`V'_{pq} = J_p V_{pq}
J_q^H + n_{pq}`: ``--sefd`` noise is added after the corruptions and is not
gain-modulated.  A noise-only run (``--sefd`` with no sky model) therefore has
nothing for ``--corruptions`` to act on, and warns.

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
