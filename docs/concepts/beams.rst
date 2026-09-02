.. _beams:

Primary beams and a-terms
===========================

Every antenna has a direction-dependent voltage response -- the primary beam --
that attenuates sources away from where the dish points and rotates on the sky
with parallactic angle. simms can apply that response while predicting
visibilities, per antenna, without assuming the array is homogeneous.

This page covers what the options mean and when to reach for them. For the exact
flags and defaults, see :doc:`../cli`.

Where the beam is centred
---------------------------

The beam is centred on the **antenna pointing centre**, ``POINTING.DIRECTION``
-- not on ``FIELD.PHASE_DIR``, which is the correlator's phase-tracking centre
and can be shifted freely. Conflating the two silently mis-centres the beam. See
:doc:`ms-conventions` for the full rule.

``POINTING`` is keyed by ``(ANTENNA_ID, TIME)`` and carries no ``FIELD_ID``, so
``--field-id`` cannot index it directly. simms instead picks a ``POINTING`` row
recorded while the selected rows were being observed, which is what keeps the
beam on the right target in a multi-field or mosaic MS -- the first row belongs
to whichever field the MS observed first. One row serves the whole array: a
single beam centre is assumed for every antenna and the whole track.

When an MS has no usable ``POINTING.DIRECTION``, the beam centre is taken from
``FIELD.REFERENCE_DIR``, then ``FIELD.DELAY_DIR``, and only then from the phase
centre (with a warning). Phase-rotation tools -- ``chgcentre``, ``phaseshift`` --
move ``PHASE_DIR`` alone and leave the other two at the pointing, so on a
rephased MS those columns are the last honest record of where the dishes were
looking. Each candidate is validated before it is used: an absent column or a
non-finite value is skipped, since some writers leave ``REFERENCE_DIR``
unpopulated. On an MS that has not been rephased all four directions are equal
and the chain changes nothing.

Choosing a beam model
-----------------------

``--primary-beam`` takes either of two things.

**A simms beam-config YAML** maps each telescope/type label found in the
``ANTENNA`` table to a beam model. The label comes from the column named by
``--telescope-name-column`` (default ``TELESCOPE_NAME``), which is the single
source of truth for which antenna is which kind of dish:

.. code-block:: yaml

    MKAT-MA:
      jimbeam: L
    MKAT-EA:
      jimbeam: MKAT-EA-L-JIM-2026.csv

Each entry selects a provider by its key:

``jimbeam``
    A cosine-taper ("JimBeam") model: a band shorthand (``L``, ``UHF``), a
    bundled coefficient table, or a path to your own CSV. When the value omits
    the band, ``--beam-band`` supplies the default.

``fits``
    A FITS beam cube for that telescope type.

``cattery``
    A Cattery/DDFacet beam set: eight FITS files, a real/imaginary pair per
    entry of the 2x2 voltage Jones matrix. The value is either a bare prefix,
    expanded to ``<prefix>_$(corr)_$(reim).fits``, or a full DDFacet-style
    ``--Beam-FITSFile`` pattern containing the placeholders itself.

    Optional per-entry ``pol_basis`` (default ``linear``), ``l_axis`` (default
    ``-X``) and ``m_axis`` (default ``Y``) sit alongside it. Because the
    convention is declared here, the ``--beam-l-axis`` / ``--beam-m-axis``
    command-line options do not apply to YAML configs.

    .. code-block:: yaml

        MKAT-MA:
          cattery: /beams/meerkat            # -> /beams/meerkat_xx_re.fits, ...
        MKAT-EA:
          cattery: /beams/ska_$(corr)_$(reim).fits
          pol_basis: linear
          l_axis: -X
          m_axis: Y

    The recognised placeholders are ``$(corr)`` (or its synonym ``$(xy)``),
    which takes ``xx,xy,yx,yy`` for linear feeds and ``rr,rl,lr,ll`` for
    circular, and ``$(reim)`` (``re``/``im``) or ``$(realimag)``
    (``real``/``imag``). An upper-case variable name -- ``$(CORR)``,
    ``$(REIM)`` -- substitutes the upper-case value, for file sets named that
    way. ``pol_basis`` must match how the files were written: simms stores
    every beam in the feed frame internally and rotates a circular set back on
    load, so declaring the wrong basis silently mixes the feeds.

An entry with none of these keys, or a telescope label with no entry at all,
falls back to a unity (flat) beam and logs a warning -- worth checking for in
the log if a run comes out unattenuated.

The bundled ``MKAT-AA-*`` tables and the cosine-taper model itself are vendored
from `katbeam <https://github.com/ska-sa/katbeam>`_ under BSD-3-Clause; see
``src/simms/skymodel/beam_data/NOTICE``.

**A Cattery/DDFacet heterogeneous-beam JSON** (any path ending in ``.json``) is
the ``--Beam-FITSFile`` json form. It points at the same eight-file Cattery FITS
sets, but types antennas DDFacet's way -- by raw ``ANTENNA.NAME``, not by the
``TELESCOPE_NAME`` label a YAML config keys on -- so a config written for
DDFacet can be handed to simms unchanged:

.. code-block:: json

    {
      "lband": {
        "patterns": {"cmd::default": ["/beams/$(stype)_$(corr)_$(reim).fits"]},
        "define-stationtypes": {"cmd::default": "meerkat", "~SKA[0-9]{3}": "ska"}
      }
    }

Each top-level block (the name ``lband`` is arbitrary) holds a
``define-stationtypes`` mapping of antenna name to station-type label and a
``patterns`` mapping whose values are ``$(stype)``-templated file patterns.
Blocks are chained: rules accumulate in file order and the first rule for a
given name wins. An antenna is matched by exact name first, then by any key
prefixed with ``~`` (a regex over antenna names), and finally by the
``cmd::default`` fallback; an antenna that matches nothing is an error rather
than a unity beam.

Only a single distinct file pattern is supported. DDFacet lets a station type
list several patterns covering different frequency ranges and picks one by
proximity to the MS band; simms rejects such a config with a clear error
instead of guessing which to use.

Because the FITS cubes carry their own axis conventions, ``--beam-l-axis`` and
``--beam-m-axis`` must be set to the same values you would pass DDFacet's
``--Beam-FITSLAxis`` / ``--Beam-FITSMAxis``. These two options apply only to
the ``.json`` form; a YAML config declares the convention per entry instead.
The polarisation basis is not an option here -- it is read from the MS's
``POLARIZATION.CORR_TYPE``, so the file set must be written in the basis the MS
correlates in.

Either form is passed the same way:

.. code-block:: console

    $ simms skysim --ascii-sky skymodel.txt --primary-beam beams.yaml \
        --column DATA obs.ms

    $ simms skysim --ascii-sky skymodel.txt --primary-beam beams.json \
        --beam-l-axis -X --beam-m-axis Y --column DATA obs.ms

A beam is only applied when there is a sky model to apply it to: on a
noise-only run ``--primary-beam`` is ignored with a warning.

Heterogeneous arrays are the normal case here: an MS whose ``ANTENNA`` table
mixes ``MKAT-MA`` and ``MKAT-EA`` dishes gets a different beam per antenna, and
each baseline sees the product of its two antennas' responses rather than a
single array-wide beam.

Diagonal or full Jones
------------------------

``--beam-jones diagonal`` applies a per-feed voltage response: each feed is
attenuated independently and the cross-hands carry no beam leakage.

``--beam-jones full`` applies the full 2x2 E-Jones, so off-diagonal terms
(instrumental polarisation leakage from the beam) are included. Use it when the
leakage matters to what you are simulating -- polarisation calibration or
wide-field polarimetry -- and when your beam model actually provides the
off-diagonal elements. For Cattery/DDFacet beams, a circular-correlation MS is
supported in this mode; the default diagonal beam is per-feed and so requires
linear correlations.

Beams in the FITS-image path
------------------------------

Applying a beam to a component list is straightforward: each component sits at
one direction, so it gets one gain. A FITS *image* sky model has no such
shortcut, and ``--fits-beam-mode`` picks between two ways of handling it.

``--fits-beam-mode aterm`` (the exact treatment) applies per-antenna a-terms in
the image domain, interpolated in time and frequency and aware of array
heterogeneity. It makes no spatial approximation: the beam varies across the
image as it really does, and each baseline gets its own pair of antenna
responses.

``--fits-beam-mode average`` multiplies the sky by a single parallactic-angle
averaged power beam. This is the legacy approximation -- one beam for the whole
array, averaged over the track. It is cheaper, and adequate when the beam is
nearly constant over your field and observation, but it cannot represent a
heterogeneous array or the rotation of an asymmetric beam.

Controlling a-term cost
-------------------------

The exact a-term path samples the beam on grids that you can trade against
accuracy:

``--aterm-freq-tol`` bounds the error of interpolating the a-term linearly in
frequency between knot channels, in voltage-beam units where the beam peak is
about 1. Smaller values insert more frequency knots and cost more; ``0`` or
negative samples the beam at every channel and does no interpolation at all.

``--beam-pa-step`` sets the spacing, in degrees, of the parallactic-angle grid
the beam is sampled on. A long track through transit rotates the beam quickly
near zenith, so a coarse step there is the usual source of error.

``--beam-grid-max-gib`` is a hard ceiling on how much of the sampled beam grid
is held in memory for the whole run. It is a guard rail, not a tuning knob: hit
it and you need a coarser grid, not a bigger machine.

The standalone ``primary-beam`` tool
--------------------------------------

``simms primary-beam`` exposes the beam machinery without simulating any
visibilities. It has four modes:

``to-fits``
    Evaluate a beam model onto a FITS cube. ``--fits-format simms`` writes
    simms's own single-file 4-plane HH/VV cube; ``--fits-format cattery`` writes
    the Cattery/DDFacet 8-file per-Jones-element schema that ``--Beam-Model
    FITS`` expects, in which case ``--pol-basis`` must match the target MS's feed
    basis (``linear`` for xx/xy/yx/yy, ``circular`` for rr/rl/lr/ll).

``tag-ms``
    Write the per-antenna telescope/type labels into the ``ANTENNA`` table
    column named by ``--telescope-name-column``, so an MS from elsewhere can be
    given the metadata skysim needs to pick a beam per antenna.

``apply`` / ``correct``
    Multiply a sky model by the beam, or divide it out, writing a new sky model
    rather than touching visibilities. Takes ``--fits-sky`` or ``--ascii-sky``.

Because ``to-fits`` and ``tag-ms`` write ordinary MS metadata and FITS files,
they compose with tools outside simms -- you can build a beam here and hand it
to DDFacet, or tag an MS that another simulator produced.
