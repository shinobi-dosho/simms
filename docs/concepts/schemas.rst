.. _schemas:

Schemas
=======

Sky model and cab schemas are declared as YAML files under
``src/simms/schemas/`` and used to validate/clickify CLI parameters (see
:doc:`../cli`).

What an ASCII sky model looks like
------------------------------------

The schema below lists the columns; a sky model file is a plain table of them,
one source per line, with a header line naming the columns it uses. The header
must start with ``#format:``. Only a position and a flux are required, so the
smallest valid file is two lines:

.. code-block:: text

    #format: name ra dec stokes_i
    src1 0h24m20s -30d12m33s 1.0

That is a 1 Jy point source, flat in frequency. Every other column is optional
-- adding some gives the source more structure. Here a point source and a
Gaussian, both with a spectral index:

.. code-block:: text

    #format: name ra dec stokes_i emaj emin pa cont_reffreq cont_coeff_1
    src1 0h24m20s -30d12m33s 1.0  null     null    null 1.4GHz -0.7
    src2 0h25m10s -30d08m00s 0.35 12arcsec 6arcsec 45   1.4GHz -0.9

Predict it with:

.. code-block:: console

    $ simms skysim --ascii-sky skymodel.txt --column DATA obs.ms

Three things to know when writing one:

- **Units go on the value, not in the header** -- ``0h24m20s``,
  ``-30d12m33s``, ``12arcsec``, ``1.4GHz``, or anything else ``astropy``
  parses. A bare number takes the column's ``units`` from the schema below
  (deg for ``ra``/``dec``/``emaj``, Jy for the Stokes columns, Hz for
  frequencies, seconds for the transient timings), so ``-30d12m33s`` and
  ``-30.209167`` are the same declination.
- **A catalogue is one table, so every row carries every column.** Rows that
  are not of a given type mark those cells ``null`` (or ``none``, ``nan``, in
  any case) -- as ``src1`` does for the Gaussian columns above. Such a cell
  behaves exactly as if the column were absent from the header for that row.
  With ``--ascii-delimiter ,`` an empty field does the same job.
- **A source type is claimed only when all of its columns are given** -- some
  but not all of the ``line_*`` or ``transient_*`` fields is an error, not
  silently a plain source.

See :doc:`skysim` for what each source type models. A catalogue whose columns
are named by another tool does not need converting by hand:
``--ascii-species bdsf_gaul|aegean|wsclean`` maps those names onto this
schema. For column names of your own, copy ``source_schema.yaml``, fill in the
``alias`` fields, and pass it to ``--source-schema``.

simms sky model schema
-----------------------

.. literalinclude:: ../../src/simms/schemas/source_schema.yaml
    :language: yaml


PyBDSF Gaussian-list schema mapper (GAUL)
------------------------------------------

.. literalinclude:: ../../src/simms/schemas/bdsf_gaul_source_mapper.yaml
    :language: yaml

PyBDSF source-list schema mapper (SRL)
---------------------------------------

.. literalinclude:: ../../src/simms/schemas/bdsf_srl_source_mapper.yaml
    :language: yaml

Aegean sky model schema mapper
-------------------------------

.. literalinclude:: ../../src/simms/schemas/aegean_source_mapper.yaml
    :language: yaml

WSClean sky model schema mapper
--------------------------------

.. literalinclude:: ../../src/simms/schemas/wsclean_source_mapper.yaml
    :language: yaml

Using a built-in mapper
-------------------------

To predict from a catalogue produced by PyBDSF, Aegean or WSClean without
renaming columns, pass the species name:

.. code-block:: console

    $ simms skysim --ascii-sky pybdsf_gaul.txt --ascii-species bdsf_gaul \
        --column DATA obs.ms

    $ simms skysim --ascii-sky pybdsf_srl.txt --ascii-species bdsf_srl \
        --column DATA obs.ms

    $ simms skysim --ascii-sky aegean_catalogue.txt --ascii-species aegean \
        --column DATA obs.ms

    $ simms skysim --ascii-sky wsclean_catalogue.txt --ascii-species wsclean \
        --column DATA obs.ms

PyBDSF can write either a Gaussian list (``catalog_type='gaul'``) or a source
list (``catalog_type='srl'``). ``bdsf_gaul`` maps the deconvolved Gaussian
sizes ``DC_Maj``/``DC_Min``/``DC_PA`` to simms's shape parameters; ``bdsf_srl``
maps the measured source sizes ``Maj``/``Min``/``PA``. Both map ``Total_flux``,
``RA``/``DEC``, ``Source_id`` and, when present, ``Total_Q``/``Total_U``/
``Total_V`` and ``Spec_Indx``.

The delimiter is detected automatically unless the file needs ``--ascii-delimiter ,``.

Writing a custom mapper
-------------------------

If your catalogue uses different column names, copy ``source_schema.yaml`` and
fill the ``alias`` field for each column you want to rename. For example, to
read a file whose flux and major-axis columns are named ``flux`` and ``maj``:

.. code-block:: yaml

    stokes_i:
      info: Stokes I
      alias: flux
      units: Jy
      ptype: flux
      required: true
    emaj:
      info: Gaussian major-axis FWHM
      alias: maj
      units: deg
      ptype: angle

Save that as ``my_schema.yaml`` and run:

.. code-block:: console

    $ simms skysim --ascii-sky custom_catalogue.txt --source-schema my_schema.yaml \
        --column DATA obs.ms
