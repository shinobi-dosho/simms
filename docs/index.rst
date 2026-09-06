simms
=====

**simms** simulates radio-interferometer observations end to end.

* ``telsim`` builds a `Measurement Set (MS) <https://casa.nrao.edu/Memos/229.html>`_
  from a telescope layout.
* ``skysim`` predicts model visibilities from a sky model (ASCII catalogue,
  FITS image, or WSClean component list) into an existing MS.
* ``primary-beam`` exposes the beam machinery on its own: build a beam cube,
  tag an MS with per-antenna telescope labels, or apply/correct a sky model.

.. code-block:: console

    $ simms telsim --telescope kat-7 --startfreq 900MHz --dfreq 1MHz --nchan 64 obs.ms
    $ simms skysim --ascii-sky skymodel.txt --column DATA obs.ms

Or chain both steps in one invocation:

.. code-block:: console

    $ simms --ms obs.ms --chain \
        telsim --telescope kat-7 --startfreq 900MHz --dfreq 1MHz --nchan 64 \
        skysim --ascii-sky skymodel.txt --column DATA

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Concepts

   concepts/telsim
   concepts/skysim
   concepts/beams
   concepts/schemas
   concepts/ms-conventions

.. toctree::
   :maxdepth: 2
   :caption: Using simms

   cli

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Project

   contributing

What's new
----------

See `CHANGES.md <https://github.com/shinobi-dosho/simms/blob/main/CHANGES.md>`_
for a detailed changelog. Highlights in recent releases include time/bandwidth
smearing (``--smearing``), RIME Jones corruptions (``--corruptions``), sub-sampled
smearing for FITS-image backends (``--smearing subsample``), and improved CLI
error handling.


Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
