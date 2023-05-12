Introduction
============

Pastas is an open source Python package to analyse hydro(geo)logical time
series. The objective of Pastas is twofold: to provide a scientific framework
to develop and test new methods, and to provide a reliable ready‐to‐use
software tool for groundwater practitioners. All code is available from the
`Pastas GitHub <https://github.com/pastas/pastas>`_. Want to contribute to the
project? Check out the :doc:`Developers <developers/index>` section.

.. grid::

    .. grid-item-card:: User Guide
        :link: userguide/index
        :link-type: doc

        User guide on the basic concepts of Pastas.

    .. grid-item-card:: Examples
        :link: examples/index
        :link-type: doc

        Examples of Pastas usage.

    .. grid-item-card:: Code Reference
        :link: api/index
        :link-type: doc

        Pastas code reference.


.. grid::

    .. grid-item-card:: Contribute
        :link: developers/index
        :link-type: doc

        Want to contribute to Pastas? Find resources and guides for developers here.

    .. grid-item-card:: Publications
        :link: about/publications
        :link-type: doc

        Find an overview of scientific peer-reviewed studies that used Pastas.

    .. grid-item-card:: More Pastas
        :link: https://github.com/pastas/

        Find out more useful resources developed by the Pastas community!


Quick Example
-------------

.. tab-set::

    .. tab-item:: Python

        In this example a head time series is modelled in just a few lines of Python code.

        .. code-block:: python

            # Import python packages
            import pandas as pd
            import pastas as ps

            # Read head and stress data
            obs = pd.read_csv("head.csv", index_col=0, parse_dates=True).squeeze("columns")
            rain = pd.read_csv("rain.csv", index_col=0, parse_dates=True).squeeze("columns")
            evap = pd.read_csv("evap.csv", index_col=0, parse_dates=True).squeeze("columns")

            # Create and calibrate model
            ml = ps.Model(obs, name="head")
            sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Exponential(), name="recharge")
            ml.add_stressmodel(sm)
            ml.solve()
            ml.plots.results()

    .. tab-item:: Result

        .. figure:: _static/example_output.png
            :figwidth: 500px


Using Pastas? Please cite us!
-----------------------------

If you find Pastas useful and use it in your research or project, we kindly ask
you to cite the Pastas article published in Groundwater journal as follows:

- Collenteur, R.A., Bakker, M., Caljé, R., Klop, S.A., Schaars, F. (2019)
  `Pastas: open source software for the analysis of groundwater time series.
  Groundwater <https://ngwa.onlinelibrary.wiley.com/doi/full/10.1111/gwat
  .12925>`_. doi: 10.1111/gwat.12925.

.. toctree::
    :maxdepth: 2
    :hidden:

    User Guide <userguide/index>
    Examples <examples/index>
    API Docs <api/index>
    Benchmarks <benchmarks/index>
    Developers <developers/index>
    About <about/index>
