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

        User guide on installation and the basic concepts of Pastas.

    .. grid-item-card:: Examples
        :link: examples/index
        :link-type: doc

        Examples of Pastas usage.

    .. grid-item-card:: API Reference
        :link: api/index
        :link-type: doc

        Pastas application programming interface (API) reference.


.. grid::

    .. grid-item-card:: Development
        :link: developers/index
        :link-type: doc

        Want to contribute to Pastas? Find resources and guides for developers here.

    .. grid-item-card:: Publications
        :link: about/publications
        :link-type: doc

        Find an overview of scientific peer-reviewed studies that used Pastas.

    .. grid-item-card:: More Pastas
        :link: https://github.com/pastas/

        Find out more useful resources developed by the Pastas community on GitHub!


Quick Example
-------------

.. tab-set::

    .. tab-item:: Python

        In this example a head time series is modelled in just a few lines of Python code.

        .. code-block:: python

            # Import Python packages
            import pandas as pd
            import pastas as ps

            # Load head and meteorological observations into a pandas Series
            obs = pd.read_csv("head.csv", index_col="datetime", parse_dates=["datetime"]).squeeze()
            prec = pd.read_csv("prec.csv", index_col="datetime", parse_dates=["datetime"]).squeeze()
            evap = pd.read_csv("evap.csv", index_col="datetime", parse_dates=["datetime"]).squeeze()

            # Create and calibrate Pastas model
            ml = ps.Model(obs, name="head")
            sm = ps.RechargeModel(prec, evap, rfunc=ps.Exponential(), name="recharge")
            ml.add_stressmodel(sm)
            ml.solve()

            # Visualize the model results
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
    API Reference <api/index>
    Benchmarks <benchmarks/index>
    Development <developers/index>
    About <about/index>
    Map <map>
