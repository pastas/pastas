Introduction
============

Pastas is an open source Python package to analyse hydro(geo)logical time
series. The objective of Pastas is twofold: to provide a scientific
framework to develop and test new methods, and to provide a reliable
ready‐to‐use software tool for groundwater practitioners. All code is
available from the `Pastas GitHub <https://github.com/pastas/pastas>`_. Want
to contribute to the project? Check out the :doc:`Developers <developers/index>` section.

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

        In this example a head time series from Kingstown (USA) is modelled in just a few lines of Python code.

        .. code-block:: python

            ml = ps.Model(obs.loc[::14], name="Kingstown")
            rm = ps.RechargeModel(rain, evap, name="recharge", rfunc=ps.Gamma())
            ml.add_stressmodel(rm)
            ml.solve(tmax="2014")
            ml.plots.results()

    .. tab-item:: Result

        .. figure:: _static/example_output.png
            :figwidth: 500px


Using Pastas? Please cite us!
-----------------------------

If you find Pastas useful and use it in your research or project, we kindly
ask you to cite the Pastas article published in Groundwater journal as follows:

- Collenteur, R.A., Bakker, M., Caljé, R., Klop, S.A., Schaars, F. (2019)
  `Pastas: open source software for the analysis of groundwater time series.
  Groundwater <https://ngwa.onlinelibrary.wiley.com/doi/full/10.1111/gwat
  .12925>`_. doi: 10.1111/gwat.12925.

.. toctree::
    :maxdepth: 2
    :hidden:

    About <about/index>
    User Guide <userguide/index>
    Examples <examples/index>
    API Docs <api/index>
    Benchmarks <benchmarks/index>
    Developers <developers/index>


