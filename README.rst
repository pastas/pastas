Pastas: Analysis of Groundwater Time Series
===========================================

.. image:: /doc/_static/logo_small.png
   :width: 200px
   :align: left

.. image:: https://github.com/pastas/pastas/actions/workflows/ci.yml/badge.svg?branch=master
   :target: https://github.com/pastas/pastas/actions/workflows/ci.yml
.. image:: https://img.shields.io/pypi/v/pastas.svg
   :target: https://pypi.python.org/pypi/pastas
.. image:: https://img.shields.io/pypi/l/pastas.svg
   :target: https://mit-license.org/
.. image:: https://img.shields.io/pypi/pyversions/pastas
   :target: https://pypi.python.org/pypi/pastas
.. image:: https://img.shields.io/pypi/dm/pastas
   :target: https://pypi.org/project/pastas/
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1465866.svg
   :target: https://doi.org/10.5281/zenodo.1465866
.. image:: https://api.codacy.com/project/badge/Grade/952f41c453854064ba0ee1fa0a0b4434
   :target: https://www.codacy.com/gh/pastas/pastas
.. image:: https://api.codacy.com/project/badge/Coverage/952f41c453854064ba0ee1fa0a0b4434
   :target: https://www.codacy.com/gh/pastas/pastas
.. image:: https://readthedocs.org/projects/pastas/badge/?version=latest
   :target: https://pastas.readthedocs.io/en/latest/?badge=latest
.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/pastas/pastas/master?filepath=examples%2Fnotebooks%2F1_basic_model.ipynb

Pastas: what is it?
~~~~~~~~~~~~~~~~~~~
Pastas is an open source python package for processing, simulating and analyzing
groundwater time series. The object oriented structure allows for the quick
implementation of new model components. Time series models can be created,
calibrated, and analysed with just a few lines of python code with the
built-in optimization, visualisation, and statistical analysis tools.

Documentation & Examples
~~~~~~~~~~~~~~~~~~~~~~~~
- Documentation is provided on the dedicated website `pastas.dev <http://www.pastas.dev/>`_
- Examples can be found on the `examples directory on the documentation website <https://pastas.readthedocs.io/en/dev/examples/index.html>`_
- View and edit a working example notebook of a Pastas model in `MyBinder <https://mybinder.org/v2/gh/pastas/pastas/master?filepath=examples%2Fnotebooks%2F1_basic_model.ipynb>`_
- A list of publications that use Pastas is available in a `dedicated Zotero group <https://www.zotero.org/groups/4846685/pastas/items/32FS5PTW/item-list>`_

Get in Touch
~~~~~~~~~~~~
- Questions on Pastas can be asked and answered on `Github Discussions <https://github.com/pastas/pastas/discussions>`_.
- Bugs, feature requests and other improvements can be posted as `Github Issues <https://github.com/pastas/pastas/issues>`_.
- Pull requests will only be accepted on the development branch (dev) of
  this repository. Please take a look at the `developers section
  <http://pastas.readthedocs.io/>`_ on the documentation website for more
  information on how to contribute to Pastas.

Quick installation guide
~~~~~~~~~~~~~~~~~~~~~~~~
To install Pastas, a working version of Python 3.8, 3.9, 3.10, 3.11 has to be
installed on your computer. We recommend using the `Anaconda Distribution
<https://www.continuum.io/downloads>`_ as it includes most of the python
package dependencies and the Jupyter Notebook software to run the notebooks.
However, you are free to install any Python distribution you want.

Stable version
--------------
To get the latest stable version, use::

  pip install pastas

Update
------
To update pastas, use::

  pip install pastas --upgrade

Developers
----------
To get the latest development version, use::

   pip install git+https://github.com/pastas/pastas.git@dev#egg=pastas

Related packages
~~~~~~~~~~~~~~~~
- `Pastastore <https://github.com/pastas/pastastore>`_ is a Python package for managing multiple timeseries and pastas models
- `Metran <https://github.com/pastas/metran>`_ is a Python package to perform multivariate timeseries analysis using a technique called dynamic factor modelling.
- `Hydropandas <https://github.com/ArtesiaWater/hydropandas/blob/master/examples/03_hydropandas_and_pastas.ipynb>`_ can be used to obtain Dutch timeseries (KNMI, Dinoloket, ..)
- `PyEt <https://github.com/phydrus/pyet>`_ can be used to compute potential evaporation from meteorological variables.

Dependencies
~~~~~~~~~~~~
Pastas depends on a number of Python packages, of which all of the necessary
are automatically installed when using the pip install manager. To
summarize, the dependencies necessary for a minimal function installation of
Pastas

- numpy>=1.7
- matplotlib>=3.1
- pandas>=1.1
- scipy>=1.8
- numba>=0.51

To install the most important optional dependencies (solver LmFit and function visualisation Latexify) at the same time with Pastas use::

   pip install pastas[full]

or for the development version use::

   pip install git+https://github.com/pastas/pastas.git@dev#egg=pastas[full]

How to Cite Pastas?
~~~~~~~~~~~~~~~~~~~
If you use Pastas in one of your studies, please cite the Pastas article in Groundwater:

- Collenteur, R.A., Bakker, M., Caljé, R., Klop, S.A., Schaars, F. (2019) `Pastas: open source software for the analysis of groundwater time series <https://ngwa.onlinelibrary.wiley.com/doi/abs/10.1111/gwat.12925>`_. Groundwater. doi: 10.1111/gwat.12925.

To cite a specific version of Python, you can use the DOI provided for each official release (>0.9.7) through Zenodo. Click on the link to get a specific version and DOI, depending on the Pastas version.

- Collenteur, R., Bakker, M., Caljé, R. & Schaars, F. (XXXX). Pastas: open-source software for time series analysis in hydrology (Version X.X.X). Zenodo. http://doi.org/10.5281/zenodo.1465866

