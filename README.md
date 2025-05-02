# Pastas: Analysis of Groundwater Time Series

> [!IMPORTANT]
> As of Pastas 1.5, noisemodels are not added to the Pastas models by default anymore. [Read more about this change here](https://github.com/pastas/pastas/issues/735).

![image](/doc/_static/logo_small.png)

[![image](https://img.shields.io/pypi/v/pastas.svg)](https://pypi.python.org/pypi/pastas)
[![image](https://img.shields.io/pypi/l/pastas.svg)](https://mit-license.org/)
[![image](https://img.shields.io/pypi/pyversions/pastas)](https://pypi.python.org/pypi/pastas)
[![image](https://img.shields.io/pypi/dm/pastas)](https://pypi.org/project/pastas/)
[![image](https://zenodo.org/badge/DOI/10.5281/zenodo.1465866.svg)](https://doi.org/10.5281/zenodo.1465866)

[![image](https://app.codacy.com/project/badge/Grade/952f41c453854064ba0ee1fa0a0b4434)](https://app.codacy.com/gh/pastas/pastas/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![image](https://api.codacy.com/project/badge/Coverage/952f41c453854064ba0ee1fa0a0b4434)](https://app.codacy.com/gh/pastas/pastas/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage9)
[![image](https://readthedocs.org/projects/pastas/badge/?version=stable)](https://pastas.readthedocs.io/)
[<img src="https://github.com/codespaces/badge.svg" height="20">](https://codespaces.new/pastas/pastas?quickstart=1)
[![image](https://github.com/pastas/pastas/actions/workflows/test_unit_pytest.yml/badge.svg?branch=master)](https://github.com/pastas/pastas/actions/workflows/test_unit_pytest.yml)
[![image](https://github.com/pastas/pastas/actions/workflows/test_format_lint.yml/badge.svg?branch=master)](https://github.com/pastas/pastas/actions/workflows/test_format_lint.yml)

## Pastas: what is it?

Pastas is an open source python package for processing, simulating and
analyzing groundwater time series. The object oriented structure allows
for the quick implementation of new model components. Time series models
can be created, calibrated, and analysed with just a few lines of python
code with the built-in optimization, visualisation, and statistical
analysis tools.

## Documentation & Examples

-   Documentation is provided on the dedicated website
    [pastas.dev](http://www.pastas.dev/)
-   Examples can be found on the [examples directory on the
    documentation website](https://pastas.readthedocs.io/stable/examples/)
-   A list of publications that use Pastas is available in a
    [dedicated Zotero group](https://www.zotero.org/groups/4846685/pastas/items/32FS5PTW/item-list)
-   View and edit the example notebooks of Pastas in
    [GitHub Codespaces](https://codespaces.new/pastas/pastas?quickstart=1))

## Get in Touch

-   Questions on Pastas can be asked and answered on [Github
    Discussions](https://github.com/pastas/pastas/discussions).
-   Bugs, feature requests and other improvements can be posted as
    [Github Issues](https://github.com/pastas/pastas/issues).
-   Pull requests will only be accepted on the development branch (dev)
    of this repository. Please take a look at the [developers
    section](http://pastas.readthedocs.io/) on the documentation website
    for more information on how to contribute to Pastas.

## Quick installation guide

To install Pastas, a working version of Python 3.9, 3.10, 3.11, or 3.12
has to be installed on your computer. We recommend using the [Anaconda
Distribution](https://www.continuum.io/downloads) as it includes most of
the python package dependencies and the Jupyter Notebook software to run
the notebooks. However, you are free to install any Python distribution
you want.

### Stable version

To get the latest stable version, use:

    pip install pastas

### Update

To update pastas, use:

    pip install pastas --upgrade

### Developers

To get the latest development version, use:

    pip install git+https://github.com/pastas/pastas.git@dev#egg=pastas

## Related packages

-   [Pastastore](https://github.com/pastas/pastastore) is a Python
    package for managing multiple timeseries and pastas models
-   [Metran](https://github.com/pastas/metran) is a Python package to
    perform multivariate timeseries analysis using a technique called
    dynamic factor modelling.
-   [Hydropandas](https://hydropandas.readthedocs.io/en/stable/examples/03_hydropandas_and_pastas.html)
    can be used to obtain Dutch timeseries (KNMI, Dinoloket, ..)
-   [PyEt](https://github.com/phydrus/pyet) can be used to compute
    potential evaporation from meteorological variables.

## Dependencies

Pastas depends on a number of Python packages, of which all of the
necessary are automatically installed when using the pip install
manager. To summarize, the dependencies necessary for a minimal function
installation of Pastas

-   numpy\>=1.7
-   matplotlib\>=3.1
-   pandas\>=1.1
-   scipy\>=1.8
-   numba\>=0.51

To install the most important optional dependencies (solver LmFit and
function visualisation Latexify) at the same time with Pastas use:

    pip install pastas[full]

or for the development version use:

    pip install git+https://github.com/pastas/pastas.git@dev#egg=pastas[full]

## How to Cite Pastas?

If you use Pastas in one of your studies, please cite the Pastas article
in Groundwater:

-   Collenteur, R.A., Bakker, M., Caljé, R., Klop, S.A., Schaars, F.
    (2019) [Pastas: open source software for the analysis of groundwater
    time
    series](https://ngwa.onlinelibrary.wiley.com/doi/abs/10.1111/gwat.12925).
    Groundwater. doi: 10.1111/gwat.12925.

To cite a specific version of Pastas, you can use the DOI provided for
each official release (\>0.9.7) through Zenodo. Click on the link to get
a specific version and DOI, depending on the Pastas version.

-   Collenteur, R., Bakker, M., Caljé, R. & Schaars, F. (XXXX). Pastas:
    open-source software for time series analysis in hydrology (Version
    X.X.X). Zenodo. <http://doi.org/10.5281/zenodo.1465866>
