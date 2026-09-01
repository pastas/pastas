Getting started with Pastas
===========================
On this page you will find all the information to get started with Pastas.
A basic knowledge of programming in Python is assumed, but nothing more than
that. See the Examples section for some example-code.

Installing Python
-----------------
To install Pastas, a working version of Python has to be
installed on your computer. We recommend using `uv <https://docs.astral.sh/uv/>`_
to manage Python and your project dependencies. However, you are free to use
any Python installation method you prefer (i.e., the `Anaconda Distribution
<https://www.anaconda.com/products/distribution>`_).

Installing Pastas
-----------------
To install Pastas, there are several options available. The easiest is to
use the Python Package Index (`PyPI <https://pypi.python.org/pypi>`_),
where many official python packages are gathered. To get the latest version
of Pastas, open a terminal and type::

    pip install pastas

Or using uv::

    uv pip install pastas

Pastas will now be installed on your computer, including the packages
necessary for Pastas to work properly (called dependencies in Python
language). To install Pastas with the optional dependencies use::

    pip install pastas[full]

Updating Pastas
---------------
If you have already installed Pastas, it is possible to update Pastas
easily. To update, open a terminal and type::

    pip install pastas --upgrade

Dependencies
------------
Pastas depends on a number of Python packages, of which all of the necessary are
automatically installed when using the pip install manager. To summarize, the
following packages are necessary for a minimal function installation of
Pastas::

    numpy
    pandas
    scipy
    matplotlib
    numba  #(significant speed-up)

Other optional, but recommended dependencies include::

    jupyter  #(for running notebooks)
    lmfit  #(alternative solver)
    cachetools  #(for caching results)
    bokeh/plotly  #(for interactive visualisation)
    requests #(for downloading data from the pastas database)
