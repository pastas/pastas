Getting Started with Pastas
===========================
On this page you will find all the information to get started with Pastas.
A basic knowledge of programming in Python is assumed, but nothing more than
that. See the :ref:`Examples`-section for some example-code.

Installing Python
-----------------
To install Pastas, a working version of Python 3.4 or higher has to be
installed on your computer. We recommend using the `Anaconda Distribution
<https://www.continuum.io/downloads>`_ of Python. This Python distribution
includes most of the python package dependencies and the Jupyter Notebook
software to run the notebooks. Moreover, it includes the Graphical User
Interface (GUI) Spyder to start scripting in Python. However, you are free
to install any Python distribution you want.

Installing Pastas
-----------------
To get Pastas, there are several options available. The easiest is to use
the Python Package Index (`PyPI <https://pypi.python.org/pypi>`_), where
many official python packages are gathered. To get the latest version of
Pastas, open the Anaconda Prompt, a Windows Command Prompt (also called
command window) or a Mac/Linux terminal and type::

    pip install pastas

Pastas will now be installed on your computer, including the packages
necessary for Pastas to work properly (called dependencies in Python
language).

It sometimes occurs that the automatic installation of the
dependencies does not work. A safe method to update another package if you are
using Anaconda is to install a package with the follow command line::

    conda install package

Updating Pastas
---------------
If you have already installed Pastas, it is possible to update Pastas
easily. To update, open a Windows command screen or a Mac terminal and type::

    pip install pastas --upgrade

Dependencies
------------
Pastas depends on a number of Python packages, of which all of the necessary are
automatically installed when using the pip install manager. To summarize, the
following packages are necessary for a minimal function installation of
Pastas:

.. include:: ../requirements.travis.txt
    :literal:

Other optional dependencies include::

    numba (large speed-up, highly recommended)
    lmfit (alternative solver)
    requests (downloading data from the web)

.. tip::
    Installing Numba is highly recommended when using Pastas. Not only this
    package, but also pandas makes use of Numba. Since Pastas has a large
    dependency on Pandas, installing Numba will also speed up Pastas.
