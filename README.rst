PASTAS: HYDROLOGICAL TIME SERIES ANALYSIS
=========================================

.. image:: /doc/_static/logo_small.png
   :width: 200px
   :align: left

==============  ==================================================================
Build Status    .. image:: https://travis-ci.org/pastas/pastas.svg?branch=master
                    :target: https://travis-ci.org/pastas/pastas
Pypi            .. image:: https://img.shields.io/pypi/v/pastas.svg
                    :target: https://pypi.python.org/pypi/pastas
License         .. image:: https://img.shields.io/pypi/l/pastas.svg
                    :target: https://mit-license.org/
Latest Release  .. image:: https://img.shields.io/github/release/pastas/pastas.svg
                    :target: https://github.com/pastas/pastas/releases
Code quality    .. image:: https://api.codacy.com/project/badge/Grade/0e0fad469a3c42a4a5c5d1c5fddd6bee
                    :target: https://app.codacy.com/app/raoulcollenteur/pastas?utm_source=github.com&utm_medium=referral&utm_content=pastas/pastas&utm_campaign=Badge_Grade_Dashboard    
==============  ==================================================================


Pastas: what is it?
~~~~~~~~~~~~~~~~~~~
Pastas is an open source python package for processing, simulating and analyzing 
hydrological time series (models). The object oriented stucture allows for the 
quick implementation of new model components. Time series models can be created,
calibrated, and analysed with just a few lines of python code with the built-in 
optimization, visualisation, and statistical analysis tools.

Documentation & Examples
~~~~~~~~~~~~~~~~~~~~~~~~
- Documentation is provided on a dedicated website: http://pastas.readthedocs.io/
- Examples can be found on the `examples directory on the documentation website <http://pastas.readthedocs.io/en/dev/examples.html>`_.

Get in Touch
~~~~~~~~~~~~
- Questions on Pastas can be asked and answered on `StackOverFlow <https://stackoverflow.com/questions/tagged/pastas>`_.
- Bugs, feature requests and other improvements can be posten as the `Github Issues <https://github.com/pastas/pastas/issues>`_.
- Pull requests will only be accepted on the development branch (dev) of this repository. Please take a look at the `developers section <http://pastas.readthedocs.io/>`_ on the documentation website for more information on how to develop Pastas.

Quick installation guide
~~~~~~~~~~~~~~~~~~~~~~~~
To install Pastas, a working version of Python 3.4, 3.5 or 3.6 has to be installed on 
your computer. We recommend using the `Anaconda Distribution <https://www.continuum.io/downloads>`_
with Python 3.6 as it includes most of the python package dependencies and the Jupyter
Notebook software to run the notebooks. However, you are free to install any
Python distribution you want.

Stable version
--------------
To get the latest stable version, use::

  pip install pastas
  
or directly from Github::
  
  pip install https://github.com/pastas/pastas/zipball/master

Update
------
To update pastas, use::

  pip install pastas --upgrade  
  
Developers
----------
To get the latest development version, use::

   pip install https://github.com/pastas/pastas/zipball/dev
  
Dependencies
~~~~~~~~~~~~
Pastas depends on a number of Python packages, of which all of the necessary are 
automatically installed when using the pip install manager. To summarize, the 
following pacakges are necessary for a minimal function installation of Pasta:

- numpy>=1.9
- matplotlib>=1.5
- pandas>=0.20,
- scipy>=0.15,
- statsmodels>=0.8.

Background
~~~~~~~~~~
Work on Pastas started in the spring of 2016 at the Delft University of Technology and `Artesia Water <http://www.artesia-water.nl/>`_. 

How to Cite Pastas?
~~~~~~~~~~~~~~~~~~~
`Bakker, M., Collenteur, R., Calje, F. Schaars (2018, April) Untangling groundwater head series using time series analysis and Pastas. In EGU General Assembly 2018. <https://meetingorganizer.copernicus.org/EGU2018/EGU2018-7194.pdf>`_

License
~~~~~~~
MIT

