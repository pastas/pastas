Developers guide
================
On this page resources for developers working on |Project| are provided. One
of the main goals of |Project| is to boost research in the use of Time Series
Analysis methods in hydrology. The software is written as an Object Oriented
Program (OOP) and can therefore easily be adapted. Under **Resources** you will
find guides to how to write different classes that add functionality to |Project|.

Before you start
~~~~~~~~~~~~~~~~
Fork the project from the `projects' Github page <http://github
.com/gwtwa/gwtsa>`_ and create a local copy of |Project|. Work on your new
function or class and test it by writing your own test suite. Once you have
tested your code and want to make it available for other |Project| users, you can
submit a pull request on the development branch (dev) of the repository.

Bug reports
~~~~~~~~~~~
If you think you have found a bug in |Project|, or if you would like to suggest an
improvement or enhancement, please submit a new Issue through the Github Issue
tracker toward the upper-right corner of the Github repository. Pull requests will
only be accepted on the development branch (dev) of the repository.

Writing documentation
~~~~~~~~~~~~~~~~~~~~~
When writing new function or classes, it is strongly recommended to provide
documentation with your new code. Documentation is created using
`Sphinxdoc <http://www.sphinx-doc.org>`_. Docstrings within the method or class
need to be written in `NumPy docformat
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_ to
enable automatic documentation on this website. A Jupyter Notebook explaining the
use of your new code can be added the to examples folder. This Notebook will also
be automatically converted and placed on the Examples page on this website.

Resources
~~~~~~~~~
.. toctree::
  :maxdepth: 1
  :glob:

  developers/**