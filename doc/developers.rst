==========
Developers
==========
Since |Project| is an open-source framework, it depends on the Open
Source Community for continuous development of the software. Any help in
maintaining the code, writing or updating documentation is more than
welcome.

Developers guide
----------------
On this page resources for developers working on |Project| are provided. One
of the main goals of |Project| is to boost research in the use of Time Series
Analysis methods in hydrology. The software is designed as an Object Oriented
Program (OOP) with maximum flexibility in mind to quickly implement new ideas.
Under **Resources** you will find guides to how to write different classes
that add functionality to |Project|.

Before you start
----------------
Fork the project from the `projects' Github page <http://github
.com/pastas/pastas>`_ and create a local copy of |Project|. Work on your new
function or class and test it by writing your own test suite. Providing
a working example of a bug fix, improved or new functionality is highly
appreciated. When submitting a new function, method or class, special care
needs to be taken on providing documentation of the code. Please read the
`writing documentation` section below.

Creating a pull request
-----------------------
Once you have tested your code and want to make it available for other
|Project| users, you can submit a pull request on the development branch
(dev) of the repository. Pull requests can only be submitted to the
dev-branch and need to be reviewed by one of the core developers.

Bug reports
-----------
If you think you have found a bug in |Project|, or if you would like to suggest an
improvement or enhancement, please submit a new Issue through the Github Issue
tracker toward the upper-right corner of the Github repository. Pull requests will
only be accepted on the development branch (dev) of the repository.

Writing documentation
---------------------
When submitting a new function, method or class, docstrings are required
before the new code will be pulled into the dev branch. Documentation is
created using `Sphinxdoc <http://www.sphinx-doc.org>`_. Docstrings within
the method or class need to be written in `NumPy docformat
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_ to
enable automatic documentation on this website. A Jupyter Notebook explaining the
use of your new code can be added the to examples folder. This Notebook will also
be automatically converted and placed on the Examples page on this website.

Resources
---------
.. toctree::
  :maxdepth: 1
  :glob:

  developers/*
