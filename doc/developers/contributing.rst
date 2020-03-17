Contributing
============
We welcome you to contribute code and documentation to Pastas! This section
describes how you can contribute to Pastas and how the process works of
finally implementing your code into the stable version of the software.
Let's start!

Forking Pastas
--------------



Writing Code
------------


Testing Code
------------
The ensure a proper functioning of the Pastas, the software is automatically
tested using Travis when changes are made.

Document Code
-------------

When submitting a new function, method or class, docstrings are required
before the new code will be pulled into the dev branch. Documentation is
created using `Sphinxdoc <http://www.sphinx-doc.org>`_. Docstrings within
the method or class need to be written in `NumPy docformat
<https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_ to
enable automatic documentation on this website. A Jupyter Notebook explaining the
use of your new code can be added the to examples folder. This Notebook will also
be automatically converted and placed on the Examples page on this website.

Creating a pull request
-----------------------
Once you have written, tested, and documented your code you can start a pull
request on the development branch (dev) of Pastas. Pull requests can only
be submitted to the dev-branch and need to be reviewed by one of the core
developers. When you start your Pull Request, you will automatically see a
checklist to go through to check if your PR is up to standards.

.. figure:: checklist.png
   :figwidth: 500px

Pastas will run automatic code tests to ensure that the code works, is
documented and has a good code style.

.. figure:: automated_tests.png
   :figwidth: 500px
