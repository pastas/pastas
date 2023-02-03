Pastas Code Style
=================
This page provides information on the code style to use when writing code for Pastas.

Black formatting
----------------
To ensure high quality code that is easy to read and maintain we follow the
`Black <https://black.readthedocs.io/en/stable/index.html>`_ code formatting standard.
Please checkout the `Black Documentation <https://black.readthedocs.io/en/stable/index.html>`_
on how to format Python code this way.

Type Hints
----------
Pastas uses TypeHinting, which is used to check user-provided input options. Please provide
TypeHints when creating new methods and classes.

Docstrings
----------
Documentation is
created using `Sphinxdoc <http://www.sphinx-doc.org>`_. Docstrings within
the method or class need to be written in `NumPy docformat <https://numpydoc
.readthedocs.io/en/latest/format.html#docstring-standard>`_ to enable
automatic documentation on this website.

Optimization
------------
Much of the Pastas code is highly optimized to main high speed and productivity. Please
make sure to optimize your code in terms of performance, particularly when changing or
adding methods that are called by `ml.simulate` and `ml.solve`. It may also be
possible to use Numba for speed-ups.


Private Methods
---------------
Private methods in Pastas are identified by a leading underscore. For example:

>>> ml._get_response()

This basically means that these methods may be removed or their behaviour may
be changed without DeprecationWarning or any other form of notice.

Logger Calls
------------
When a message to the logger is provided, it is important to use the
s-string format (no f-strings) to prevent performance issues:

>>> logger.info("Message here: %s", value)

