Pastas Code Style
=================
This page provides information on the code style to use when writing code for Pastas.

Formatting and Linting
----------------------
To ensure high quality code that is easy to read and maintain we follow the
`Ruff <https://docs.astral.sh/ruff/>`_ code standard. Please checkout the `Ruff
Documentation <https://docs.astral.sh/ruff/>`_ on how to format Python code
this way. Use these commands locally to autoformat with ruff::

    ruff check --fix
    ruff format


Type Hints
----------
Pastas uses Type Hinting, which is used to check user-provided input options.
Please provide Type Hints when creating new methods and classes.

Docstrings
----------
Documentation is created using `Sphinxdoc <http://www.sphinx-doc.org>`_.
Docstrings within the method or class need to be written in `NumPy docformat
<https://numpydoc .readthedocs.io/en/latest/format.html#docstring-standard>`_
to enable automatic documentation on this website.

Version Directives in Docstrings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When adding new features, changing existing behaviour, or deprecating methods,
add the appropriate Sphinx version directive to the docstring so users can
understand when the API changed. Place these directives at the end of the
relevant parameter description or at the end of the overall docstring.

``.. versionadded:: <version>``
    Use when a new function, class, method, or parameter is introduced::

        Parameters
        ----------
        modified : bool, optional
            Use the modified KGE formulation. Default is False.

            .. versionadded:: 2.0.0

``.. versionchanged:: <version>``
    Use when the behaviour or signature of an existing function or parameter
    changes::

        Parameters
        ----------
        noise : bool, optional
            Fit a noise model. Default is True.

            .. versionchanged:: 1.5.0
                The ``noise`` parameter is deprecated and will be removed in a
                future release.

``.. deprecated:: <version>``
    Use when a function, method, or parameter is deprecated and will be
    removed in a future version. Include the replacement where applicable::

        .. deprecated:: 2.0.0
            Use :func:`kge` with ``modified=True`` instead.

The version number must match the Pastas release in which the change was
introduced (e.g. ``2.0.0``). Check the
`GitHub Releases page <https://github.com/pastas/pastas/releases>`_ if you
are unsure of the version number.

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
be changed without FutureWarning or any other form of notice.

Logger Calls and Errors
-----------------------

A logger is used to provide information to the user and log info, warning, error
messages. The logger is created using the `logging <https://docs.python.org/3/library/logging.html>`_
module. The logger is created at the top of each file and can be imported in any module
using:

>>> import logging
>>> logger = logging.getLogger(__name__)

Info messages are logged using:

>>> msg = "Message here, %s"
>>> logger.info(msg, value)

When a message to the logger is provided, it is important to use the
s-string format (no f-strings) to prevent performance issues.

Warning messages are logged using:

>>>  logger.warning(msg, value)

Error messages are logged, and raised using the appropriate error using:

>>> logger.error(msg, value)
>>> raise ValueError(msg % value)

When raising an error, the error message should be provided in the logger.error method,
and the error should be raised immediately after. This is to ensure that the error
message is logged before the error is raised.
