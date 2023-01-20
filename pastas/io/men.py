"""This module is deprecated.
"""

msg = (
    "Pastas no longer supports export to Menyanthes files and this functionality "
    "will be removed in Pastas 1.0."
)


def load(*args, **kwargs):
    raise DeprecationWarning(msg)


def dump(*args, **kwargs):
    raise DeprecationWarning(msg)
