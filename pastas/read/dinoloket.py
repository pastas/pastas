"""The read module of pastas is deprecated please use hydropandas instead ->
https://hydropandas.readthedocs.io
"""

msg = (
    "The read module of pastas is deprecated please use hydropandas instead -> "
    "https://hydropandas.readthedocs.io",
)


def read_dino(*args, **kwargs):
    raise DeprecationWarning(msg)


def read_dino_level_gauge(*args, **kwargs):
    raise DeprecationWarning(msg)
