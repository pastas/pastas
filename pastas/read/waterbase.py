"""The read module of pastas is deprecated please use hydropandas instead ->
https://hydropandas.readthedocs.io
"""

msg = (
    "The read module of pastas is deprecated please use hydropandas instead -> "
    "https://hydropandas.readthedocs.io",
)


def read_waterbase(*args, **kwargs):
    raise DeprecationWarning(msg)
