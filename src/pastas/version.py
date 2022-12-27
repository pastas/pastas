from platform import python_version

if python_version() < "3.8.0":
    __version__ = None
    with open("../../pyproject.toml") as fo:
        l = fo.readline()
        while l:
            if "version" in l:
                __version__ = l.split()[-1].strip().strip('"')
                break
            l = fo.readline()
    if __version__ is None:
        raise ValueError("No version found in pyproject.toml")
else:
    from importlib.metadata import version
    __version__ = version("pastas")
