Adding read methods
===================
In this section it is described what steps need to be taken to add a "read"
method to |Project|. The way this subpackage is structured allows for the
easy implementation of your own read methods. The |Project| read subpackage
is meant for importing timeseries from files or download.

Steps to be taken:
~~~~~~~~~~~~~~~~~~

1. Add your python script (E.g. read_new.py) that contains a function or
class to read the data (E.g. MyReader). The name of this function should
start with `read`, followed by an underscore and the name of the import
source (e.g. `ps.read_knmi`).

2. Make a new class (E.g. mydata) that inherits from DataModel class and
parses all available attributes as provided in the DataModel.

3. Add the correct import statement to the __init__.py in the read folder.
