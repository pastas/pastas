Pastas Code Style
=================

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

