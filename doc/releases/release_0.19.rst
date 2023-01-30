Version 0.19 (Expected 1st of December 2021)
--------------------------------------------
Minor update introducing a few new features and fixes some smaller bugs.

New Features / Enhancements
***************************

- :class:`ps.ChangeModel` is added, simulating a response that changes over
  time (PR #332).
- :meth:`ps.utils.get_equidistant_series` is added, a method to get
  equidistant timeseries using nearest reindexing.
- :class:`ps.rfunc.Kraijenhoff` is added, simulating the response in a 
  domain between two drainage channels.
- :class:`ps.rch.FlexModel` now has an optional snow and interception
  bucket (PR #343).

Deprecations
************
- No methods have been deprecated in this version.

Backwards incompatible API changes
**********************************


New Example Notebooks
*********************

- An example notebook for the :class:`ps.ChangeModel` is added.
- An example notebook showcasing methods for creating equidistant timeseries is
  added.
- An example notebook for the new snow model option in the
  :class:`ps.rch.FlexModel` is added.
