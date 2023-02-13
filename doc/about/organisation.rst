Pastas Organisation
===================

Pastas has grown in to a full `GitHub organisation
<https://github.com/pastas>`_ with many useful tools to complement and improve
the time series analysis experience. Important packages are:

PastaStore
----------

`PastaStore <https://github.com/pastas/pastastore>`_ is a module that stores
Pastas time series and models in a database. Storing time series and models in
a database allows the user to manage time series and Pastas models on disk,
which allows the user to pick up where they left off without having to reload
everything. Additionally, PastaStore has a lot of tools to plot time series
spatially.

Metran
------

While Pastas can only do univariate time series analysis, `Metran
<https://github.com/pastas/metran>`_ can perform multivariate timeseries
analysis using a technique called dynamic factor modelling. It can be used to
describe the variation among many variables in terms of a few underlying but
unobserved variables called factors.