===========
StressModel
===========

Explanatory Series
------------------
Most StressModel-classes use one or more stress-series. Each TimeStamp in the
series represents the end of the period that that record describes. For
example, the precipitation of January 1st, has the TimeStamp of January
2nd 0:00 (this can be counter-intuitive). The stress-series have to be
equidistant (at the moment, the observation-series can be non-equidistant).

The user can use Pandas resample-methods to make sure the Series satisfy this
condition, before using the Series for |Project|. The model frequency is set at
the highest frequency of all the StressModel. Other frequencies are upscaled by
using the bfill()-method. For these frequency-manipulations, the series need to
have a frequency-independent unit. For example, precipitation needs to have
the unit L/T, and not L.
