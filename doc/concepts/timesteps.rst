Handling time
=============
Time is at the heart of time series analysis, and therefore needs to be
carefully considered when dealing with time series models. In this section
the choices of how |Project| handles all kind of timesettings are discussed, as
well as the methods that are available for changing these.

 .. Note::
     * Standard format for a date is the Pandas Timestamp.
     * Standard format for a list of dates is the Pandas DatetimeIndex.
	
TimeSeries Object
~~~~~~~~~~~~~~~~~
All the timeseries in |Project|, for observations or for StressModels, consist of (or are transformed to) Pastas TimeSeries objects.
The TimeSeries class inherits all methods from the Pandas Series class, and adds some methods to check the series for errors and transfrom the series where necessary.
	 
Timing of StressModel-data
~~~~~~~~~~~~~~~~~~~~~~~~~~
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

Setting model tmin and tmax
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Basic tasks:

1. Setting tmin and tmax for simulation.
For  simulation, the values for tmin and tmax are solely dependent on the
periods where the independent time series are available (the tseries).

get_tmin_tmax() will return the tmin and tmax that you can simulate with.

2. Setting tmin and tmax for optimization.
For optimization, the tmin and tmax depend on both the dependent (oseries)
and the independent time series (tseries). The following rules apply:
  
  1. tmin and tmax dominated by the oseries
  2. tmin and tmax dominated by the tseries
  3. applying a warmup period

3. User intervention with tmin and tmax.
When a user intervenes with the tmin and tmax, it can only be done within
the boundaries of task 1 and 2.


What values do the stored tmin and tmax have?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The values for tmin and tmax are stored in the model class. These values change
 in three seperate cases (in order of events):

1. When creating a model, tmin and tmax are set to None.
2. When added time series, tmin and tmax are set by the tseries.
3. When solving a model, tmin and tmax are set by what is possible or by the
   user.
