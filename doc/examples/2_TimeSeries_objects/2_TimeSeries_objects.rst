
.. raw:: html

   <figure>

.. raw:: html

   </figure>

The Pastas TimeSeries Class
===========================

*Developed by Raoul Collenteur* -

**WARNING: This Notebook is work in progress, please check the
development Branch at Github.**

In this Jupyter Notebook, the concept of the Pastas TimeSeries class is
explained in full detail.

Objective of the Pastas TimeSeries class:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"*To create one class that deals with all user-provided time series and
the manipulations of the series while maintaining the original series.*"

Desired Capabilities:
~~~~~~~~~~~~~~~~~~~~~

-  Validate user-provided time series
-  Extend before and after
-  Fill nan-values
-  Change frequency

   -  Upsample
   -  Downsample

-  Normalize values

Resources
~~~~~~~~~

The definition of the class can be found on Github
(https://github.com/pastas/pastas/blob/master/pastas/timeseries.py)
Documentation on the Pandas Series can be found here:
http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html

Basic concept
~~~~~~~~~~~~~

The central idea behind the TimeSeries class is to solve all data
manipulations in a single class while maintaining the original time
series. While manipulating the TimeSeries when working with your Pastas
model, the original data are to be maintained such that only the
settings and the original series can be stored.

.. code:: ipython3

    # Import some packages
    import pastas as ps
    import pandas as pd
    import matplotlib.pyplot as plt
    %matplotlib inline
    

1. Importing groundwater time series
------------------------------------

Let's first import some time series so we have some data to play around
with. We use Pandas read\_csv method and obtain a Pandas Series object,
pandas data structure to efficiently deal with 1D Time Series data. By
default, Pandas adds a wealth of functionalities to a Series object,
such as descriptive statistics (e.g. ``series.describe()``) and plotting
funtionality.

.. code:: ipython3

    gwdata = pd.read_csv('../data/B58C0698001_0.csv', skiprows=11,
                         parse_dates=['PEIL DATUM TIJD'],
                         index_col='PEIL DATUM TIJD',
                         skipinitialspace=True)
    gwdata.rename(columns={'STAND (MV)': 'h'}, inplace=True)
    gwdata.index.names = ['date']
    gwdata = gwdata.h * 0.01
    gwdata.plot(figsize=(15,4))




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1c171adcf8>




.. image:: output_4_1.png


2. Creating a Pastas TimeSeries object
--------------------------------------

The user will provide time series data when creating a model instance,
or one of the stressmodels found in stressmodels.py. Pastas expects
Pandas Series as a standard format in which time series are provided,
but will internally transform these to Pastas TimeSeries objects to add
the necessary funtionality. It is therefore also possible to provide a
TimeSeries object directly instead of a Pandas Series object.

We will now create a TimeSeries object for the groundwater level
(gwdata). When creating a TimeSeries object the time series that are
provided are validated, such that Pastas can use the provided time
series for simulation without errors. The time series are checked for:

1. Being actual Pandas Series object;
2. Making sure the indices are all TimeStamps;
3. Making sure the indices are ordered in time;
4. Dropping any nan-values before and after the first and final valid
   value;
5. Frequency of the Series is inferred, or otherwise the user-provided
   value for "freq" is applied;
6. Nan-values within the series are handled, depending on the value for
   the "fill\_nan" argument;
7. Duplicate indices are dropped from the series.

If all of the above is OK, a TimeSeries object is returned. When valid
time series are provided all of the above checks are no problem and no
settings are required. However, all too often this is not the case and
at least "fill\_nan" and "freq" are required. The first argument tells
the TimeSeries object how to handle nan-values, and the freq argument
provides the frequency of the original time series (by default, freq=D,
fill\_nan="interpolate").

.. code:: ipython3

    oseries = ps.TimeSeries(gwdata, name="Groundwater Level")
    
    # Plot the new time series and the original
    plt.figure(figsize=(10,4))
    oseries.plot(label="pastas timeseries")
    gwdata.plot(label="original")
    plt.legend()




.. parsed-literal::

    <matplotlib.legend.Legend at 0x1041f6e80>




.. image:: output_6_1.png


3. Configuring a TimeSeries object
----------------------------------

So let's see how we can configure a TimeSeries object. In the case of
the observed groundwater levels (oseries) as in the example above,
interpolating between observations might not be the preffered method to
deal with gaps in your data. In fact, the do not have to be constant for
simulation, one of the benefits of the method of impulse response
functions. The nan-values can simply be dropped. To configure a
TimeSeries object the user has three options:

1. Use a predefined configuration by providing a string to the settings
   argument
2. Manually set all or some of the settings by providing a dictonary to
   the settings argument
3. Providing the arguments as keyword arguments to the TimeSeries object
   (not recommended)

For example, when creating a TimeSeries object for the groundwater
levels consider the three following examples for setting the fill\_nan
option:

.. code:: ipython3

    # Options 1
    oseries = ps.TimeSeries(gwdata, name="Groundwater Level", settings="oseries")
    print(oseries.settings)


.. parsed-literal::

    {'freq': None, 'sample_up': None, 'sample_down': 'drop', 'fill_nan': 'drop', 'fill_before': None, 'fill_after': None, 'tmin': Timestamp('1985-11-14 00:00:00'), 'tmax': Timestamp('2015-06-28 00:00:00'), 'norm': None, 'time_offset': Timedelta('0 days 00:00:00')}
    

.. code:: ipython3

    # Option 2
    oseries = ps.TimeSeries(gwdata, name="Groundwater Level", settings=dict(fill_nan="drop"))
    print(oseries.settings)


.. parsed-literal::

    {'freq': None, 'sample_up': None, 'sample_down': None, 'fill_nan': 'drop', 'fill_before': None, 'fill_after': None, 'tmin': Timestamp('1985-11-14 00:00:00'), 'tmax': Timestamp('2015-06-28 00:00:00'), 'norm': None, 'time_offset': Timedelta('0 days 00:00:00')}
    

.. code:: ipython3

    # Options 3
    oseries = ps.TimeSeries(gwdata, name="Groundwater Level", fill_nan="drop")
    print(oseries.settings)


.. parsed-literal::

    {'freq': None, 'sample_up': None, 'sample_down': None, 'fill_nan': 'drop', 'fill_before': None, 'fill_after': None, 'tmin': Timestamp('1985-11-14 00:00:00'), 'tmax': Timestamp('2015-06-28 00:00:00'), 'norm': None, 'time_offset': Timedelta('0 days 00:00:00')}
    

Predefined settings
~~~~~~~~~~~~~~~~~~~

All of the above methods yield the same result. It is up to the user
which one is preferred.

A question that may arise with options 1, is what the possible strings
for ``settings`` are and what configuration is then used. The TimeSeries
class contains a dictionary with predefined settings that are used
often. You can ask the TimeSeries class this question:

.. code:: ipython3

    pd.DataFrame(ps.TimeSeries._predefined_settings).T




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>fill_after</th>
          <th>fill_before</th>
          <th>fill_nan</th>
          <th>sample_down</th>
          <th>sample_up</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>oseries</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>drop</td>
          <td>drop</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>prec</th>
          <td>mean</td>
          <td>mean</td>
          <td>0</td>
          <td>sum</td>
          <td>divide</td>
        </tr>
        <tr>
          <th>evap</th>
          <td>mean</td>
          <td>mean</td>
          <td>interpolate</td>
          <td>sum</td>
          <td>divide</td>
        </tr>
        <tr>
          <th>well</th>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>sum</td>
          <td>divide</td>
        </tr>
        <tr>
          <th>waterlevel</th>
          <td>mean</td>
          <td>mean</td>
          <td>interpolate</td>
          <td>mean</td>
          <td>interpolate</td>
        </tr>
        <tr>
          <th>level</th>
          <td>mean</td>
          <td>mean</td>
          <td>interpolate</td>
          <td>mean</td>
          <td>interpolate</td>
        </tr>
        <tr>
          <th>flux</th>
          <td>mean</td>
          <td>mean</td>
          <td>0</td>
          <td>mean</td>
          <td>bfill</td>
        </tr>
        <tr>
          <th>quantity</th>
          <td>mean</td>
          <td>mean</td>
          <td>0</td>
          <td>sum</td>
          <td>divide</td>
        </tr>
      </tbody>
    </table>
    </div>



4. Let's explore the possibilities
----------------------------------

As said, Pastas TimeSeries are capable of handling time series in a way
that is convenient for Pastas.

-  Changing the frequency of the time series (sample\_up, sameple\_down)
-  Extending the time series (fill\_before and fill\_after)
-  Normalizing the time series (norm \*not fully supported yet)

We will now import some precipitation series measured at a daily
frequency and show how the above methods work

.. code:: ipython3

    # Import observed precipitation series
    precip = pd.read_csv('../data/Heibloem_rain_data.dat', skiprows=4, 
                         delim_whitespace=True, parse_dates=['date'], 
                         index_col='date')
    precip = precip.precip["2012"]
    precip /= 1000.0  # Meters
    prec = ps.TimeSeries(precip, name="Precipitation", settings="prec")

.. code:: ipython3

    fig, ax = plt.subplots(2, 1, figsize=(10,8))
    prec.update_series(freq="D")
    prec.series.plot.bar(ax=ax[0])
    prec.update_series(freq="7D")
    prec.series.plot.bar(ax=ax[1])
    
    import matplotlib.dates as mdates
    ax[1].fmt_xdata = mdates.DateFormatter('%m')
    fig.autofmt_xdate()



.. image:: output_15_0.png


Wait, what?
~~~~~~~~~~~

We just changed the frequency of the TimeSeries. When reducing the
frequency, the values were summed into the new bins. Conveniently, all
pandas methods are still available and functional, such as the great
plotting functionalities of Pandas.

All this happened ``inplace``, meaning the same object just took another
shape based on the new settings. Moreover, it performed those new
settings (``freq="W"`` weekly values) on the original series. This means
that going back and forth between frequencies does not lead to any
information loss.

Why is this so important? Because when solving or simulating a model,
the Model will ask every member of the TimeSeries family to prepare
itself with the necessary settings (e.g. new freq) and perform that
operation only once. When asked for a time series, the TimeSeries object
will "be" in that new shape.

Some more action
~~~~~~~~~~~~~~~~

Let's say, we want to simulate the groundwater series for a period where
no data is available for the time series, but we need some kind of value
for the warmup period to prevent things from getting messy. The
TimeSeries object can easily extend itself, as the following example
shows.

.. code:: ipython3

    prec.update_series(tmin="2011")
    prec.plot()
    prec.settings




.. parsed-literal::

    {'freq': '7D',
     'sample_up': 'divide',
     'sample_down': 'sum',
     'fill_nan': 0.0,
     'fill_before': 'mean',
     'fill_after': 'mean',
     'tmin': Timestamp('2011-01-01 00:00:00'),
     'tmax': Timestamp('2012-12-31 00:00:00'),
     'norm': None,
     'time_offset': Timedelta('0 days 00:00:00')}




.. image:: output_17_1.png


5. Exporting the TimeSeries
---------------------------

When done, we might want to store the TimeSeries object for later use. A
``dump`` method is built-in to export the original time series to a json
format, along with its current settings and name. This way the original
data is maintained and can easily be recreated from a json file.

.. code:: ipython3

    data = prec.dump()
    print(data.keys())


.. parsed-literal::

    dict_keys(['series', 'name', 'settings', 'metadata', 'freq_original'])
    

.. code:: ipython3

    # Tadaa, we have our extended time series in weekly frequency back!
    ts = ps.TimeSeries(**data)
    ts.plot()



.. image:: output_20_0.png


