
A Basic Model
=============

In this example application it is shown how a simple time series model
can be developed to simulate groundwater levels. The recharge
(calculated as preciptation minus evaporation) is used as the
explanatory time series.

.. code:: ipython3

    # First perform the necessary imports
    import pandas as pd
    import matplotlib.pyplot as plt
    import pastas as ps
    %matplotlib inline

1. Importing the dependent time series data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this codeblock a time series of groundwater levels is imported using
the ``read_csv`` function of ``pandas``. As ``pastas`` expects a
``pandas`` ``Series`` object, the data is squeezed. To check if you have
the correct data type (a ``pandas Series`` object), you can use
``type(oseries)`` as shown below.

The following characteristics are important when importing and preparing
the observed time series: - The observed time series are stored as a
``pandas Series`` object. - The time step can be irregular.

.. code:: ipython3

    # Import groundwater time seriesm and squeeze to Series object
    gwdata = pd.read_csv('../data/head_nb1.csv', parse_dates=['date'],
                         index_col='date', squeeze=True)
    print('The data type of the oseries is: %s' % type(gwdata))
    
    # Plot the observed groundwater levels
    gwdata.plot(style='.', figsize=(10, 4))
    plt.ylabel('Head [m]');
    plt.xlabel('Time [years]');


.. parsed-literal::

    The data type of the oseries is: <class 'pandas.core.series.Series'>



.. image:: output_3_1.png


2. Import the independent time series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two explanatory series are used: the precipitation and the potential
evaporation. These need to be ``pandas Series`` objects, as for the
observed heads.

Important characteristics of these time series are: - All series are
stored as ``pandas Series`` objects. - The series may have irregular
time intervals, but then it will be converted to regular time intervals
when creating the time series model later on. - It is preferred to use
the same length units as for the observed heads.

.. code:: ipython3

    # Import observed precipitation series
    precip = pd.read_csv('../data/rain_nb1.csv', parse_dates=['date'],
                         index_col='date', squeeze=True)
    print('The data type of the precip series is: %s' % type(precip))
    
    # Import observed evaporation series
    evap = pd.read_csv('../data/evap_nb1.csv', parse_dates=['date'],
                       index_col='date', squeeze=True)
    print('The data type of the evap series is: %s' % type(evap))
    
    # Calculate the recharge to the groundwater
    recharge = precip - evap
    print('The data type of the recharge series is: %s' % type(recharge))
    
    # Plot the time series of the precipitation and evaporation
    plt.figure()
    recharge.plot(label='Recharge', figsize=(10, 4))
    plt.xlabel('Time [years]')
    plt.ylabel('Recharge (m/year)');



.. parsed-literal::

    The data type of the precip series is: <class 'pandas.core.series.Series'>
    The data type of the evap series is: <class 'pandas.core.series.Series'>
    The data type of the recharge series is: <class 'pandas.core.series.Series'>



.. image:: output_5_1.png


3. Create the time series model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this code block the actual time series model is created. First, an
instance of the ``Model`` class is created (named ``ml`` here). Second,
the different components of the time series model are created and added
to the model. The imported time series are automatically checked for
missing values and other inconsistencies. The keyword argument fillnan
can be used to determine how missing values are handled. If any
nan-values are found this will be reported by ``pastas``.

.. code:: ipython3

    # Create a model object by passing it the observed series
    ml = ps.Model(gwdata, name="GWL")
    # Add the recharge data as explanatory variable
    ts1 = ps.StressModel(recharge, ps.Gamma, name='recharge', settings="evap")
    ml.add_stressmodel(ts1)



.. parsed-literal::

    INFO: Cannot determine frequency of series head
    INFO: Inferred frequency from time series None: freq=D 


4. Solve the model
~~~~~~~~~~~~~~~~~~

The next step is to compute the optimal model parameters. The default
solver uses a non-linear least squares method for the optimization. The
python package ``scipy`` is used (info on ``scipy's`` least_squares
solver can be found
`here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>`__).
Some standard optimization statistics are reported along with the
optimized parameter values and correlations.

.. code:: ipython3

    ml.solve()


.. parsed-literal::

    
    Model Results GWL                Fit Statistics
    ============================    ============================
    nfev     28                     EVP                    91.35
    nobs     644                    NSE                     0.91
    noise    1                      Pearson R2              0.91
    tmin     1985-11-14 00:00:00    RMSE                    0.13
    tmax     2015-06-28 00:00:00    AIC                     7.15
    freq     D                      BIC                    29.49
    warmup   3650                   __                          
    solver   LeastSquares           ___                         
    
    Parameters (5 were optimized)
    ============================================================
                    optimal   stderr     initial vary
    recharge_A   758.688413   ±5.02%  215.674528    1
    recharge_n     1.053159   ±1.48%    1.000000    1
    recharge_a   136.699494   ±6.78%   10.000000    1
    constant_d    27.554115   ±0.08%   27.900078    1
    noise_alpha   63.976525  ±13.01%   14.000000    1
    
    Warnings
    ============================================================
    
            


5. Plot the results
~~~~~~~~~~~~~~~~~~~

The solution can be plotted after a solution has been obtained.

.. code:: ipython3

    ml.plot()




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x1c23b9eda0>




.. image:: output_11_1.png


6. Advanced plotting
~~~~~~~~~~~~~~~~~~~~

There are many ways to further explore the time series model. ``pastas``
has some built-in functionalities that will provide the user with a
quick overview of the model. The ``plots`` subpackage contains all the
options. One of these is the method ``plots.results`` which provides a
plot with more information.

.. code:: ipython3

    ml.plots.results(figsize=(10, 6))




.. parsed-literal::

    [<matplotlib.axes._subplots.AxesSubplot at 0x1c23cf1b00>,
     <matplotlib.axes._subplots.AxesSubplot at 0x1c23d381d0>,
     <matplotlib.axes._subplots.AxesSubplot at 0x1c2420d908>,
     <matplotlib.axes._subplots.AxesSubplot at 0x1c2423e5c0>,
     <matplotlib.axes._subplots.AxesSubplot at 0x1c24a47358>]




.. image:: output_13_1.png


7. Statistics
~~~~~~~~~~~~~

The ``stats`` subpackage includes a number of statistical functions that
may applied to the model. One of them is the ``summary`` method, which
gives a summary of the main statistics of the model.

.. code:: ipython3

    ml.stats.summary()




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
          <th>Value</th>
        </tr>
        <tr>
          <th>Statistic</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Akaike Information Criterion</th>
          <td>7.150262</td>
        </tr>
        <tr>
          <th>Average Deviation</th>
          <td>-0.000661</td>
        </tr>
        <tr>
          <th>Bayesian Information Criterion</th>
          <td>29.488755</td>
        </tr>
        <tr>
          <th>Explained variance percentage</th>
          <td>91.351929</td>
        </tr>
        <tr>
          <th>Pearson R^2</th>
          <td>0.913517</td>
        </tr>
        <tr>
          <th>Root mean squared error</th>
          <td>0.126401</td>
        </tr>
      </tbody>
    </table>
    </div>



8. Improvement: estimate evaporation factor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the previous model, the recharge was estimated as precipitation minus
potential evaporation. A better model is to estimate the actual
evaporation as a factor (called the evaporation factor here) times the
potential evaporation. First, new model is created (called ``ml2`` here
so that the original model ``ml`` does not get overwritten). Second, the
``StressModel2`` object is created, which combines the precipitation and
evaporation series and adds a parameter for the evaporation factor
``f``. The ``StressModel2`` object is added to the model, the model is
solved, and the results and statistics are plotted to the screen. Note
that the new model gives a better fit (lower root mean squared error and
higher explained variance), and that the Akiake information criterion
indicates that the addition of the additional parameter improved the
model signficantly (the Akaike criterion for model ``ml2`` is higher
than for model ``ml``).

.. code:: ipython3

    # Create a model object by passing it the observed series
    ml2 = ps.Model(gwdata)
    
    # Add the recharge data as explanatory variable
    ts1 = ps.StressModel2([precip, evap], ps.Gamma, name='rainevap', settings=("prec", "evap"))
    ml2.add_stressmodel(ts1)
    
    # Solve the model
    ml2.solve()
    
    # Plot the results
    ml2.plot()
    
    # Statistics
    ml2.stats.summary()


.. parsed-literal::

    INFO: Cannot determine frequency of series head
    INFO: Inferred frequency from time series rain: freq=D 
    INFO: Inferred frequency from time series evap: freq=D 
    
    Model Results head                Fit Statistics
    ============================    ============================
    nfev     32                     EVP                    92.86
    nobs     644                    NSE                     0.93
    noise    1                      Pearson R2              0.93
    tmin     1985-11-14 00:00:00    RMSE                    0.11
    tmax     2015-06-28 00:00:00    AIC                     9.23
    freq     D                      BIC                    36.03
    warmup   3650                   __                          
    solver   LeastSquares           ___                         
    
    Parameters (6 were optimized)
    ============================================================
                    optimal   stderr     initial vary
    rainevap_A   695.097044   ±4.15%  215.674528    1
    rainevap_n     1.019110   ±1.13%    1.000000    1
    rainevap_a   150.793525   ±0.00%   10.000000    1
    rainevap_f    -1.253062   ±4.42%   -1.000000    1
    constant_d    27.863219   ±0.23%   27.900078    1
    noise_alpha   52.470925  ±12.20%   14.000000    1
    
    Warnings
    ============================================================
    
            




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
          <th>Value</th>
        </tr>
        <tr>
          <th>Statistic</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Akaike Information Criterion</th>
          <td>9.227841</td>
        </tr>
        <tr>
          <th>Average Deviation</th>
          <td>-0.001252</td>
        </tr>
        <tr>
          <th>Bayesian Information Criterion</th>
          <td>36.034033</td>
        </tr>
        <tr>
          <th>Explained variance percentage</th>
          <td>92.864784</td>
        </tr>
        <tr>
          <th>Pearson R^2</th>
          <td>0.928639</td>
        </tr>
        <tr>
          <th>Root mean squared error</th>
          <td>0.114820</td>
        </tr>
      </tbody>
    </table>
    </div>




.. image:: output_17_2.png


Origin of the series
~~~~~~~~~~~~~~~~~~~~

-  The rainfall data is taken from rainfall station Heibloem in The
   Netherlands.
-  The evaporation data is taken from weather station Maastricht in The
   Netherlands.
-  The head data is well B58C0698, which was obtained from Dino loket
