Add Tseries classes
===================
In this section it is described how a new tseries object can be added to
|Project|. The classes are defined in `tseries.py`, but this is not strictly
necessary for your own project. All that is necessary is to write a class that
can be imported and used by |Project|. Before you start, it is a good idea to
study the Tseries and the TseriesBase classes in `tseries.py` to get a good idea of
the general form of a tseries class.

In general, the following steps are necessary:

1. Write a new tseries class according to the below requirements.
2. Add the tseries class to the imports in __init__.py.
3. Test the class in a time series model.

Minimum requirements of a tseries class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
List of attributes:

- name: Name of the time series component [str].
- metadata: dictionary containing additional information [dict].
- xy: location of the stress in lat-lon format [tuple].
- nparam : number of parameters [int].

List of class methods:

- __init__: setting the above mentioned attributes.
- set_init_parameters: used to set the initial parameters.
- simulate: returns a pd.Series object of the simulate time series.

Frequently asked questions
~~~~~~~~~~~~~~~~~~~~~~~~~~
**Q: Why does every simulate method have to following line?**

>>> if p is None:
>>>    p = np.array(self.parameters.value)

A: These two lines of code are necessary to be able to simulate the time series when
 no parameters are provided. After optimization the calibrated parameters are
 returned to the tseries object and can be used for simulation.

**Q: Why does every tseries class have a set_init_parameters method?**

A: This way the initial parameter can be set by the user after optimization. In
the future, this will support the keyword argument in the Model class solve method.

>>> ml.solve(initialize=True)


Special classes
~~~~~~~~~~~~~~~
The `NoiseModel` class is considered as a special class and does not comply with
the above descriptions. It is recommended to closely studied the NoiseModel class
 when adding a new noise model.
