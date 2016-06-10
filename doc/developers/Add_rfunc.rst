Add Response functions
======================
This section explains how a new response function can be implemented. All
response functions are defined as classes in the `rfunc.py` file. These classes
are imported in the __init__.py file in the |Project| folder to make them
available in the program. Adding a new response is as simple as:

1. Writing a new class in the rfunc.py file
2. Adding the import statements in the __init__.py file.


Minimum requirements for a rfunc class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
List of Attributes:

- nparam: [int] that defines the number of parameters that is needed for this
  function.
- cutoff: [float between 0 and 1] that defines the length of the response time that
  is used for simulation. By default, this value should be 0.99, meaning that
  after 99% of the response has passed, the function is cut off.

List of methods:

- set_parameters: used by the tseries class to set the initial parameters.
- block: calculate the block response (typically by using the step response).
- step: caluclate the step response .

Frequently asked questions
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Q: Why does every class have both a step and a block response function?**
A: Depending on the user, one might be interested in the block or the step
response function for model investigation.

**Q: What is the standard scaling of the response function?**
A: The response function need to be scaled such that the scaling parameter can
have some physical interpretation for the step response function. E.g. if the
groundwater recharge is continuously 1 mm/d, the groundwater level would rise 500
 mm before it reaches a steady state again.


Example Response function class:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    class Exponential:
        __doc__ = """
        Exponential response function with 2 parameters: A and a.

        .. math:: step(t) = A * (1 - exp(-t / a))

        %(doc)s
        """ % {'doc': _class_doc}

        def __init__(self):
            self.nparam = 2
            self.cutoff = 0.99

        def set_parameters(self, name):
            parameters = pd.DataFrame(columns=['value', 'pmin', 'pmax', 'vary'])
            parameters.loc[name + '_A'] = (500.0, 0.0, 5000.0, 1)
            parameters.loc[name + '_a'] = (100.0, 1.0, 5000.0, 1)
            return parameters

        def step(self, p):
            self.tmax = -np.log(1.0 / p[1]) * p[1]
            t = np.arange(1.0, self.tmax)
            s = p[0] * (1.0 - np.exp(-t / p[1]))
            return s

        def block(self, p):
            s = self.step(p)
            return s[1:] - s[:-1]