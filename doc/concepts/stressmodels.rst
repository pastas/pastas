===========
StressModel
===========

A StressModel transforms the stress to the contribution on the simulation.
Most StressModels consist of one or more TimeSeries and one or more Response Functions.
The most-used stress model classes are:

* StressModel (one stress and one response-function)
* StressModel2 (two stresses, one response function, and a factor)
* Constant (no stresses, just one parameter)

