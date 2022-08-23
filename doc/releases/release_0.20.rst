Version 0.20 (Expected 28th of February 2022)
---------------------------------------------

New Features / Enhancements
***************************

- The model fit report now includes a warnings section. This includes warnings 
  when the optimization is not successful, when parameters hit bounds or when the 
  length of the response function exceeds the length of the calibration period.
- The release of Scipy 1.8 has solved a potential issue with small parameter 
  values during optimization. This means `ps.Leastsquares` is preferred solver in 
  most situations. The WellModel will no longer warn the user to use
  `ps.LmFitSolve` if scipy>=1.8.


Deprecations
************

- The `warnbounds` argument in `ml.fit_report()` is deprecated. Replaced with 
  argument `warnings`.

Backwards incompatible API changes
**********************************

