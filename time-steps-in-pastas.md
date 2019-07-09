---
description: >-
  This page describes how Pastas deals with the frequencies of the different
  time series in a model.
---

# Time steps in Pastas

### Time steps of the dependent time series

The dependent time series \(e.g., the observed heads\) can have irregular measurement frequencies and data gaps. There are cases when you want to change the frerquency of the observed heads. This can be done by updating the TimeSeries object that contains the observed heads:

 `ml.oseries.update_settings(freq="14D")`

### Time steps of the independent time series

The independent time series \(e.g., rain or evaporation\) need to have a regular measurement frequency. When Adding a time series to a stress model, Pastas will check if the provided Pandas Series has a regular interval by trying to infer the frequency. If this fails, the user will receive a warning.

### Time step of the simulation





