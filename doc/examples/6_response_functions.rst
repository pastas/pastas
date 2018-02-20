
Impulse reponse functions
=========================

.. code:: ipython3

    import pastas as ps
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

Overview of the response functions available in Pastas
------------------------------------------------------

In this notebook an overview is provided of the response function that
are available in Pastas. The below table summarizes all the response
functions that are tested and supported. Others may be available but are
not yet fully supported.

+-------+-------+-------+
| Respo | No.   | Descr |
| nse   | of    | iptio |
| funct | Param | n     |
| ion   | eters |       |
+=======+=======+=======+
| Expon | 2 -   | Respo |
| entia | A, a  | nse   |
| l     |       | funct |
|       |       | ion   |
|       |       | that  |
|       |       | can   |
|       |       | be    |
|       |       | used  |
|       |       | for   |
|       |       | stres |
|       |       | ses   |
|       |       | that  |
|       |       | have  |
|       |       | an    |
|       |       | (almo |
|       |       | st)   |
|       |       | insta |
|       |       | nt    |
|       |       | effec |
|       |       | t.    |
+-------+-------+-------+
| Gamma | 3 -   | Respo |
|       | A, a, | nse   |
|       | n     | funct |
|       |       | ion   |
|       |       | commo |
|       |       | nly   |
|       |       | used  |
|       |       | for   |
|       |       | the   |
|       |       | groun |
|       |       | dwate |
|       |       | r     |
|       |       | respo |
|       |       | nse   |
|       |       | to    |
|       |       | preci |
|       |       | pitat |
|       |       | ion   |
|       |       | and   |
|       |       | evapo |
|       |       | ratio |
|       |       | n.    |
|       |       | This  |
|       |       | funct |
|       |       | ion   |
|       |       | is    |
|       |       | very  |
|       |       | versa |
|       |       | tile  |
|       |       | in    |
|       |       | its   |
|       |       | shape |
|       |       | and   |
|       |       | is    |
|       |       | often |
|       |       | used  |
|       |       | for   |
|       |       | other |
|       |       | stres |
|       |       | ses   |
|       |       | as    |
|       |       | well. |
+-------+-------+-------+
| Hantu | 3 -   | Respo |
| sh    | A,    | nse   |
|       | Rho,  | funct |
|       | Cs    | ion   |
|       |       | commo |
|       |       | nly   |
|       |       | used  |
|       |       | for   |
|       |       | groun |
|       |       | dwate |
|       |       | r     |
|       |       | abstr |
|       |       | actio |
|       |       | n     |
|       |       | wells |
+-------+-------+-------+
| Brugg | 3 -   | Respo |
| eman  | a, b, | nse   |
|       | c     | funct |
|       |       | ion   |
|       |       | commo |
|       |       | nly   |
|       |       | used  |
|       |       | for   |
|       |       | (rive |
|       |       | r)    |
|       |       | water |
|       |       | level |
|       |       | s     |
+-------+-------+-------+
| Theis | 3 -   | Respo |
|       | S, T, | nse   |
|       | r     | funct |
|       |       | ion   |
|       |       | commo |
|       |       | nly   |
|       |       | used  |
|       |       | for   |
|       |       | groun |
|       |       | dwate |
|       |       | r     |
|       |       | abstr |
|       |       | actio |
|       |       | n     |
|       |       | wells |
+-------+-------+-------+

Parameters
----------

**up** - This parameters determines whether the influence of the stress
goes up or down, hence a positive or a negative response function.

**meanstress** - This parameter is used to estimante the initial value
of the stationary effect of a stress. Hence the effect when a stress
stays at an unit level for infinite amount of time.

**cutoff** - This parameter determines for how many time steps the
response is calculated. This reduces calculation times as it reduces the
length of the array the stress is convolved with. The default value is
0.99, meaning that the response is cutoff after 99% of the effect of the
stress impuls has occurred. A minimum of length of three times the
simulation time step is applied.

.. code:: ipython3

    # Exponential function
    responses = {}
    cutoff = 0.99 # 
    meanstress = 1
    up = True
    
    Exp = ps.Exponential(up=up, meanstress=meanstress, cutoff=cutoff)
    responses["Exponential"] = Exp
    
    Gamma = ps.Gamma(up=up, meanstress=meanstress, cutoff=cutoff)
    responses["Gamma"] = Gamma
    
    Hantush = ps.Hantush(up=up, meanstress=meanstress, cutoff=cutoff)
    responses["Hantush"] = Hantush
    
    Bruggeman = ps.Bruggeman(up=up, meanstress=meanstress, cutoff=cutoff)
    responses["Bruggeman"] = Bruggeman
    
    # Theis = ps.Theis(up=up, meanstress=meanstress, cutoff=cutoff)
    # responses["Theis"] = Theis
    
    parameters = pd.DataFrame()
    
    for name, response in responses.items():
        p = response.set_parameters(name)
        parameters = parameters.append(p)
        plt.plot(response.step(p.initial), label=name)
    plt.title("Step responses for the different response function in Pastas")
    plt.legend()
    plt.show()
    parameters



.. image:: output_3_0.png




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>initial</th>
          <th>pmin</th>
          <th>pmax</th>
          <th>vary</th>
          <th>name</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Exponential_A</th>
          <td>1</td>
          <td>0</td>
          <td>100</td>
          <td>1</td>
          <td>Exponential</td>
        </tr>
        <tr>
          <th>Exponential_a</th>
          <td>10</td>
          <td>0.01</td>
          <td>5000</td>
          <td>1</td>
          <td>Exponential</td>
        </tr>
        <tr>
          <th>Gamma_A</th>
          <td>1</td>
          <td>0</td>
          <td>100</td>
          <td>1</td>
          <td>Gamma</td>
        </tr>
        <tr>
          <th>Gamma_n</th>
          <td>1</td>
          <td>0.1</td>
          <td>10</td>
          <td>1</td>
          <td>Gamma</td>
        </tr>
        <tr>
          <th>Gamma_a</th>
          <td>10</td>
          <td>0.01</td>
          <td>5000</td>
          <td>1</td>
          <td>Gamma</td>
        </tr>
        <tr>
          <th>Hantush_A</th>
          <td>1</td>
          <td>0</td>
          <td>100</td>
          <td>1</td>
          <td>Hantush</td>
        </tr>
        <tr>
          <th>Hantush_rho</th>
          <td>1</td>
          <td>0.0001</td>
          <td>10</td>
          <td>1</td>
          <td>Hantush</td>
        </tr>
        <tr>
          <th>Hantush_cS</th>
          <td>100</td>
          <td>0.001</td>
          <td>1000</td>
          <td>1</td>
          <td>Hantush</td>
        </tr>
        <tr>
          <th>Bruggeman_a</th>
          <td>1</td>
          <td>0</td>
          <td>100</td>
          <td>1</td>
          <td>Bruggeman</td>
        </tr>
        <tr>
          <th>Bruggeman_b</th>
          <td>0.1</td>
          <td>0</td>
          <td>10</td>
          <td>1</td>
          <td>Bruggeman</td>
        </tr>
        <tr>
          <th>Bruggeman_c</th>
          <td>7.38906</td>
          <td>0</td>
          <td>738.906</td>
          <td>1</td>
          <td>Bruggeman</td>
        </tr>
      </tbody>
    </table>
    </div>


