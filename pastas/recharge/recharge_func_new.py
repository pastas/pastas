"""recharge_func module

Author: R.A. Collenteur

Contains the classes for the different models that are available to calculate
the recharge from evaporation and precipitation data.

Each Recharge class contains at least the following:

Attributes
----------
nparam: int
    Number of parameters needed for this model.

Functions
---------
set_parameters(self, name)
    A function that returns a Pandas DataFrame of the parameters of the
    recharge function. Columns of the dataframe need to be ['value', 'pmin',
    'pmax', 'vary']. Rows of the DataFrame have names of the parameters. Input
    name is used as a prefix. This function is called by a Tseries object.
simulate(self, evap, prec, p=None)
    A function that returns an array of the simulated recharge series.

"""

import pandas as pd
import numpy as np


class Percolation:
    """
    Percolation flow recharge model

    Other water balance for the root zone s calculated as:

    dS/dt = Pe - Kp * (Sr / Srmax)**Gamma - Epu * min(1, Sr / (0.5 * Srmax))

    """

    def __init__(self):
        self.nparam = 4
        self.dt = 1
        self.solver = 0

    def get_init_parameters(self, name):
        parameters = pd.DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        parameters.loc[name + '_Srmax'] = (0.26, 0.0, np.nan, 1, name)
        parameters.loc[name + '_Kp'] = (2.0e-2, 0.0, np.nan, 1, name)
        parameters.loc[name + '_Gamma'] = (2.0, 0.0, np.nan, 1, name)
        parameters.loc[name + '_Imax'] = (1.e-3, 0.0, np.nan, 0, name)
        return parameters

    def simulate(self, prec, evap, p=None):
        t = np.arange(len(prec))
        recharge = self.perc(t, prec.values, evap.values, p[0], p[1], p[2],
                             p[3], self.dt, self.solver)[0]
        return recharge

    @numba.jit
    def perc(self, t, P, E, Srmax=0.1, Kp=0.03, Gamma=2.0, Imax=0.001, dt=1.0,
             solver=0):
        n = int(len(t) / dt)
        error = 1.0e-5

        # Create an empty array to store the soil state in
        S = np.zeros(n)
        S[0] = 0.5 * Srmax  # Set the initial system state
        Si = np.zeros(n)
        Si[0] = 0.0
        Pe = np.zeros(n)
        Pe[0] = 0.0
        Ei = np.zeros(n)
        Ei[0] = 0.0
        Epu = np.zeros(n)
        Epu[0] = 0.0
        Ea = np.zeros(n)
        Ea[0] = 0.0

        for t in range(n - 1):
            # Fill interception bucket with new rain
            Si[t + 1] = Si[t] + P[t + 1]
            # Calculate effective precipitation
            Pe[t + 1] = np.max([0.0, Si[t + 1] - Imax])
            Si[t + 1] = Si[t + 1] - Pe[t + 1]
            # Evaporation from interception
            Ei[t + 1] = np.min([Si[t + 1], E[t + 1]])
            # Update interception state
            Si[t + 1] = Si[t + 1] - Ei[t + 1]
            # Update potential evapotranspiration
            Epu[t + 1] = E[t + 1] - Ei[t + 1]
            Last_S = S[t]
            iteration = 0
            bisection = 1

            # Use explicit Euler scheme to find an initial estimate for the newton raphson-method
            S[t + 1] = np.max(
                [0.0, S[t] + dt * (Pe[t] - Kp * (S[t] / Srmax) ** Gamma -
                                   Epu[t] * np.min([1.0, (S[t] / (0.5 *
                                                                  Srmax))]))])

            if solver == 1:
                # Start the while loop for the Newton-Raphson iteration
                while abs(Last_S - S[t + 1]) > error:
                    if iteration > 100:
                        break  # Check if the number of iterations is not too high
                    iteration += 1
                    Last_S = S[t + 1]

                    g = Last_S - S[t] - dt * (
                            Pe[t] - Kp * (Last_S / Srmax) ** Gamma -
                            Epu[t] * np.min([1, (Last_S / (0.5 *
                                                           Srmax))]))
                    # Derivative depends on the state of the system
                    if Last_S > (0.5 * Srmax):
                        g_derivative = 1.0 - dt * (
                                -Gamma * Kp * (Last_S / Srmax) ** (Gamma - 1))
                    else:
                        g_derivative = 1.0 - dt * (
                                -Gamma * Kp * (Last_S / Srmax) ** (Gamma - 1) -
                                Epu[
                                    t] * (
                                        0.5 * Srmax))

                    # Check if there is no zero-division error
                    if np.isnan(g / g_derivative):
                        bisection = 0
                        break
                    # if there is no zero-division error
                    else:  # use Newton-Raphson
                        S[t + 1] = Last_S - g / g_derivative

                if bisection == 0:
                    iteration = 0
                    a = S[t]
                    b = S[t + 1]
                    c = a + b / 2.0

                    while ((b - a) / 2.0) > error:
                        if iteration > 100:
                            print('iteration in bisection method exceeded 100',
                                  iteration)
                            break
                        iteration += 1  # increase the number of iterations by 1

                        if (c - S[t] - dt * (
                                Pe[t] - Kp * (c / Srmax) ** Gamma - Epu[t]
                                * np.min([1, (c / (0.5 * Srmax))]))) == 0.0:
                            return c  # Return the current value if it is correct
                        elif (a - S[t] - dt * (
                                Pe[t] - Kp * (a / Srmax) ** Gamma - Epu[
                            t] * np.min([1.0, (a / (0.5 * Srmax))]))) * (
                                c - S[t] - dt
                                * (Pe[t] - Kp * (c / Srmax) ** Gamma - Epu[
                            t] * np.min(
                            [
                                1.0, (c / (0.5 * Srmax))]))) > 0.0:
                            b = c
                        else:
                            a = c

                        c = a + b / 2.0

                    S[t + 1] = c

                assert ~np.isnan(
                    S[t + 1]), 'NaN-value calculated for soil state'

            # Make sure the solution is larger then 0.0 and smaller than Srmax
            S[t + 1] = np.min([Srmax, np.max([0.0, S[t + 1]])])
            Ea[t + 1] = Epu[t + 1] * np.min([1, (S[t + 1] / (0.5 * Srmax))])

        R = np.append(0.0, Kp * dt * 0.5 * (
                (S[:-1] ** Gamma + S[1:] ** Gamma) / (Srmax ** Gamma)))

        return R, S, Ea, Ei


class HBV(RechargeBase):
    def __init__(self):
        RechargeBase.__init__(self)
        self.temp = True
        self.nparam = 11

    def get_init_parameters(self, name="recharge"):
        parameters = pd.DataFrame(
            columns=['initial', 'pmin', 'pmax', 'vary', 'name'])
        parameters.loc[name + '_Beta'] = (1, 1, 6, 1, name)
        parameters.loc[name + '_CET'] = (0.0, 0, 0.3, 0, name)
        parameters.loc[name + '_FC'] = (250, 50, 500, 0, name)
        parameters.loc[name + '_LP'] = (0.7, 0.3, 1, 1, name)
        parameters.loc[name + '_PERC'] = (1.5, 0, 3, 1, name)
        parameters.loc[name + '_PCORR'] = (1, 0.5, 2, 0, name)
        parameters.loc[name + '_TT'] = (0, -1.5, 2.5, 0, name)
        parameters.loc[name + '_CFMAX'] = (5.0, 1, 10, 0, name)
        parameters.loc[name + '_SFCF'] = (0.7, 0.4, 1, 0, name)
        parameters.loc[name + '_CFR'] = (0.05, 0, 0.1, 0, name)
        parameters.loc[name + '_CWH'] = (0.1, 0, 0.2, 0, name)
        return parameters

    def simulate(self, prec, evap, temp, p):
        """
        Implementation of HBV model (Bergstrom, 1986)
        Input:
        1. data
        pandas dataframe with columns 'Temp', 'Prec', 'Evap'
        assosiated with correspondent daily time series derived
        from WFDEI meteorological forcing dataset.
        'Temp' - Celsius degrees
        'Prec' - mm/day
        'Evap' - mm/day
        2. params
        List of 16 HBV model parameters
        [parBETA, parCET,  parFC,    parLP,
        parPERC,  parPCORR, parTT,
        parCFMAX, parSFCF, parCFR,   parCWH]
        init_params = [ 1.0,   0.15,    250,   0.7,   3.0,
                        1.5,   120,     1.0,   0.0,
                        5.0,   0.7,     0.05,  0.1]
        16 parameters
        BETA   - parameter that determines the relative contribution to
               runoff from rain or snowmelt
                 [1, 6]
        CET    - Evaporation correction factor (should be 0 if we don't
                 want to change (Oudin et al., 2005) formula values)
                 [0, 0.3]
        FC     - maximum soil moisture storage
                 [50, 500]
        LP     - Threshold for reduction of evaporation (SM/FC)
                 [0.3, 1]
        PERC   - percolation from soil to upper groundwater box
                 [0, 3]
        PCORR  - Precipitation (input sum) correction factor
                 [0.5, 2]
        TT     - Temperature which separate rain and snow fraction of precipitation
                 [-1.5, 2.5]
        CFMAX  - Snow melting rate (mm/day per Celsius degree)
                 [1, 10]
        SFCF   - SnowFall Correction Factor
                 [0.4, 1]
        CFR    - Refreezing coefficient
                 [0, 0.1] (usually 0.05)
        CWH    - Fraction (portion) of meltwater and rainfall which retain
                 in snowpack (water holding capacity)
                 [0, 0.2] (usually 0.1)
        Output:
        simulated river runoff (daily timestep)

        """
        # 1. read input data
        Temp = temp
        Prec = prec
        Evap = evap

        # 2. set the parameters
        parBETA, parCET, parFC, parLP, parPERC, parPCORR, parTT, parCFMAX, \
        parSFCF, parCFR, parCWH = p

        # 3. initialize boxes and initial conditions
        # snowpack box
        SNOWPACK = np.zeros(len(Prec))
        SNOWPACK[0] = 0.0001
        # meltwater box
        MELTWATER = np.zeros(len(Prec))
        MELTWATER[0] = 0.0001
        # soil moisture box
        SM = np.zeros(len(Prec))
        SM[0] = 0.0001
        # soil upper zone box
        SUZ = np.zeros(len(Prec))
        SUZ[0] = 0.0001
        # soil lower zone box
        SLZ = np.zeros(len(Prec))
        SLZ[0] = 0.0001
        # actual evaporation
        ETact = np.zeros(len(Prec))
        ETact[0] = 0.0001

        Qr = np.zeros(len(Prec))

        # 4. meteorological forcing pre-processing
        # overall correction factor
        Prec = parPCORR * Prec
        # precipitation separation
        # if T < parTT: SNOW, else RAIN
        RAIN = np.where(Temp > parTT, Prec, 0)
        SNOW = np.where(Temp <= parTT, Prec, 0)
        # snow correction factor
        SNOW = parSFCF * SNOW
        # evaporation correction
        # a. calculate long-term averages of daily temperature
        Temp_mean = np.array([Temp.loc[Temp.index.dayofyear == x].mean() \
                              for x in range(1, 367)])
        # b. correction of Evaporation daily values
        Evap = Evap.index.map(
            lambda x: (1 + parCET * (Temp[x] - Temp_mean[x.dayofyear - 1])) *
                      Evap[x])
        # c. control Evaporation
        Evap = np.where(Evap > 0, Evap, 0)

        # 5. The main cycle of calculations
        for t in range(1, len(Qr)):

            # 5.1 Snow routine
            # how snowpack forms
            SNOWPACK[t] = SNOWPACK[t - 1] + SNOW[t]
            # how snowpack melts
            # day-degree simple melting
            melt = parCFMAX * (Temp[t] - parTT)
            # control melting
            if melt < 0: melt = 0
            melt = min(melt, SNOWPACK[t])
            # how meltwater box forms
            MELTWATER[t] = MELTWATER[t - 1] + melt
            # snowpack after melting
            SNOWPACK[t] = SNOWPACK[t] - melt
            # refreezing accounting
            refreezing = parCFR * parCFMAX * (parTT - Temp[t])
            # control refreezing
            if refreezing < 0: refreezing = 0
            refreezing = min(refreezing, MELTWATER[t])
            # snowpack after refreezing
            SNOWPACK[t] = SNOWPACK[t] + refreezing
            # meltwater after refreezing
            MELTWATER[t] = MELTWATER[t] - refreezing
            # recharge to soil
            tosoil = MELTWATER[t] - (parCWH * SNOWPACK[t]);
            # control recharge to soil
            if tosoil < 0: tosoil = 0
            # meltwater after recharge to soil
            MELTWATER[t] = MELTWATER[t] - tosoil

            # 5.2 Soil and evaporation routine
            # soil wetness calculation
            soil_wetness = (SM[t - 1] / parFC) ** parBETA
            # control soil wetness (should be in [0, 1])
            if soil_wetness < 0: soil_wetness = 0
            if soil_wetness > 1: soil_wetness = 1
            # soil recharge
            recharge = (RAIN[t] + tosoil) * soil_wetness
            # soil moisture update
            SM[t] = SM[t - 1] + RAIN[t] + tosoil - recharge
            # excess of water calculation
            excess = SM[t] - parFC
            # control excess
            if excess < 0: excess = 0
            # soil moisture update
            SM[t] = SM[t] - excess

            # evaporation accounting
            evapfactor = SM[t] / (parLP * parFC)
            # control evapfactor in range [0, 1]
            if evapfactor < 0: evapfactor = 0
            if evapfactor > 1: evapfactor = 1
            # calculate actual evaporation
            ETact[t] = Evap[t] * evapfactor
            # control actual evaporation
            ETact[t] = min(SM[t], ETact[t])

            # last soil moisture updating
            SM[t] = SM[t] - ETact[t]

            Qr[t] = recharge + excess

        #         # 5.3 Groundwater routine
        #         # upper groundwater box
        #         SUZ[t] = SUZ[t-1] + recharge + excess
        #         # percolation control
        #         perc = min(SUZ[t], parPERC)
        #         # update upper groundwater box
        #         SUZ[t] = SUZ[t] - perc
        #         # runoff from the highest part of upper grondwater box (surface runoff)
        #         Q0 = parK0 * max(SUZ[t] - parUZL, 0)
        #         # update upper groundwater box
        #         SUZ[t] = SUZ[t] - Q0
        #         # runoff from the middle part of upper groundwater box
        #         Q1 = parK1 * SUZ[t]
        #         # update upper groundwater box
        #         SUZ[t] = SUZ[t] - Q1
        #         # calculate lower groundwater box
        #         SLZ[t] = SLZ[t-1] + perc
        #         # runoff from lower groundwater box
        #         Q2 = parK2 * SLZ[t]
        #         # update lower groundwater box
        #         SLZ[t] = SLZ[t] - Q2

        return Qr
