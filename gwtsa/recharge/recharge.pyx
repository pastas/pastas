"""
Created on Wed Apr 15 18:20:35 2015

This python script is used to solve the differential equations that are used for the
unsaturated zone module. The output of the function is the soil state (how full
it is) at each timestep, and the groundwater recharge N.

-------------------------------------Three models can be used:
- Percolation Flow:
dS/dt = Pe[t] - Kp * (Sr/Srmax)**Gamma - Epu * min(1, Sr/0.5Srmax) 

- Preferential Flow:
dS/ Dt = Pe[t] * (1 - (Sr[t] / Srmax)**Beta)- Epu * min(1, Sr/0.5Srmax)

- Combination:
dS/ Dt = Pe[t] * (1 - (Sr[t] / Srmax)**Beta) - Kp * (Sr/Srmax)**Gamma - Epu
         * min(1, Sr/0.5*Srmax)

-------------------------------------Numerical info:
The Soil module is solved with an implicit euler and Newton Raphson iteration.
The initial estimate for the the NR-iteration is provided by an Explicit Euler
solution of the above differential equation.

-------------------------------------To Do:
    - Built in more external / internal checks for water balance

-------------------------------------References:     
- Kavetski, D., Kuczera, G. & Franks, S.W. [2006]. Calibration of conceptual
hydrological models revisited: 1. Overcoming numerical artefacts.
Journal of Hydrology, 320, p. 173-186.

@author: Raoul Collenteur
"""

# cython: profile=True
# filename: recharge.pyx

import numpy as np
cimport numpy as np

# Define some C-function for more efficient computation
cdef inline double c_max(double a, double b): return a if a >= b else b
cdef inline double c_min(double a, double b): return a if a <= b else b


"""-----------------------------------------------
In this section the preferential flow model is defined. 
dS/ Dt = Pe[t] * (1 - (Sr[t] / Srmax)**Beta)- Epu * min(1, Sr/0.5Srmax)
----------------------------------------------- """    
    
def pref(np.ndarray[np.int_t] Time_Model, np.ndarray[np.float_t, ndim=1] P,
np.ndarray[np.float_t, ndim=1] E, double Srmax = 0.1, double Beta = 2.0,
double Imax = 0.001, int dt = 1, int solver = 1):
    
    cdef int t, iteration, bisection, n
    cdef double error, Last_S, g, g_derivative, a, b, c
    
    n = len(Time_Model) / dt

    error = 1.0e-5
    
    # Create an empty array to store the soil state in
    cdef np.ndarray[np.float_t] S = np.zeros(n)
    S[0] = 0.5 * Srmax   #Set the initial system state
    cdef np.ndarray[np.float_t] Si = np.zeros(n)
    Si[0] = 0.0
    cdef np.ndarray[np.float_t] Pe = np.zeros(n)
    Pe[0] = 0.0    
    cdef np.ndarray[np.float_t] Ei = np.zeros(n)
    Ei[0] = 0.0  
    cdef np.ndarray[np.float_t] Epu = np.zeros(n)
    Epu[0] = 0.0
    cdef np.ndarray[np.float_t] Ea = np.zeros(n)
    Ea[0] = 0.0 
    
    for t in range(n-1):
        Si[t+1] = Si[t] + P[t+1]                # Fill intercEpution bucket with new rain
        Pe[t+1] = c_max(0.0, Si[t+1] - Imax)    # Calculate effective precipitation
        Si[t+1] = Si[t+1] - Pe[t+1]             
        Ei[t+1] = c_min(Si[t+1], E[t+1])       # Evaporation from intercEpution
        Si[t+1] = Si[t+1] - Ei[t+1]             # Update intercEpution state
        Epu[t+1] = E[t+1] - Ei[t+1]             # Update potential evapotranspiration    
    
        Last_S = S[t]
        iteration = 0
        bisection = 1
        
        #Use explicit Euler scheme to find an initial estimate for the newton raphson-method
        
        S[t+1] = c_max(0.0, S[t] + dt * ( Pe[t] * (1 -(S[t] / Srmax)**Beta) - Epu[t] * c_min(1.0, (S[t] / (0.5 * Srmax)) )))
        
        if solver == 1: #If implicit euler is used
            #Start the while loop for the newton-Raphson iteration   
            while abs(Last_S - S[t+1]) > error:
                if iteration > 100:
                    break #Check if the number of iterations is not too high  
                iteration += 1
                Last_S = S[t+1]      
           
                g = Last_S - S[t] - dt * ( Pe[t] * (1 -(Last_S / Srmax)**Beta) - Epu[t] * c_min(1.0, (Last_S / (0.5 * Srmax)) ))       
                # Derivative dEpuends on the state of the system
                if Last_S > (0.5 * Srmax):
                    g_derivative = 1.0 - dt * ( -Beta * Pe[t]* (Last_S / Srmax)**(Beta-1))
                else:        
                    g_derivative = 1.0 - dt * ( -Beta * Pe[t] * (Last_S / Srmax)**(Beta-1) - Epu[t] * (0.5 * Srmax) ) 
                
                # Check if there is no zero-division error            
                if np.isnan(g / g_derivative):
                    bisection = 0
                    break
                # if there is no zero-division error                
                else: # use newton raphson
                    S[t+1] = Last_S - g / g_derivative
            
            if bisection == 0:
                iteration = 0
                a = S[t]
                b = S[t+1]
                c = a + b / 2.0      
                
                while ((b - a)/2.0) > error:
                    if iteration > 100:
                        print 'iteration in bisection method exceeded 100', iteration                        
                        break   
                    iteration += 1 #increase the number of iterations by 1
    
                    if (c - S[t] - dt *( Pe[t] * (1-(c / Srmax)**Beta) - Epu[t] * c_min(1, (c / (0.5 * Srmax))) )) == 0.0:
                        return c # Return the current value if it is correct
                    elif (a - S[t] - dt *( Pe[t] * (1-(a / Srmax)**Beta) - Epu[t] * c_min(1.0, (a / (0.5 * Srmax))) )) * (c - S[t] - dt *( Pe[t] * (1-(c / Srmax)**Beta) - Epu[t] * c_min(1.0, (c / (0.5 * Srmax))) )) > 0.0 :
                        b = c
                    else : 
                        a = c
                        
                    c = a + b / 2.0 
                    
                S[t+1] = c    
                    
            assert ~np.isnan(S[t+1]), 'NaN value calculated for soil state'    
        
        S[t+1] = c_min(Srmax, c_max(0.0,S[t+1])) #Make sure the solution is larger then 0.0 and smaller than Srmax
        Ea[t+1] = Epu[t+1]*c_min(1,(S[t+1]/(0.5*Srmax)))
        
    cdef np.ndarray[np.float_t] R = np.append(0.0, Pe[1:] * dt * 0.5 * ((S[:-1]**Beta + S[1:] ** Beta) / (Srmax **Beta)))
    
    
    return R, S, Ea, Ei

""" -----------------------------------------------
In this section the percolation model is defined. 
dS/dt = Pe[t] - Kp * (Sr/Srmax)**Gamma - Epu * min(1, Sr/0.5Srmax) 
----------------------------------------------- """

def perc(np.ndarray[np.int_t] Time_Model, np.ndarray[np.float_t, ndim=1] P,
np.ndarray[np.float_t, ndim=1] E, double Srmax = 0.1, double Kp = 0.03,
double Gamma = 2.0, double Imax = 0.001, int dt = 1, int solver = 1):
    
    cdef int t, iteration, bisection, n
    cdef double error, Last_S, g, g_derivative, a, b, c
    
    n = len(Time_Model) / dt

    error = 1.0e-5
    
    # Create an empty array to store the soil state in
    cdef np.ndarray[np.float_t] S = np.zeros(n)
    S[0] = 0.5 * Srmax   #Set the initial system state
    cdef np.ndarray[np.float_t] Si = np.zeros(n)
    Si[0] = 0.0
    cdef np.ndarray[np.float_t] Pe = np.zeros(n)
    Pe[0] = 0.0    
    cdef np.ndarray[np.float_t] Ei = np.zeros(n)
    Ei[0] = 0.0  
    cdef np.ndarray[np.float_t] Epu = np.zeros(n)
    Epu[0] = 0.0
    cdef np.ndarray[np.float_t] Ea = np.zeros(n)
    Ea[0] = 0.0 
    
    for t in range(n-1):
        Si[t+1] = Si[t] + P[t+1]                # Fill intercEpution bucket with new rain
        Pe[t+1] = c_max(0.0, Si[t+1] - Imax)    # Calculate effective precipitation
        Si[t+1] = Si[t+1] - Pe[t+1]             
        Ei[t+1] = c_min(Si[t+1], E[t+1])       # Evaporation from intercEpution
        Si[t+1] = Si[t+1] - Ei[t+1]             # Update intercEpution state
        Epu[t+1] = E[t+1] - Ei[t+1]             # Update potential evapotranspiration    
    
        Last_S = S[t]
        iteration = 0
        bisection = 1
        #Use explicit Euler scheme to find an initial estimate for the newton raphson-method
        
        S[t+1] = c_max(0.0, S[t] + dt * ( Pe[t] - Kp * (S[t] / Srmax)**Gamma - Epu[t] * c_min(1.0, (S[t] / (0.5 * Srmax)) )))
        
        if solver == 1:
            #Start the while loop for the newton-Raphson iteration   
            while abs(Last_S - S[t+1]) > error:
                if iteration > 100:
                    break #Check if the number of iterations is not too high  
                iteration += 1
                Last_S = S[t+1]      
           
                g = Last_S - S[t] - dt *( Pe[t] - Kp * (Last_S / Srmax)**Gamma - Epu[t] * c_min(1, (Last_S / (0.5 * Srmax))) )       
                # Derivative dEpuends on the state of the system
                if Last_S > (0.5 * Srmax):
                    g_derivative = 1.0 - dt * ( -Gamma * Kp * (Last_S / Srmax)**(Gamma-1))
                else:        
                    g_derivative = 1.0 - dt * ( -Gamma * Kp * (Last_S / Srmax)**(Gamma-1) - Epu[t] * (0.5 * Srmax) ) 
                
                # Check if there is no zero-division error            
                if np.isnan(g / g_derivative):
                    bisection = 0
                    break
                # if there is no zero-division error                
                else: # use newton raphson
                    S[t+1] = Last_S - g / g_derivative
            
            if bisection == 0:
                iteration = 0
                a = S[t]
                b = S[t+1]
                c = a + b / 2.0      
                
                while ((b - a)/2.0) > error:
                    if iteration > 100:
                        print 'iteration in bisection method exceeded 100', iteration                        
                        break   
                    iteration += 1 #increase the number of iterations by 1
    
                    if (c - S[t] - dt *( Pe[t] - Kp * (c / Srmax)**Gamma - Epu[t] * c_min(1, (c / (0.5 * Srmax))) )) == 0.0:
                        return c # Return the current value if it is correct
                    elif (a - S[t] - dt *( Pe[t] - Kp * (a / Srmax)**Gamma - Epu[t] * c_min(1.0, (a / (0.5 * Srmax))) )) * (c - S[t] - dt *( Pe[t] - Kp * (c / Srmax)**Gamma - Epu[t] * c_min(1.0, (c / (0.5 * Srmax))) )) > 0.0:
                        b = c
                    else : 
                        a = c
                        
                    c = a + b / 2.0 
                    
                S[t+1] = c    
                    
            assert ~np.isnan(S[t+1]), 'NaN-value calculated for soil state'       
        
        S[t+1] = c_min(Srmax, c_max(0.0,S[t+1])) #Make sure the solution is larger then 0.0 and smaller than Srmax
        Ea[t+1] = Epu[t+1]*c_min(1,(S[t+1]/(0.5*Srmax)))
        
    cdef np.ndarray[np.float_t] R = np.append(0.0, Kp * dt * 0.5 * ((S[:-1]**Gamma + S[1:] ** Gamma) / (Srmax **Gamma)))  
    
    return R, S, Ea, Ei
    
    
"""-----------------------------------------------
In this section a combination of the percolation and the preferential flow model is applied:

dS/ Dt = Pe[t] * (1 - (Sr[t] / Srmax)**Beta) - Kp * (Sr/Srmax)**Gamma - Epu * min(1, Sr/0.5*Srmax)
----------------------------------------------- """    
    
def comb(np.ndarray[np.int_t] Time_Model, np.ndarray[np.float_t, ndim=1] P,
np.ndarray[np.float_t, ndim=1] E, double Srmax = 0.1, double Kp = 0.03,
double Beta = 2.0, double Gamma = 2.0, double Imax = 0.001, int dt = 1,
int solver = 1):
    
    cdef int t, iteration, bisection, n
    cdef double error, Last_S, g, g_derivative, a, b, c
    
    n = len(Time_Model) / dt

    error = 1.0e-5
    
    # Create an empty array to store the soil state in
    cdef np.ndarray[np.float_t] S = np.zeros(n)
    S[0] = 0.5 * Srmax   #Set the initial system state
    cdef np.ndarray[np.float_t] Si = np.zeros(n)
    Si[0] = 0.0
    cdef np.ndarray[np.float_t] Pe = np.zeros(n)
    Pe[0] = 0.0    
    cdef np.ndarray[np.float_t] Ei = np.zeros(n)
    Ei[0] = 0.0  
    cdef np.ndarray[np.float_t] Epu = np.zeros(n)
    Epu[0] = 0.0 
    cdef np.ndarray[np.float_t] Ea = np.zeros(n)
    Ea[0] = 0.0 
    
    for t in range(n-1):
        Si[t+1] = Si[t] + P[t+1]                # Fill intercEpution bucket with new rain
        Pe[t+1] = c_max(0.0, Si[t+1] - Imax)    # Calculate effective precipitation
        Si[t+1] = Si[t+1] - Pe[t+1]             
        Ei[t+1] = c_min(Si[t+1], E[t+1])        # Evaporation from interception
        Si[t+1] = Si[t+1] - Ei[t+1]             # Update interception state
        Epu[t+1] = E[t+1] - Ei[t+1]             # Update potential evapotranspiration    
    
        Last_S = S[t]
        iteration = 0
        bisection = 1
        
        #Use explicit Euler scheme to find an initial estimate for the newton raphson-method
        
        S[t+1] = c_max(0.0, S[t] + dt * ( Pe[t] * (1 -(S[t] / Srmax)**Beta) - Kp * (S[t] / Srmax)**Gamma - Epu[t] * c_min(1.0, (S[t] / (0.5 * Srmax)) )))
        
        if solver == 1: #If implicit euler is used
            #Start the while loop for the newton-Raphson iteration   
            while abs(Last_S - S[t+1]) > error:
                if iteration > 100:
                    break #Check if the number of iterations is not too high  
                iteration += 1
                Last_S = S[t+1]      
           
                g = Last_S - S[t] - dt * ( Pe[t] * (1 -(Last_S / Srmax)**Beta) - Kp * (Last_S / Srmax)**Gamma - Epu[t] * c_min(1.0, (Last_S / (0.5 * Srmax)) ))       
                # Derivative depends on the state of the system
                if Last_S > (0.5 * Srmax):
                    g_derivative = 1.0 - dt * ( -Beta * Pe[t]* (Last_S / Srmax)**(Beta-1) -Gamma * Kp * (Last_S / Srmax)**(Gamma-1))
                else:        
                    g_derivative = 1.0 - dt * ( -Beta * Pe[t] * (Last_S / Srmax)**(Beta-1) -Gamma * Kp * (Last_S / Srmax)**(Gamma-1) - Epu[t] * (0.5 * Srmax) ) 
                
                # Check if there is no zero-division error            
                if np.isnan(g / g_derivative):
                    bisection = 0
                    break
                # if there is no zero-division error                
                else: # use newton raphson
                    S[t+1] = Last_S - g / g_derivative
#            
            if bisection == 0:
                iteration = 0
                a = S[t]
                b = S[t+1]
                c = a + b / 2.0      
#                
                while ((b - a)/2.0) > error:
                    if iteration > 100:
                        print 'iteration in bisection method exceeded 100', iteration                        
                        break   
                    iteration += 1 #increase the number of iterations by 1
    
                    if (c - S[t] - dt *( Pe[t] * (1-(c / Srmax)**Beta) - Kp * (c / Srmax)**Gamma - Epu[t] * c_min(1, (c / (0.5 * Srmax))) )) == 0.0:
                        return c # Return the current value if it is correct
                    elif (a - S[t] - dt *( Pe[t] * (1-(a / Srmax)**Beta) - Kp * (a / Srmax)**Gamma - Epu[t] * c_min(1.0, (a / (0.5 * Srmax))) )) * (c - S[t] - dt *( Pe[t] * (1-(c / Srmax)**Beta) - Kp * (c / Srmax)**Gamma - Epu[t] * c_min(1.0, (c / (0.5 * Srmax))) )) > 0.0 :
                        b = c
                    else : 
                        a = c
                       
                    c = a + b / 2.0 
                    
                S[t+1] = c    
                    
            assert ~np.isnan(S[t+1]), 'NaN-value calculated for soil state'       
        
        S[t+1] = c_min(Srmax, c_max(0.0,S[t+1])) #Make sure the solution is larger then 0.0 and smaller than Srmax
        Ea[t+1] = Epu[t+1]*c_min(1,(S[t+1]/(0.5*Srmax)))
    
    cdef np.ndarray[np.float_t] Rs = np.append(0.0, Kp * dt * 0.5 * ((S[:-1]**Gamma + S[1:] ** Gamma) / (Srmax **Gamma))) # Percolation
    cdef np.ndarray[np.float_t] Rf = np.append(0.0, Pe[1:] * dt * 0.5 * ((S[:-1]**Beta + S[1:] ** Beta) / (Srmax **Beta))) # Preferential
    
    return Rs, Rf, S, Ea, Ei