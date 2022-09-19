#!/usr/bin/python
# neural_circuit.py by Patrick D Roberts (2022)
# Neural mass model of similar to Wilson-Cowan model for 2-state activity

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import scipy.optimize as opt  # root-finding algorithm

class Neural_Circuit:    
    """Class for modeling a two-state neural circuit.
    :param p: input parameter
    :type p: int
    """
    def __init__(self):
        self.pars = self.default_pars()

    def default_pars(self):
        pars = {}
        # Excitatory parameters
        pars['tau1'] = 1.     # Timescale of the E population [ms]
        pars['mu1'] = 1.2     # Gain of the E population
        pars['th1'] = 2.8  # Threshold of the E population
        # Inhibitory parameters
        pars['tau_0'] = 2.0    # Timescale of the I population [ms]
        pars['mu0'] = 1.0      # Gain of the I population
        pars['th0'] = 4.0     # Threshold of the I population
        # Connection strength
        pars['w11'] = 8.65  # E to E
        pars['w10'] = 4.   # I to E
        pars['w01'] = 13.  # E to I
        pars['w00'] = 9    # I to I
        # External input
        pars['I_ext_1'] = 0.
        pars['I_ext_0'] = 0.
        # simulation parameters
        pars['T'] = 50.        # Total duration of simulation [ms]
        pars['dt'] = .1        # Simulation time step [ms]
        pars['r1_init'] = 0.2  # Initial value of E
        pars['r0_init'] = 0.2  # Initial value of I
        pars['range_t'] = np.arange(0, pars['T'], pars['dt'])
        return pars

    def wilson_cowan_rhs(self, x):
    # define the right hand of wilson-cowan equations
        rE, rI = x
        pars = self.pars
        tau1, mu1, th1 = pars['tau1'], pars['mu1'], pars['th1']
        tau_0, mu0, th0 = pars['tau_0'], pars['mu0'], pars['th0']
        w11, w10 = pars['w11'], pars['w10']
        w01, w00 = pars['w01'], pars['w00']
        I_ext_1, I_ext_0 = pars['I_ext_1'], pars['I_ext_0']
        drEdt = (-rE + self.f(w11 * rE - w10 * rI + I_ext_1, mu1, th1)) / tau1
        drIdt = (-rI + self.f(w01 * rE - w00 * rI + I_ext_0, mu0, th0)) / tau_0
        y = np.array([drEdt, drIdt])
        return y

    def find_fp(self, pars, r1_init, r0_init):
        """
        Use opt.root function to solve Equations (2)-(3) from initial values
        """
        self.pars = pars
        tau1, mu1, th1 = pars['tau1'], pars['mu1'], pars['th1']
        tau_0, mu0, th0 = pars['tau_0'], pars['mu0'], pars['th0']
        w11, w10 = pars['w11'], pars['w10']
        w01, w00 = pars['w01'], pars['w00']
        I_ext_1, I_ext_0 = pars['I_ext_1'], pars['I_ext_0']


        x0 = np.array([r1_init, r0_init])
        x_fp = opt.root(self.wilson_cowan_rhs, x0).x
        return x_fp

    def f(self, x, a, theta):
        """
        Population activation function, F-I curve
        Args:
        x     : the population input
        a     : the gain of the function
        theta : the threshold of the function
        Returns:
        f     : the population activation response f(x) for input x
        """
        # add the expression of f = f(x)
        f = (1 + np.exp(-a * (x - theta)))**-1 - (1 + np.exp(a * theta))**-1
        return f

    def f_inv(self, x, a, theta):
        """
        Args:
        x         : the population input
        a         : the gain of the function
        theta     : the threshold of the function
        Returns:
        f_inverse : value of the inverse function
        """
        # Calculate Finverse (ln(x) can be calculated as np.log(x))
        f_inverse = -1/a * np.log((x + (1 + np.exp(a * theta))**-1)**-1 - 1) + theta
        return f_inverse


    def get_E_nullcline(self, rE, mu1, th1, w11, w10, I_ext_1, **other_pars):
        """
        Solve for rI along the rE from drE/dt = 0.
        Args:
        rE    : response of excitatory population
        mu1, th1, w11, w10, I_ext_1 : Wilson-Cowan excitatory parameters
        Other parameters are ignored
        Returns:
        rI    : values of inhibitory population along the nullcline on the rE
        """
        # calculate rI for E nullclines on rI
        rI = 1 / w10 * (w11 * rE - self.f_inv(rE, mu1, th1) + I_ext_1)
        return rI


    def get_I_nullcline(self, rI, mu0, th0, w01, w00, I_ext_0, **other_pars):
        """
        Solve for E along the rI from dI/dt = 0.
        Args:
        rI    : response of inhibitory population
        mu0, th0, w01, w00, I_ext_0 : Wilson-Cowan inhibitory parameters
        Other parameters are ignored
        Returns:
        rE    : values of the excitatory population along the nullcline on the rI
        """
        # calculate rE for I nullclines on rI
        rE = 1 / w01 * (w00 * rI + self.f_inv(rI, mu0, th0) - I_ext_0)
        return rE

    def EIderivs(self, rE, rI, tau1, mu1, th1, w11, w10, I_ext_1,
                 tau_0, mu0, th0, w01, w00, I_ext_0, **other_pars):
        """Time derivatives for E/I variables (dE/dt, dI/dt)."""
        # Compute the derivative of rE
        drEdt = (-rE + self.f(w11 * rE - w10 * rI + I_ext_1, mu1, th1)) / tau1
        # Compute the derivative of rI
        drIdt = (-rI + self.f(w01 * rE - w00 * rI + I_ext_0, mu0, th0)) / tau_0
        return drEdt, drIdt


    def calc_barrier(self, pars, verbose=False):
        tau1 = 1
        tau_0 = 1
        I_ext_1 = 0
        I_ext_0 = 0
        Exc_null_rE = np.linspace(-0.01, 0.96, 100)
        Inh_null_rI = np.linspace(-0.01, 0.8, 100)
        Exc_null_rI = self.get_E_nullcline(Exc_null_rE, **pars)
        Inh_null_rE = self.get_I_nullcline(Inh_null_rI, **pars)
        i_nullcline = pd.DataFrame({'re':Inh_null_rE, 'ri':Inh_null_rI})    
        drEdt_list = []
        drIdt_list = []
        for p in range(len(i_nullcline)):
            drEdt, drIdt = self.EIderivs(i_nullcline.loc[p,'re'], i_nullcline.loc[p,'ri'],
                                     tau1, pars['mu1'], pars['th1'], pars['w11'], pars['w10'], I_ext_1,
                                     tau_0, pars['mu0'], pars['th0'], pars['w01'], pars['w00'], I_ext_0)
            drEdt_list.append(drEdt)
            drIdt_list.append(drIdt)
        i_nullcline['dEdt'] = drEdt_list
        i_nullcline['dIdt'] = drIdt_list
        idx_mid_fp = np.abs(Inh_null_rE - self.find_fp(pars, 0.5, 0.5)[0]).argmin()
        idx_high_fp = np.abs(Inh_null_rE - self.find_fp(pars, 1, 0.9)[0]).argmin()
        barrier = i_nullcline.loc[idx_mid_fp:idx_high_fp, 'dEdt'].sum()
        if verbose: 
            print(barrier)
            print(self.find_fp(pars, 0, 0), self.find_fp(pars, 0.5, 0.5), self.find_fp(pars, 1, 0.9))
        return barrier, i_nullcline

    def plot_param_nullclines(self, params_df, index, verbose=False):
        pars = self.default_pars()
        model_pars = ['mu0','mu1','th0','th1','w00', 'w01','w10','w11']
        subject_pars = params_df.loc[index]
        for p in model_pars:
            pars[p] = pars[p]*subject_pars[p]
        Exc_null_rE = np.linspace(-0.01, 0.96, 100)
        Inh_null_rI = np.linspace(-0.01, 0.8, 100)

        # Compute nullclines
        Exc_null_rI = self.get_E_nullcline(Exc_null_rE, **pars)
        Inh_null_rE = self.get_I_nullcline(Inh_null_rI, **pars)

        self.plot_nullclines(Exc_null_rE, Exc_null_rI, Inh_null_rE, Inh_null_rI)
        if verbose:
            print(index, [pars[p] for p in model_pars])
            print(self.find_fp(pars, 0, 0), self.find_fp(pars, 0.4, 0.5), self.find_fp(pars, 1, 0.9))
   
    def plot_nullclines(self, Exc_null_rE, Exc_null_rI, Inh_null_rE, Inh_null_rI):
        plt.plot(Exc_null_rE, Exc_null_rI, 'b', label='E nullcline')
        plt.plot(Inh_null_rE, Inh_null_rI, 'r', label='I nullcline')
        plt.xlabel(r'$r_E$')
        plt.ylabel(r'$r_I$')
        plt.legend(loc='best')

    def plot_potential(self, pars, print_fp=False):
        """ Defines the approximate potential function for the equations of motion.        
        :param none: 
        :type none: nothing here
        :returns: none
        """		
        tau1 = 1
        tau_0 = 1
        I_ext_1 = 0
        I_ext_0 = 0
        # Compute nullclines
        Exc_null_rE = np.linspace(-0.01, 0.96, 100)
        Inh_null_rI = np.linspace(-0.01, 0.8, 100)
        Exc_null_rI = self.get_E_nullcline(Exc_null_rE, **pars)
        Inh_null_rE = self.get_I_nullcline(Inh_null_rI, **pars)
        i_nullcline = pd.DataFrame({'re':Inh_null_rE, 'ri':Inh_null_rI})   
        # Compute derivitives along I nullcline
        drEdt_list = []
        drIdt_list = []
        for p in range(len(i_nullcline)):
            drEdt, drIdt = self.EIderivs(i_nullcline.loc[p,'re'], i_nullcline.loc[p,'ri'],
                                     tau1, pars['mu1'], pars['th1'], pars['w11'], pars['w10'], I_ext_1,
                                     tau_0, pars['mu0'], pars['th0'], pars['w01'], pars['w00'], I_ext_0)
            drEdt_list.append(drEdt)
            drIdt_list.append(drIdt)
        i_nullcline['dEdt'] = drEdt_list
        i_nullcline['dIdt'] = drIdt_list
        i_nullcline['integ_dEdt'] = np.cumsum(-i_nullcline['dEdt'])
        if print_fp: print(self.find_fp(pars, 0, 0), self.find_fp(pars, 0.4, 0.5), self.find_fp(pars, 1, 0.9))
        plt.plot(Exc_null_rE, i_nullcline['integ_dEdt'])
