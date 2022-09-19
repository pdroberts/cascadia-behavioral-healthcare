#!/usr/bin/python
# neural_circuit.py by Patrick D Roberts (2016)
# Neural mass model of similar to Wilson-Cowan model for 2-state activity

import numpy as np
from sympy import *   
from scipy.integrate import cumtrapz

 
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import seaborn as sns           # Public graphing commands
# import matplotlib.cm as cm     # Color map for STH (gray), etc
# import cPickle     # For saving and loading data structures from disk
#=======================================
class NeuralCircuit:    
    """Class for modeling a two-state neural circuit.
    :param p: input parameter
    :type p: int
    """
    def __init__(self, p=None):
        self.p = p
        self.x, self.mu0, self.mu1 = symbols('x, mu0, mu1')
        self.a, self.b0, self.b1 = symbols('a, b0, b1')
        self.x0, self.x1 = symbols('x0, x1')
        self.w00, self.w01, self.w10, self.w11 = symbols('w00, w01, w10, w11')
        self.p00, self.p01, self.p10, self.p11 = symbols('p00, p01, p10, p11')
        self.tau = symbols('tau')
        M_w, M_w_n = self.equation_motion()

    def spike_prob(self):
        """ Defines the spike probability function.        
        :param none: 
        :type none: nothing here
        :returns: sympy function
        """		
        f0 = 1/(1+exp(-self.mu0*self.x))
        f1 = 1/(1+exp(-self.mu1*self.x))
        return f0, f1

    def cubic_poly(self):
        """ Defines the  expansion of the spike probability function to fourth order.        
        :param none: 
        :type none: nothing here
        :returns: sympy function, numpy function
        """		
        f0, f1 = self.spike_prob()
        f0_4 = f0.series(self.x, 0, 5).removeO()
        f1_4 = f1.series(self.x, 0, 5).removeO()
        F0 = f0_4.subs(self.x, self.a * self.x0 + self.b0)
        F0_n = lambdify((self.x0, self.mu0, self.b0, self.a), F0 - self.x0, 'numpy')  # create a numpy functions for numerics
        F1 = f1_4.subs(self.x, self.a * self.x0 + self.b1)
        F1_n = lambdify((self.x0, self.mu1, self.b1, self.a), F1 - self.x0, 'numpy')  # create a numpy functions for numerics
        return F0, F0_n, F1, F1_n

    def equation_motion(self):
        """ Defines the right hand side of the equations of motion.        
        :param none: 
        :type none: nothing here
        :returns: sympy function, numpy function
        """		
        u0, u1 = symbols('u0, u1')
        F0, F0_n, F1, F1_n = self.cubic_poly()
        F0 = F0.subs({self.x0:self.w00*u0 + self.w10*u1})
        F1 = F1.subs({self.x0:self.w10*u0 + self.w11*u1})
        self.M_w = Matrix([ self.w00*F0/self.tau - u0/self.tau, self.w11*F1/self.tau - u1/self.tau])
        M0_w_n = lambdify((u0, u1, self.mu0, self.b0, self.a, self.tau, self.w00, self.w01, self.w10, self.w11), self.M_w[0], 'numpy')  # create a numpy functions for numerics
        M1_w_n = lambdify((u0, u1, self.mu1, self.b1, self.a, self.tau, self.w00, self.w01, self.w10, self.w11), self.M_w[1], 'numpy')  # create a numpy functions for numerics
        self.M_w_n = [M0_w_n, M1_w_n]
        return self.M_w, self.M_w_n

    def potential(self, u_range, w_n, mu0_n, mu1_n, b0_n, b1_n, a_n, tau_n):
        """ Defines the approximate potential function for the equations of motion.        
        :param none: 
        :type none: nothing here
        :returns: sympy function, numpy function
        """		
        pot =  [-cumtrapz(self.M_w_n[0](w_n[0,0]*u_range, w_n[1,1]*u_range, 
                                    mu0_n, b0_n, a_n, tau_n, w_n[0,0], w_n[0,1], w_n[1,0], w_n[1,1]), u_range, initial=0),
               -cumtrapz(self.M_w_n[1](w_n[0,0]*u_range, w_n[1,1]*u_range, 
                                    mu1_n, b0_n, a_n, tau_n, w_n[0,0], w_n[0,1], w_n[1,0], w_n[1,1]), u_range, initial=0)]
        return pot
        
    def fixed_points(self, u_range, w_n, mu0_n, mu1_n, b0_n, b1_n, a_n, tau_n):
        """ Defines the approximate potential function for the equations of motion.        
        :param none: 
        :type none: nothing here
        :returns: sympy function, numpy function
        """		
        pot = self.potential(u_range, w_n, mu0_n, mu1_n, b0_n, b1_n, a_n, tau_n)
        pot = pot[0]
        fp = np.where(np.diff(pot-np.roll(pot,-1) > 0))[0]
        if len(fp)!=3: 
            return [0, 0, 0], 0, 0, 0, 0
        fp1 = fp[0]+1
        fp2 = fp[1]+1
        fp3 = fp[2]+1
        if pot[fp1] < pot[fp3]:
            up_barrier = pot[fp2]-pot[fp1]
            dn_barrier = pot[fp2]-pot[fp3]
            rel_state = pot[fp3]-pot[fp1]			
        else:
            up_barrier = pot[fp2]-pot[fp3]
            dn_barrier = pot[fp2]-pot[fp1]
            rel_state = pot[fp1]-pot[fp3]			
        return [fp1, fp2, fp3], up_barrier, dn_barrier, rel_state, pot
