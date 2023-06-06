
from saddlepoint_accountant.abstract_privacy_mechanism import PrivacyMechanism
from saddlepoint_accountant.common import _log_add, _log_sub 
from typing import Tuple

import sympy
import numpy as np; import math
from scipy.integrate import quad

class LaplaceMechanism(PrivacyMechanism):
    def __init__(self, 
                 sens: float) -> None:
        
        if sens <= 0:
              raise ValueError(f'Sensitivity is not a positive real number: '
                       f'{sens}')
        self.sens = float(sens)
    
    def cumulant(self,t: float) -> float:
        '''
        The cumulant of the PLRV for the Laplace mechanism has a closed form.
        This function evaluates and returns this functions

        Args:
        t: point at which to evaluate the cumulant of the PLRV

        Returns:
        The cumulant evaluated at t. 
        '''
        sens = self.sens
        if t == 0:
            return 0
        else: 
            denom = _log_add(0, np.log(2) + math.log(t)) #assumption is t is large, could also do np.log1p(2*t)
            num = _log_add(t*sens + math.log1p(t), math.log(t) - sens*(t+1))
            return num - denom

    def diff_j_MGF(self,
                   tt: float,
                   j: int) -> float:
        '''
        Note that the moment generating function has a closed form expression.
        This function computes the jth derivative of MGF at point tt via symbolic
        differentiation.
        
        Args:
        tt: point at which to evaluate the  jth derivative of the MGF
        j: number of derivatvies to compute of the MGF

        Returns:
        The jth derivative of MGF at tt. 
        '''

        #note that t is a sympy symolic variable, while tt is the float input 
        t = sympy.symbols('t', positive = True); s = sympy.symbols('s', positive = True)

        #create symbolic mgf
        symbolic_mgf = (sympy.exp(t * s)*(1+t) + t * sympy.exp(-s-t*s)) / (1+2*t)

        #differentiate and simplify 
        symbolic_mgf_diff = sympy.diff(symbolic_mgf, t, j)
        symbolic_mgf_diff = sympy.simplify(symbolic_mgf_diff)
        symbolic_mgf_diff = sympy.expand_power_exp(symbolic_mgf_diff)

        #create math lambda function
        #TODO: save this lambda function, so as to not recompute it several times
        mgf_diff = sympy.lambdify([t,s], symbolic_mgf_diff, 'math')

        #return value
        return mgf_diff(tt,self.sens)
    
    def Pt(self,
           t: float,
           normalization: float,
           shift: float) -> float:

        '''
        Compute the absolute central third moment of the Laplace privacy
        random variable with tilting parameter t. See section 2.5 of the ICML paper
        for its formal definition. Since the Laplace PLRV has a discrete
        (2 Dirac delta functions at +- sens) and continuous component, the Pt
        integral collpases into 2 constant terms plus a integral with no closed form.

        The function calls the 2 constant terms from the Dirac delta t1 and t3. 
        The remaining calculation is done via numerical quadrature.
        Note that the numerical quadrature is carried out over a finite
        range [-self.sens, self.sens]

        Args:
        
        t: tilting parameter
        
        normalization: Normalization constant needed for the tilted random variable
        (not needed for this particular implementation, since the normalization has a closed form)
        
        shift: The shift needed for this to be a centered moment. Using the ICML paper notation,
        this is E( \tilde{L} )

        Returns:
        absolute central third moment of the Laplace privacy
        random variable
        '''
        sens = self.sens
        t1 = math.exp(- (1+t)*sens) * (shift+sens)**3 / 2
        t3 = math.exp(-t*sens) * math.fabs(shift-sens)**3 / 2
        integrand = lambda x: math.exp( (x-sens)/2 + t*x + 3 * math.log(math.fabs(x - shift)))
        return t1 + quad(integrand, -sens, sens)[0]/4 + t3
