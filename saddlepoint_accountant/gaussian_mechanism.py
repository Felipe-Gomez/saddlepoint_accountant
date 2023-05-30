
from abstract_privacy_mechanism import PrivacyMechanism
from typing import Tuple

import numpy as np; import math; from scipy.stats import norm
from common import _log_sub, _compute_gaussian_cumulant
from scipy.integrate import quad

class GaussianMechanism(PrivacyMechanism):
    def __init__(self, 
                 sampling_probability: float, 
                 noise_multiplier: float,
                log_mass_truncation_bound: float = -700) -> None:
        
        if noise_multiplier <= 0:
              raise ValueError(f'Noise multiplier is not a positive real number: '
                       f'{standard_deviation}')
        if log_mass_truncation_bound > 0:
              raise ValueError(f'Log mass truncation bound is not a non-positive real '
                       f'number: {log_mass_truncation_bound}')

        self.p = float(sampling_probability)
        self.sigma = float(noise_multiplier)
        
        self.gaussian_random_variable = norm(scale=noise_multiplier)
        self.log_mass_truncation_bound = float(log_mass_truncation_bound)
        
    def log_p(self,
              z: float,
              t: float = 0) -> float:
        '''
        Log density of subsampled Gaussian PLRV at point z and tilting 
        parameter t.
        '''
        p = self.p; sigma = self.sigma

        if math.isclose(z, math.log(1-p)):
            return -np.inf
        logA = math.log(sigma) + (math.log(p) - math.log(2*np.pi))/2 - 1/(8*sigma*sigma)
        logg = _log_sub(z,math.log(1-p))
        non_tilted_pdf = logA + 2*z - 3/2*logg - sigma*sigma/(2)*(logg-math.log(p)) * (logg-math.log(p))
        
        if t == 0:
            return non_tilted_pdf
        else:
            return math.exp(t * z - self.cumulant(t)) * non_tilted_pdf


    def privacy_loss_without_subsampling(self, x: float) -> float:
        '''
        Privacy loss log ( P(x+sensitivity) / P(x) ) assuming sensitivity 1
        with Gaussian pdfs. 
        '''
        return (-0.5 + x) / (self.sigma*self.sigma) 
    
    def get_integration_bounds(self) -> Tuple[float, float]:
        """
        compute the bounds on epsilon values to use in all quadratures.
            epsilon_upper = privacy_loss(lower_x_truncation)
            epsilon_lower = privacy_loss(upper_x_truncation)
        """
        
        
        lower_x_truncation = self.gaussian_random_variable.ppf(
            0.5 * math.exp(self.log_mass_truncation_bound))
        upper_x_truncation = -lower_x_truncation
        upper_x_truncation += 1
        
        return self.privacy_loss(lower_x_truncation), \
             self.privacy_loss(upper_x_truncation)

    
    def cumulant(self,t: float) -> float:
        '''
        Use Mironov's algorithm from Google's TF Privacy to compute the cumulant
        '''
        if self.p != 1:
            return _compute_gaussian_cumulant(q = self.p,sigma=self.sigma, alpha = t + 1)
        
        else: 
            return t/2/self.sigma**2 + (t/self.sigma)**2/2

    def diff_j_MGF(self,
                   t: float,
                   j: int) -> float:
        '''
        Compute jth derivative of MGF at point t via numerical quadrature.
        '''
        bounds = self.get_integration_bounds()
        integrand = lambda l: math.pow(l,j) * math.exp(self.log_p(l) + l * t)
        return quad(integrand, math.log(1-self.p), bounds[1], epsabs = 1e-15, epsrel = 1e-15)[0]

    def Pt(self,
           t: float,
           normalization: float,
           shift: float) -> float:
        bounds = self.get_integration_bounds()
        integrand = lambda l: math.fabs(l - shift)**3 * math.exp(self.log_p(l) + l * t)
        return quad(integrand, math.log(1-self.p), bounds[1], epsabs = 1e-15, epsrel = 1e-15)[0] / normalization
