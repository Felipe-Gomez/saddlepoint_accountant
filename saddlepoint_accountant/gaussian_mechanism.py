
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

        if sampling_probability == 1:
            raise ValueError(f"Only Poisson subsampled Gaussian is implemented for now!"
                             f" Try subsampling probabilities less than 1.")

        self._lambda = float(sampling_probability)
        self.sigma = float(noise_multiplier)
        
        self.gaussian_random_variable = norm(scale=noise_multiplier)
        self.log_mass_truncation_bound = float(log_mass_truncation_bound)

        
        
    def log_p(self,
              z: float) -> float:
        '''
        Log density of subsampled Gaussian PLRV at point z and tilting 
        parameter t. See Appendix C in ICML paper for the expression.

        Args:
        z: point to evaluate the log probability density

        Returns:
        Log probability density
        '''
        
        _lambda = self._lambda; sigma = self.sigma

        if math.isclose(z, math.log(1-_lambda)):
            return -np.inf
        logA = math.log(sigma) + (math.log(_lambda) - math.log(2*np.pi))/2 - 1/(8*sigma*sigma)
        logg = _log_sub(z,math.log(1-_lambda))
        return logA + 2*z - 3/2*logg - sigma*sigma/(2)*(logg-math.log(_lambda)) * (logg-math.log(_lambda))



    def privacy_loss_without_subsampling(self, x: float) -> float:
        '''
        Privacy loss log ( P(x+sensitivity) / P(x) ) assuming sensitivity 1
        with Gaussian pdfs.

        Args:
        x: Point at which to evaluate the privacy loss at.

        Returns:
        Privacy loss at x
        '''
        return (-0.5 + x) / (self.sigma*self.sigma) 
    
    def get_integration_bounds(self) -> Tuple[float, float]:
        """
        compute the bounds on epsilon values to use in all quadratures.
        Based on code from
        https://github.com/google/differential-privacy/blob/main/python/dp_accounting/pld/privacy_loss_mechanism.py

        The truncations lower_x_truncation and upper_x_truncation
        in this function ensure that the tails of both P(x) and P(x-1)
        are smaller than 0.5 * exp(log_mass_truncation_bound). These
        truncations are then passed through self.privacy_loss to convert
        them into bounds on the privacy loss random variable.

        Returns:
        A bounded region of the privacy loss random variable. This region
        is used as an integration range for diff_j_MGF. 
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
        return _compute_gaussian_cumulant(q = self._lambda,sigma=self.sigma, alpha = t + 1)

    def diff_j_MGF(self,
                   t: float,
                   j: int) -> float:
        '''
        Computes the jth derivative of MGF at point t via numerical quadrature.
        Note that the numerical quadrature is carried out over a finite
        range, which is found by get_integration_bounds()

        Args:
        t: point at which to evaluate the  jth derivative of the MGF
        j: number of derivatvies to compute of the MGF

        Returns:
        The jth derivative of MGF at t. 
        '''

        bounds = self.get_integration_bounds()
        integrand = lambda l: math.pow(l,j) * math.exp(self.log_p(l) + l * t)
        return quad(integrand, math.log(1-self._lambda), bounds[1], epsabs = 1e-15, epsrel = 1e-15)[0]

    def Pt(self,
           t: float,
           normalization: float,
           shift: float) -> float:

        '''
        Compute the absolute central third moment of the Gaussian privacy
        random variable with tilting parameter t. See section 2.5 of the ICML paper
        for its formal definition.
        The calculation is done via numerical quadrature.
        Note that the numerical quadrature is carried out over a finite
        range, which is found by get_integration_bounds()

        Args:
        
        t: tilting parameter
        
        normalization: Normalization constant needed for the tilted random variable. 
        
        shift: The shift needed for this to be a centered moment. Using the ICML paper notation,
        this is E( \tilde{L} )

        Returns:
        absolute central third moment of the Gaussian privacy
        random variable
        '''
        bounds = self.get_integration_bounds()
        integrand = lambda l: math.fabs(l - shift)**3 * math.exp(self.log_p(l) + l * t)
        return quad(integrand, math.log(1-self._lambda), bounds[1], epsabs = 1e-15, epsrel = 1e-15)[0] / normalization
