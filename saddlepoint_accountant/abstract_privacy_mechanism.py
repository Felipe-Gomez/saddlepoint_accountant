from abc import ABC, abstractmethod
from typing import Callable, List, Mapping, Optional, Tuple, Union
import common #CHANGE IN GITHUB!!!
import math
import numpy as np
from sympy import bell
from scipy.stats import norm
from scipy.optimize import root_scalar


ArrayLike = Union[np.ndarray, List[float]]


class PrivacyMechanism(ABC):

    def log_p(self, t: float) -> float:
        """
        Compute the log probability density function of the privacy random variable at point t
        conditioned on the value being finite.
        """
        raise NotImplementedError(f"{type(self)} has not provided an implementation for a pdf.")

    def privacy_loss_without_subsampling(self, x: float) -> float:
        """
        Computes the privacy loss, i.e. P(x-sensitivity) / P(x)  at a given point without sub-sampling.
        This is only used if the derivatives of the moment generating function have no closed form. We
        need lower and upper truncation bounds to compute the integral, and this function is needed for that.
        """
        
        raise NotImplementedError(f"{type(self)} has not provided an implementation for privacy loss without subsampling.")

    def privacy_loss(self, x: float) -> float:
        """
        Computes the privacy loss at a given point.
        With sub-sampling probability of q < 1:
        the privacy loss at x is
        log(1-q + q*exp(privacy_loss_without_subsampling(x))).
        """
        privacy_loss_without_subsampling = self.privacy_loss_without_subsampling(x)

        # For performance, the case of sampling probability = 1
        # is handled separately.
        if self.p == 1.0:
          return privacy_loss_without_subsampling
        
        return common.log_a_times_exp_b_plus_c(self.p,
                                                privacy_loss_without_subsampling,
                                                1 - self.p)
    
    def diff_j_cumulant(self,
                        t: float,
                        j: int) -> ArrayLike:
        '''
        Let K(t) be the cumulant
        This function returns K(t), K'(t), K''(t), ... K^(j)(t)
        by using Bell polynomials and Faà di Bruno's formula applied to log
        '''
        cumulanty = self.cumulant(t)
        mgf_derivatives = [self.diff_j_MGF(t,kk) for kk in range(1,j+1)]
    
        output_lst = [cumulanty]
        mgf = math.exp(cumulanty)

        for kk in range(1,j+1):

            output = 0
            for jj in range(1,kk+1):

                factor = (2*(jj%2)-1) * math.factorial(jj-1) / math.pow(mgf,jj)
                bell_term = bell(int(kk), jj, mgf_derivatives[0:int(kk-jj+1)])
                output += factor * bell_term
            output_lst.append(output)

        return np.array(output_lst, dtype = float)
    

    def get_integration_bounds(self) -> Tuple[float, float]:
        """
        When 
        """
        raise NotImplementedError(f"{type(self)} has not provided an implementation for getting the integration bounds.")


    @abstractmethod
    def cumulant(self, t):
        """
        Compute the cumulant of this privacy random variable at point t
        conditioned on the value being finite.
        """
        pass
    
    @abstractmethod
    def diff_j_MGF(self, t, j):
        """
        Compute the jth derivative of the moment generating function
        of this privacy random variable at point t conditioned on the value being finite.
        """
        pass
    
    @abstractmethod
    def Pt(self, t, norm, shift):
        """
        Compute the tensorized absolute third moment of this privacy
        random variable with tilting parameter t
        """
