from abstract_privacy_mechanism import PrivacyMechanism 
from typing import List, Tuple, Union, Sequence
import common #CHANGE IN GITHUB!!!
import math
import numpy as np
from sympy import bell
from scipy.stats import norm
from scipy.optimize import root_scalar


ArrayLike = Union[np.ndarray, List[float]]


class SaddlePointAccountant:

    def __init__(self, 
                 mechanism: PrivacyMechanism) -> None:
        self.mechanism = mechanism

    def diff_j_F(self,
                 t: float,
                 j: int,
                 compositions: int,
                 epsilon: float) -> ArrayLike:
        '''
        Returns F_eps(t), F_eps'(t) , ..., F_eps^(j)(t) 
        '''
        K_lst = self.mechanism.diff_j_cumulant(t,j) #K(t), K'(t), ... K^j(t)
        
        #handle the derivatives of log(t) and log(t+1)
        second_lst = [(-2*(kk%2)+1) * math.factorial(kk-1) * (1/t**kk + 1/(t+1)**kk) for kk in range(1,j+1)]
        
        #handle F_eps(t) and F_eps'(t), both of which have unique terms 
        second_lst = [-epsilon*t -math.log(t) - math.log1p(t)] + second_lst #F_eps(t) term
        second_lst[1] += -epsilon #add missing epsilon term to F_eps'(t)
        
        output = compositions*K_lst + np.array(second_lst)
        return output
    
    def compute_delta_msd(self,
                          epsilon: float,
                          compositions: int,
                          k: int = 1,
                          spa_bounds: Tuple[float,float] = [1e-3,100]) -> float:
        '''
        Computes the order-k method-of-steepest-descent saddle-point accountant 
        '''
 
        def feps(t): return compositions*self.mechanism.cumulant(t) \
                            - epsilon*t  - math.log(t) - math.log1p(t)

        spa = common.get_minima(feps, spa_bounds)

        #compute [ F_eps(spa), F_eps'(spa), ... F_eps^{2k}(spa) ] 
        flst = self.diff_j_F(spa, 2*k, compositions, epsilon)
        
        #delta_correction corresponds to the term in parenthesis in equation 24 of ICML paper
        delta_correction = 1
        
        #update the spa_correction term if k != 1
        if k != 1:
            betas = np.array( [1] + [common.beta(flst[:2*m+1]) for m in range(2,k+1)])
            summed_betas = sum(betas)
            delta_correction = summed_betas
            
        delta_msd1 = math.exp(flst[0]) / math.sqrt(2 * np.pi * math.fabs(flst[2]))

        return delta_msd1 * delta_correction 
    
    def compute_epsilon_msd(self,
                            delta: float,
                            compositions: int,
                            k: int = 1,
                            spa_bounds: Tuple[float,float] = [1e-3, 100],
                            eps_lower: float = 1e-10) -> float:
        '''
        Use binary search to find epsilon as a function of delta. 
        The moments accountant epsilon is used as an upper bound on
        epsilon, and a small value eps_lower is used as a lower bound. 
        '''
        eps_upper = self.compute_epsilon_moments_accountant(delta, compositions, ma_bounds = spa_bounds)
        def get_delta_msd(epsilon): return self.compute_delta_msd(epsilon, compositions, k, spa_bounds)
        return root_scalar(lambda epsilon: get_delta_msd(epsilon) - delta, bracket = [eps_lower, eps_upper]).root

    def compute_epsilon_moments_accountant(self,
                                           delta: float,
                                           compositions: int,
                                           ma_bounds: Tuple[float,float] = [1e-3,100]) -> float:
        
        def f_ma(t): return (compositions*self.mechanism.cumulant(t) + math.log(1/delta)) / t
        
        t_ma = common.get_minima(f_ma, ma_bounds)
        eps_ma = f_ma(t_ma)
        return eps_ma
    
    def compute_delta_clt(self,
                          epsilon: float,
                          compositions: int,
                          spa_bounds: Tuple[float,float] = [1e-3,100]) -> Tuple[float,float]:
        '''
        The CLT version of the saddle-point accountant
        
        Returns both an approximation to delta along with the error term
        '''
        
        def feps(t): return compositions*self.mechanism.cumulant(t)\
                            - epsilon*t  - math.log(t) - math.log1p(t)
        
        spa = common.get_minima(feps, spa_bounds)
        
        #log_q_clt is the logarithm of q(z) as defined in equation 3 of ICML paper
        rv = norm() 
        log_q_clt = lambda z: 0.5 * math.log(2*np.pi) + rv.logsf(z) + z*z/2

        #grab K(spa), K'(spa), K''(spa)
        K0, K1, K2 = self.mechanism.diff_j_cumulant(spa,2)
        
        #compute constants alpha,beta,gamma as defined in Proposition 5.3
        gamma = (compositions*K1 - epsilon)/math.sqrt(compositions*K2)
        alpha = spa*math.sqrt(compositions*K2) - gamma
        beta = alpha + math.sqrt(compositions*K2)
        
        #compute log ( q(alpha) - q(beta) ) in a float point stable manner
        log_q_diff = common._log_sub(log_q_clt(alpha), log_q_clt(beta))
        
        #compute delta clt
        delta_clt = math.exp(compositions*K0 - epsilon*spa - gamma*gamma/2 + log_q_diff) / math.sqrt(2*np.pi)
        
        #compute error term
        pt = self.mechanism.Pt(spa, math.exp(K0), K1) 
        err_clt = math.exp(compositions*K0 - epsilon*spa) * spa**spa / (1+spa)**(1+spa) *\
                    1.12 * compositions * pt / (compositions*K2)**(3/2)
        return (delta_clt, err_clt)
    

    def compute_epsilon_clt(self,
                            delta: float,
                            compositions: int,
                            spa_bounds: Tuple[float,float] = [1e-3, 100],
                            eps_lower: float = 1e-10) -> ArrayLike:
        
        '''
        Use binary search to find epsilon as a function of delta. 
        The moments accountant epsilon is used as an upper bound on
        epsilon, and a small value eps_lower is used as a lower bound. 
        '''
        eps_upper = self.compute_epsilon_moments_accountant(delta, compositions, ma_bounds = spa_bounds)
        
        def get_delta_clt(epsilon, bound):
            delta_clt, err_clt = self.compute_delta_clt(epsilon, compositions, spa_bounds)
            if bound == 'lower': return delta_clt - err_clt
            elif bound == 'none': return delta_clt
            elif bound == 'upper': return delta_clt + err_clt
        
        epsilon_clt_lower = root_scalar(lambda epsilon: get_delta_clt(epsilon, 'lower') - delta, bracket = [eps_lower, eps_upper]).root
        epsilon_clt = root_scalar(lambda epsilon: get_delta_clt(epsilon, 'none') - delta, bracket = [eps_lower, eps_upper]).root
        epsilon_clt_upper = root_scalar(lambda epsilon: get_delta_clt(epsilon, 'upper') - delta, bracket = [eps_lower, eps_upper]).root
        return [epsilon_clt_lower, epsilon_clt, epsilon_clt_upper]
