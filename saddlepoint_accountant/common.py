
import numpy as np
import math
from typing import Tuple, Union, List
import mpmath as mp
from scipy import special
from scipy.special import erfc
from scipy.optimize import minimize_scalar
from sympy import bell, symbols, lambdify

ArrayLike = Union[np.ndarray, List[float]]

def get_minima(function, bounds):
    '''
    Search for the minima of "function" via binary search over "bounds".

    This function is used to compute the saddlepoint of F_eps, as the saddlepoint 
    is a minima along the real axis, and to find the optimal "order" for the moments accountant. 

    '''
    minima = minimize_scalar(function, bounds = bounds, method = 'bounded').x

    #if the saddlepoint is close to the endpoints, throw an error
    if math.isclose(minima, bounds[0]):
        raise RuntimeError("saddlepoint is close to lower bound, try decreasing the default lower bound in spa_bounds")
    if math.isclose(minima, bounds[1]):
        raise RuntimeError("saddlepoint is close to upper bound, try increasing the default upper bound in spa_bounds")
    return minima
    

def complete_bell(n):
    '''
    computes the nth complete exponential Bell polynomial in sympy
    and outputs a numpy lambda function that evaluates it
    '''
    out = 0
    for k in range(1,n+1):
        stry = str(n-k+2)
        out += bell(n,k,symbols('x:'+stry)[1:])
    stry = str(n + 1)
    return lambdify(symbols('x:'+stry)[1:], out, 'numpy')


def beta(flst: ArrayLike) -> float:
    '''
    Computes equation 22 of ICML paper, i.e. beta_{epsilon, m}
    These can be thought of as the higher order corrections to the saddlepoint
        
    input:
        
    flst: output from diff_j_F evaluated at the saddlepoint, i.e.
    a numpy array of [F_eps(spa), F_eps'(spa), F_eps''(spa), F_eps^3(spa) ... F_eps^{2m}(spa)]
        
    output: beta_{epsilon, m} = (-1)^m Beta((0,0 F_eps^3(spa) ... F_eps^{2m}(spa)) / 2^m m! F_eps''(spa)^m
    '''
    
    assert float( ( len(flst) - 1 ) / 2).is_integer()
        
    m = int( ( len(flst) - 1 ) / 2 )
    input_lst = [0,0] + flst[3:].tolist() #0, 0, F^3(spa) ... F^{2m}(spa)

    return (-2*(m%2) + 1) * complete_bell(2*m)(*input_lst) / (math.pow(2,m) * math.factorial(m) * math.pow(flst[2],m))

    
# The following code is based on Google's TF Privacy
# https://github.com/google/differential-privacy/blob/main/python/dp_accounting/rdp/rdp_privacy_accountant.py
def _log_add(logx: float, logy: float) -> float:
  """Adds two numbers in the log space."""
  a, b = min(logx, logy), max(logx, logy)
  if a == -np.inf:  # adding 0
    return b
  # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
  return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)


def _log_sub(logx: float, logy: float) -> float:
  """Subtracts two numbers in the log space. Answer must be non-negative."""
  if logx < logy:
    raise ValueError('The result of subtraction must be non-negative.')
  if logy == -np.inf:  # subtracting 0
    return logx
  if logx == logy:
    return -np.inf  # 0 is represented as -np.inf in the log space.

  try:
    # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
    return math.log(math.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
  except OverflowError:
    return logx

def mp_log_add(logx, logy):
  """Adds two numbers in the log space."""
  a, b = min(logx, logy), max(logx, logy)
  if a == -mp.inf:  # adding 0
    return b
  # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
  return mp.log1p(mp.exp(a - b)) + b  # log1p(x) = log(x + 1)


def mp_log_sub(logx, logy):
  """Subtracts two numbers in the log space. Answer must be non-negative."""
  if logx < logy:
    raise ValueError('The result of subtraction must be non-negative.')
  if logy == -mp.inf:  # subtracting 0
    return logx
  if logx == logy:
    return -mp.inf  # 0 is represented as -np.inf in the log space.

  try:
    # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
    return mp.log(mp.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
  except OverflowError:
    return logx

def log_a_times_exp_b_plus_c(a: float, b: float, c: float) -> float:
  """Computes log(a * exp(b) + c)."""
  if a == 0:
    return math.log(c)
  if a < 0:
    if c <= 0:
      raise ValueError(f'a exp(b) + c must be positive: {a}, {b}, {c}.')
    return _log_sub(math.log(c), math.log(-a) + b)
  if b == 0:
    return math.log(a + c)
  d = b + math.log(a)
  if c == 0:
    return d
  elif c < 0:
    return _log_sub(d, math.log(-c))
  else:
    return _log_add(d, math.log(c))


def _log_comb(n: int, k: int) -> float:
  """Computes log of binomial coefficient."""
  return (special.gammaln(n + 1) - special.gammaln(k + 1) -
          special.gammaln(n - k + 1))


def _compute_log_a_int(q: float, sigma: float, alpha: int) -> float:
  """Computes log(A_alpha) for integer alpha, 0 < q < 1."""

  # Initialize with 0 in the log space.
  log_a = -np.inf

  for i in range(alpha + 1):
    log_coef_i = (
        _log_comb(alpha, i) + i * math.log(q) + (alpha - i) * math.log(1 - q))

    s = log_coef_i + (i * i - i) / (2 * (sigma**2))
    log_a = _log_add(log_a, s)

  return float(log_a)


def _compute_log_a_frac(q: float, sigma: float, alpha: float) -> float:
  """Computes log(A_alpha) for fractional alpha, 0 < q < 1."""
  # The two parts of A_alpha, integrals over (-inf,z0] and [z0, +inf), are
  # initialized to 0 in the log space:
  log_a0, log_a1 = -np.inf, -np.inf
  i = 0

  z0 = sigma**2 * math.log(1 / q - 1) + .5

  while True:  # do ... until loop
    coef = special.binom(alpha, i)
    log_coef = math.log(abs(coef))
    j = alpha - i

    log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
    log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)

    log_e0 = math.log(.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
    log_e1 = math.log(.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))

    log_s0 = log_t0 + (i * i - i) / (2 * (sigma**2)) + log_e0
    log_s1 = log_t1 + (j * j - j) / (2 * (sigma**2)) + log_e1

    if coef > 0:
      log_a0 = _log_add(log_a0, log_s0)
      log_a1 = _log_add(log_a1, log_s1)
    else:
      log_a0 = _log_sub(log_a0, log_s0)
      log_a1 = _log_sub(log_a1, log_s1)

    i += 1
    if max(log_s0, log_s1) < -30:
      break

  return _log_add(log_a0, log_a1)


def _log_erfc(x: float) -> float:
  """Computes log(erfc(x)) with high accuracy for large x."""
  try:
    return math.log(2) + special.log_ndtr(-x * 2**.5)
  except NameError:
    # If log_ndtr is not available, approximate as follows:
    r = special.erfc(x)
    if r == 0.0:
      # Using the Laurent series at infinity for the tail of the erfc function:
      #     erfc(x) ~ exp(-x^2-.5/x^2+.625/x^4)/(x*pi^.5)
      # To verify in Mathematica:
      #     Series[Log[Erfc[x]] + Log[x] + Log[Pi]/2 + x^2, {x, Infinity, 6}]
      return (-math.log(math.pi) / 2 - math.log(x) - x**2 - .5 * x**-2 +
              .625 * x**-4 - 37. / 24. * x**-6 + 353. / 64. * x**-8)
    else:
      return math.log(r)

def _compute_gaussian_cumulant(q: float, sigma: float,
                   alpha: Union[int, float]) -> float:
  if float(alpha).is_integer():
    return _compute_log_a_int(q, sigma, int(alpha))
  else:
    return _compute_log_a_frac(q, sigma, alpha)


