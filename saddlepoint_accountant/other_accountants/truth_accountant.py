import mpmath as mp

two_pi = 2 * mp.pi


# The function _compute_a_mp is based on Google's TF Privacy
# https://github.com/google/differential-privacy/blob/main/python/dp_accounting/rdp/rdp_privacy_accountant_test.py

#NOTE: this code uses slightly different notation to the SaddlePoint Accountant.

# In particular, q is the subsampling rate here instead of _lambda.

def _compute_a_mp(sigma, q, alpha):
  """Compute A_alpha for arbitrary alpha by numerical integration."""

  def mu0(x):
    return mp.npdf(x, mu=0, sigma=sigma)

  def _mu_over_mu0(x, q, sigma):
    return (1 - q) + q * mp.exp((2 * x - 1) / (2 * sigma**2))

  def a_alpha_fn(z):
    return mu0(z) * _mu_over_mu0(z, q, sigma)**alpha

  bounds = (-mp.inf, mp.inf)
  a_alpha, _ = mp.quad(a_alpha_fn, bounds, error=True, maxdegree=8)
  return a_alpha


def compute_MGF(sigma, q, t):
    """
    Add 1 to arugment of _compute_a_mp to make it equivalent
    to the moment generating function integral as defined in
    this work.
    """
    return _compute_a_mp(sigma,q, t + 1)


def compute_delta_mp(sigma, q, comp, eps, a = mp.mpf('0.1')):
    """
    Find delta directly evaluating equation 11 in the ICML paper.

    Timing: On a 12-core laptop, one function takes ~3min 28 seconds
    """
    def mgf(t): return compute_MGF(sigma,q, mp.mpc(a,t))

    def g(t): return mp.power( mp.mpc(a + a*a - t*t, 2*a*t + t), -1)

    bounds = (-mp.inf, mp.inf)
    def integrand(t): return mp.re(mp.expj(-t*eps) * \
                                   mp.power(mgf(t),comp) * g(t))
    integral, _ = mp.quad(integrand, bounds, error= True, maxdegree = 8)
    return mp.exp(-a*eps) / two_pi * integral

def compute_epsilon_mp(sigma, q, comp, delta, a = mp.mpf('0.1'), eps_lower, eps_upper):

    """
    Invert compute_delta_mp using the Anderson root finding algorithm in mpmath.
    This function requires one to have a good guess on the lower and
    upper bounds of the true epsilon value. Either the PRV Accountant
    or the SPA-CLT bounds can be used for this purpose.
    """
    def function(eps): return compute_delta_mp(sigma,q,comp,eps,a) - delta
    return mp.findroot(function, (eps_lower, eps_upper), solver = 'anderson')



