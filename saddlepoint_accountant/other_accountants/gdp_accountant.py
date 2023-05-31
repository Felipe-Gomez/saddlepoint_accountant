

import numpy as np
from scipy import optimize
from scipy import stats


def compute_mu_uniform(epoch, noise_multi, n, batch_size):
  """Compute mu from uniform subsampling."""

  t = epoch * n / batch_size
  c = batch_size * np.sqrt(t) / n
  return np.sqrt(2) * c * np.sqrt(
      np.exp(noise_multi**(-2)) * stats.norm.cdf(1.5 / noise_multi) +
      3 * stats.norm.cdf(-0.5 / noise_multi) - 2)


def compute_mu_poisson(epoch, noise_multi, n, batch_size):
  """Compute mu from Poisson subsampling."""

  t = epoch * n / batch_size
  return np.sqrt(np.exp(noise_multi**(-2)) - 1) * np.sqrt(t) * batch_size / n


def delta_eps_mu(eps, mu):
  """Compute dual between mu-GDP and (epsilon, delta)-DP."""
  return stats.norm.cdf(-eps / mu + mu /
                        2) - np.exp(eps) * stats.norm.cdf(-eps / mu - mu / 2)


def eps_from_mu(mu, delta):
  """Compute epsilon from mu given delta via inverse dual."""

  def f(x):
    """Reversely solve dual by matching delta."""
    return delta_eps_mu(x, mu) - delta
  #print(f'mu is ', mu)
  return optimize.root_scalar(f, bracket=[0, 500], method='brentq').root


def compute_eps_uniform(epoch, noise_multi, n, batch_size, delta):
  """Compute epsilon given delta from inverse dual of uniform subsampling."""

  return eps_from_mu(
      compute_mu_uniform(epoch, noise_multi, n, batch_size), delta)


def compute_eps_poisson(epoch, noise_multi, n, batch_size, delta):
  """Compute epsilon given delta from inverse dual of Poisson subsampling."""

  return eps_from_mu(
      compute_mu_poisson(epoch, noise_multi, n, batch_size), delta)

def compute_delta_poisson(epoch, noise_multi, n, batch_size, eps):
    """Compute epsilon given delta from inverse dual of Poisson subsampling."""

    return delta_eps_mu(eps,
        compute_mu_poisson(epoch, noise_multi, n, batch_size))

def compute_delta_gdp(eps_in,steps,sampling_prob,sigma):
    batch_size = 1000
    n = batch_size/sampling_prob
    epoch = steps*sampling_prob
    delt = compute_delta_poisson(epoch, sigma, n, batch_size, eps_in)
    return delt

def compute_eps_gdp(delta_in,steps,sampling_prob,sigma):
    batch_size = 1000
    n = batch_size/sampling_prob
    epoch = steps*sampling_prob
    eps = compute_eps_poisson(epoch, sigma, n, batch_size, delta_in)
    return 0.0, eps, float('inf')