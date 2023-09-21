
# The Saddle-Point Accountant (SPA)
This repository contains the official code for the paper <q>*The Saddle-Point Method in Differential Privacy*</q> (ICML 2022, [[link]](https://proceedings.mlr.press/v202/alghamdi23a.html)). 

Our work introduces a novel mathematical technique to compose privacy curves of differentially private (DP) algorithms in the large composition regime, i.e., when a privacy mechanism is sequentially applied a large number of times. Our method&#8212;dubbed the *saddle-point accountant* (SPA)&#8212;utilizes exponential tilting of the privacy loss random variable to approximate the privacy curve as expressed in the form of a contour integral in our **Theorem 3.1**. The SPA is based on the saddle-point method (or the method of steepest descent), which is a well-known technique in the mathematical physics and statistics literature for approximating contour integrals.

## Replicability 
The notebook `Main_Body_Figures.ipynb` replicates all figures found in the paper.

## Code Overview
#### The directory `saddlepoint_accountant/` contains all code for the Saddle-Point Accountant (SPA) implementation. This includes:
- The method of steepest descent based version of the SPA: $\delta^{(k)}\_{L, \text{ SP-MSD}}(\varepsilon)$ as defined in **Definition 3.6**, and the inverted curve $\varepsilon^{(k)}\_{L, \text{ SP-MSD}}(\delta)$.
- The central limit theorem based version of the SPA: $\delta\_{L, \text{ SP-CLT}}(\varepsilon)$ as defined in **Definition 5.1**, and the inverted curve $\varepsilon\_{L,\text{ SP-CLT}}(\delta)$.
- The error in the central limit theorem based SPA: $\text{err}_{\text{SP}}(\varepsilon;t_0)$ as defined in **Theorem 5.7**, with the choice of $t_0$ being the saddle-point as defined in **Definition 3.2**.
  
#### The subdirectory `saddlepoint_accountant/other_accountants/` contains code for the GDP and Ground-Truth Accountant.
- Of independent interest is the ground-truth accountant, which numerically computes the exact formula for the privacy curve as a contour integral derived in **equation (23)** in **Theorem 3.1**. See **Appendix M** for how this integral is computed, and the README file under `saddlepoint_accountant/other_accountants/` for more implementation details.

## Mechanisms

Currently the following mechanisms are supported:


### Subsampled Gaussian Mechanism

``` python
from saddlepoint_accountant import GaussianMechanism
from saddlepoint_accountant import SaddlePointAccountant

sigma = 2
sampling_prob = 0.01

gaussian_mechanism = GaussianMechanism(sampling_probability = sampling_prob, noise_multiplier = sigma)
saddle_point_accountant = SaddlePointAccountant(gaussian_mechanism)

comps = 3000
eps = 1
delta_approx, error_bound = saddle_point_accountant.compute_delta_clt(eps, comps) 
```
In the code above, ```delta_approx``` computes our approximation $\delta_{L, \text{ SP-CLT}}(\varepsilon)$ of the true privacy budget value $\delta_L(\varepsilon)$, and ```error_bound``` computes the error $\text{err}\_{\text{SP}}(\varepsilon;t_0)$ in this approximation; hence, we have the theoretical guarantee

$$
\left \| \delta_L(\varepsilon) - \delta_{L, \text{ SP-CLT}}(\varepsilon) \right \| \le \text{err}_{\text{SP}}(\varepsilon;t_0)
$$

as in **equation (45)**. The setup is for the subsampled Gaussian mechanism with standard deviation $\sigma = 2$, Poisson subsampling probability $\lambda=0.01$, and $n=3000$ compositions ($L$ denotes the privacy loss random variable of the composed mechanism). The value of the privacy budget $\delta$ is computed above for the specific choice of privacy parameter $\varepsilon = 1$, denoted ```eps```. Varying the value of ```eps``` produces our results in **Figure 1** in the paper (indicated by the dashed lines therein).

Additionally, to compute the other versions of the SPA $(\delta^{(k)}\_{L, \text{ SP-MSD}}(\varepsilon), \text{ } \varepsilon^{(k)}\_{L, \text{ SP-MSD}}(\delta), \text{ }\varepsilon_{L,\text{ SP-CLT}}(\delta))$, use the ```saddle_point_accountant``` class methods:

``` python
saddle_point_accountant.compute_delta_msd(eps, comps, k)
saddle_point_accountant.compute_epsilon_msd(delta, comps, k)
saddle_point_accountant.compute_epsilon_clt(delta, comps)
```


### Laplace Mechanism

``` python
from saddlepoint_accountant import LaplaceMechanism
from saddlepoint_accountant import SaddlePointAccountant

b = 1; sensitivity = 0.01
effective_sensitivity = sensitivity / b

laplace_mechanism = LaplaceMechanism(sens = effective_sensitivity)
saddle_point_accountant = SaddlePointAccountant(laplace_mechanism)

comps = 1000
eps = 1
delta_approx, error_bound = saddle_point_accountant.compute_delta_clt(eps, comps) 
```
In the code above, ```delta_approx``` computes our approximation $\delta_{L, \, \text{SP-CLT}}(\varepsilon)$ of the true privacy budget value $\delta_L(\varepsilon)$, and ```error_bound``` computes the error $\text{err}\_{\text{SP}}(\varepsilon;t_0)$ in this approximation; hence, we have the theoretical guarantee

$$
\left|  \delta\_L(\varepsilon) - \delta\_{L, \text{ SP-CLT}}(\varepsilon) \right| \le \text{err}\_{\text{SP}}(\varepsilon;t_0)
$$

as in **equation (45)**. The setup this time is for the Laplace mechanism with [scale parameter](https://en.wikipedia.org/wiki/Laplace_distribution#Definitions) $b = 1$, sensitivity 0.01, and $n=1000$ compositions ($L$ denotes the privacy loss random variable of the composed mechanism). The value of the privacy budget $\delta$ is computed above for the specific choice of privacy parameter $\varepsilon = 1$, denoted ```eps```. Varying the value of ```eps``` produces our results in **Figure 6** in the paper (indicated by the dashed lines therein).
