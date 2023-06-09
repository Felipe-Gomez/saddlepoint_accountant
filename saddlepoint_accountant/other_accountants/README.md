# Truth Accountant

Of independent interest is the Truth Accountant. This is the accountant which directly calculates the integral 

$$\delta(\varepsilon) = \frac{1}{2\pi i} \int_{t - i \infty}^{t+ i \infty} e^{F_\varepsilon(z)} dz$$
with exponent 
$$F_\varepsilon(z) = K_L(z) - \varepsilon z - \log z - \log (z+1)$$
via direct high-precision numerical integration. See the ICML paper for details on the meaning of these equations. 

The file ```truth_accountant.py``` 
contains the functions ```compute_epsilon_mp``` and ```compute_delta_mp```, both of which will compute the corresponding privacy parameter (for the subsampled Gaussian mechanism) to arbitrary
accuracy by numerically computing the integral above. It should be emphasized that both of these functions (especially ```compute_epsilon_mp```) 
are computationally expensive, with ```compute_epsilon_mp``` taking over 30 minutes for a single function call on a commercial laptop. Since there is no clean expression for the inverse function $\varepsilon(\delta)$, the function ```compute_epsilon_mp``` takes as input a lower and upper bound on the true underlying $\varepsilon$ parameter, and uses a bounded interval root-finding method to find $\varepsilon(\delta)$. 

The file ```mpmath_s2_q0p01_delta_1em15_n1500_4500_32.pkl``` is the result of calling ```compute_epsilon_mp``` for fixed $\delta = 10^{-15}$ for varying compositions between
1500 and 4500. This file contains the values used in the "Truth" curve of the ICML paper. 

The file ```run_truth_calc_script.py``` is the python file which generated ```mpmath_s2_q0p01_delta_1em15_n1500_4500_32.pkl```. It makes use of python's multiprocessing to paralleize
the computation across multiple compositions. Moreover, it uses the upper and lower bounds on $\varepsilon$ from the SPA-CLT computation to speed up the compuation time. 
