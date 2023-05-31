import mpmath as mp
import numpy as np
import pickle
import time
from .truth_accountant import *
from multiprocessing import Pool

#set the accuracy of the privacy parameters to 20 digits
mp.mp.dps = 20

#initialize parameters
sigma = 2; q = mp.mpf('0.01'); delta = mp.mpf('1e-15')
comps_lst = np.linspace(1.5,4.5,32)*1000

#grab clt bound estimates 
with open("eps_clt_bounds_n1500_4500_32.pkl", "rb") as fb:
    clt_bounds = pickle.load(fb)
    clt_bounds = np.array(clt_bounds)[0,:,:]

def temp_compute_epsilon(comp):

    index = np.where( np.isclose(comps_lst, comp))[0][0]
    return compute_epsilon_mp(sigma,q,int(comp),delta,\
                              eps_lower = clt_bounds[index,0],
                              eps_upper = clt_bounds[index,2])

if __name__ == "__main__":

    print("running")
    pool = Pool()

    t1 = time.time()
    eps_mpmath = pool.map(temp_compute_epsilon, comps_lst)
    t2 = time.time()

    print(f"Wall time for 32 evaluations: {t2-t1} seconds")

    with open("mpmath_s2_q0.01_delta_1em15_n1500_4500_32.pkl", "wb") as fb:
        pickle.dump(eps_mpmath, fb)
        
