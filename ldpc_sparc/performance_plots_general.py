''' 
General format: 
Takes parameters and runs simulation and plot for different simulation functions comparing them.
Only need to change the parameters and what the functions are. 
'''
import sys
import os

# Get the parent directory of the current file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np 
from sparc_sophie.sparc_sim_new import *
from param_calc import param_calc, param_calc_semi_protected
from ldpc_jossy.py.ldpc import code
import matplotlib.pyplot as plt # type: ignore

#CHANGE BELOW
#-------------------------------------------------------------------------------------------------
num_sims    = 6
# sims_labels = ['SPARC', 'SPARC+LDPC', 'SPARC+LDPC+RE_RUN', 'Naive_decoder', 'Integrated_decoder']
sims_labels = ['AMP_1 Before BP', 'AMP_1 After BP','AMP_2 Before BP', 'AMP_2 After BP','AMP_3 Before BP', 'AMP_3 After BP']

test_num = 11
decode_params = {'t_max': 13}  
decode_params_integrated = {'t_max': 25}  
num_of_runs   = 1             
num_snrs    = 5
snr_start   = 2.5
snr_stop    = 4.5
semi_protected = False 

if (semi_protected == False): 
    # Assume the same length user bits is used for all sims
    P = 19.44                   
    standard = '802.16' #'802.11n'
    ldpc_rate = '1/2'
    int_rate = 1/2
    z = 150

    mults = 1                   # k = 1620 * mults with 5/6 and 81 params
    logM = 3
    M = pow(2, logM)
    R_sparc_ldpc = 1
    overall_rate, L_sparc, L_sparc_ldpc, lengths = param_calc(mults, logM, standard, ldpc_rate, int_rate, z, R_sparc_ldpc)


else: 
    P = 19.44                   
    R = 0.8
    mults = 3                   # k = 1620 * mults with these params 
    logM = 6
    M = pow(2, logM)

    percent_protected = 1.0 # 100%
    assert percent_protected >  0.0
    assert percent_protected <= 1.0
    standard = '802.11n'
    ldpc_rate = '5/6'
    int_rate = 5/6
    z = 81
    L_sparc, R_sparc_ldpc, L_sparc_ldpc, lengths, overall_rate = param_calc_semi_protected(R, mults, percent_protected, M, standard, 
                                                                            ldpc_rate, int_rate, z)

#---------------------------------------------------------------------------------------------------


sparc_params        = { 'P': P,      # Average codeword symbol power constraint
                        'R': overall_rate,       # Rate
                        'L': L_sparc,       # Number of sections
                        'M': M}       # Columns per section
sparc_ldpc_params   = { 'P': P,      # Average codeword symbol power constraint
                        'R': R_sparc_ldpc,     # Rate
                        'L': L_sparc_ldpc,      # Number of sections
                        'M': M}       # Columns per section
# c.K = 1620 for standard 802.11n; rate 5/6; z 81
ldpc_params   = {'standard': standard, 
                'rate'     : ldpc_rate, 
                'z'        : z,
                'int_rate' : int_rate,      # Rate as an integer to be used elsewhere
                'mults'    : mults}        # Num of chunks of k values 

rng = np.random.RandomState(seed=None) # Random number generator
# By having the seed as None the random number generator is different each time

ber_store = []
for i in range(num_sims): 
    ber_store.append(np.zeros((num_snrs, num_of_runs)))

ber_store_averages = np.zeros((num_sims, num_snrs))
ber_store_max = np.zeros((num_sims, num_snrs))
ber_store_min = np.zeros((num_sims, num_snrs))

snr_store = np.linspace(snr_start, snr_stop, num_snrs)
P = sparc_params['P']
awgn_var_store = P/snr_store

print("Start \n")
for i in range(num_of_runs):
    rng_seed = rng.randint(0, 2**31-1, size=2).tolist()    # This generates two random integers, not sure why
    for v in range(num_snrs):  
        # _, _, ber_store[0][v][i] = sparc_ldpc_sim(sparc_params, ldpc_params, lengths, False, decode_params, awgn_var_store[v], rng_seed) 
        # _, _, ber_store[0][v][i] = sparc_ldpc_sim(sparc_ldpc_params, ldpc_params, lengths, True, decode_params, awgn_var_store[v], rng_seed)
        # _, _, ber_store[0][v][i], ber_store[1][v][i] = sparc_ldpc_sim_test(sparc_ldpc_params, ldpc_params, lengths, True, decode_params, awgn_var_store[v], rng_seed)
        # _, _, ber_store[2][v][i] = sparc_ldpc_naive_sim(sparc_ldpc_params, ldpc_params, lengths, True, decode_params, awgn_var_store[v], rng_seed)
        _,    ber_store[0][v][i], ber_store[1][v][i], ber_store[2][v][i], ber_store[3][v][i], ber_store[4][v][i], ber_store[5][v][i] = naive_sim_test(
            sparc_ldpc_params, ldpc_params, lengths, True, decode_params, awgn_var_store[v], rng_seed)
        # _, _, ber_store[3][v][i] = sparc_ldpc_integrated_sim(sparc_ldpc_params, ldpc_params, lengths, True, decode_params_integrated, awgn_var_store[v], rng_seed)
        # _, _, ber_store[1][v][i] = sparc_ldpc_integrated_no_bp(sparc_ldpc_params, ldpc_params, lengths, True, decode_params_integrated, awgn_var_store[v], rng_seed)
        print(f"Run {i+1}: Var {v+1}/{num_snrs}")
    

print("CHANGE TEST NUMBER")

# Average over runs for each variance
for s in range(num_sims): 
    for j in range(num_snrs): 
        ber_store_averages[s][j] = np.mean(ber_store[s][j])
        ber_store_max[s][j] = np.max(ber_store[s][j])
        ber_store_min[s][j] = np.min(ber_store[s][j])

# Compute error bars
y_err_lower = ber_store_averages - ber_store_min  # Distance from avg to min
y_err_upper = ber_store_max - ber_store_averages  # Distance from avg to max
y_err = [y_err_lower, y_err_upper]  # Asymmetric error bars

R1, L1, M1 = map(sparc_params.get,['R','L','M']) 
R2, L2, M2 = map(sparc_ldpc_params.get,['R','L','M'])
R3 = round(ldpc_params['int_rate'], 2)
np.savez(f'performance_plot_arrays/Test_{test_num}.npz', ber_store_averages = ber_store_averages,
          ber_store_max = ber_store_max, ber_store_min = ber_store_min, snr_store = snr_store)

plt.figure(figsize=(15,4))
for s in range(num_sims):
    plt.errorbar(snr_store, ber_store_averages[s], yerr=[y_err_lower[s], y_err_upper[s]], 
                 fmt='o-', capsize=4, label=sims_labels[s])   

# Add title and labels
plt.title('Line Plot of Bit Error Rate against SNR')
plt.xlabel('SNR')
plt.ylabel('BER')
plt.legend()  # Show legend

# Show the plot
plt.tight_layout()
plt.savefig(f'performance_plots/Test_{test_num}.png')
plt.show()