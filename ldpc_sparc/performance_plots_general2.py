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
from sparc_sophie.sparc_sim_new import sparc_ldpc_sim
from param_calc import param_calc
from ldpc_jossy.py.ldpc import code
import matplotlib.pyplot as plt # type: ignore

#CHANGE BELOW
#-------------------------------------------------------------------------------------------------
# Parameters
num_sims    = 2
# sims_labels = ['SPARC', 'SPARC+LDPC', 'SPARC+LDPC+RE_RUN', 'Integrated_wrong_decoder']
sims_labels = ['SPARC', 'SPARC+LDPC']

test_num = 19
decode_params = {'t_max': 25, 'rtol' : 1e-6}  # Maximum number of iterations
decode_params_integrated_wrong = {'t_max': 5}
num_of_runs   = 1             # Number of encoding/decoding trials
num_snrs    = 5
snr_start   = 1 
snr_stop    = 5 

P = 15.0 
R = 0.8
mults = 1
percent_protected = 1.0 # 100%
assert percent_protected >  0.0
assert percent_protected <= 1.0
M = 256
standard = '802.11n'
ldpc_rate = '5/6'
int_rate = 5/6
z = 81
#---------------------------------------------------------------------------------------------------

L_sparc, R_sparc_ldpc, L_sparc_ldpc, lengths, updated_rate = param_calc(R, mults, percent_protected, M, standard, ldpc_rate, int_rate, z)
print(lengths)

sparc_params        = { 'P': P,      # Average codeword symbol power constraint
                        'R': updated_rate,       # Rate
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
ber_store_averages = np.zeros((num_snrs, num_sims))

snr_store = np.linspace(snr_start, snr_stop, num_snrs)
P = sparc_params['P']
awgn_var_store = P/snr_store

for i in range(num_of_runs):
    rng_seed = rng.randint(0, 2**31-1, size=2).tolist()    # This generates two random integers, not sure why
    for v in range(num_snrs):  
        _, _, ber_store[0][v][i] = sparc_ldpc_sim(sparc_params, ldpc_params, lengths, False, decode_params, awgn_var_store[v], rng_seed) 
        _, _, ber_store[1][v][i] = sparc_ldpc_sim(sparc_ldpc_params, ldpc_params, lengths, True, decode_params, awgn_var_store[v], rng_seed)
        # _, _, ber_store[2][v][i] = sparc_ldpc_sim_sophie_re_run(code_params_ldpc, ldpc_params, decode_params, awgn_var_store[v], lengths, rng_seed)
        print(f"Run {i+1}: Var {v+1}/{num_snrs}")
    
# Average over runs for each variance
for j in range(num_snrs): 
    for s in range(num_sims): 
        ber_store_averages[j][s] = np.mean(ber_store[s][j])

ber_store_transpose = ber_store_averages.T

R1, L1, M1 = map(sparc_params.get,['R','L','M']) 
R2, L2, M2 = map(sparc_ldpc_params.get,['R','L','M'])
R3 = round(ldpc_params['int_rate'], 2)
np.savez(f'performance_plot_arrays/Test_{test_num}_Power_{P}_sparc_R_{R1}_L_{L1}_M_{M1}_ldpc_Rs_{R2}_L_{L2}_M_{M2}_Rl_{R3}.npz', ber_store = ber_store_transpose, snr_store = snr_store)


plt.figure(figsize=(15,4))
for s in range(num_sims): 
    plt.plot(snr_store, ber_store_transpose[s], marker='o', label=sims_labels[s])

# Add title and labels
plt.title('Line Plot of Bit Error Rate against SNR')
plt.xlabel('SNR')
plt.ylabel('BER')
plt.legend()  # Show legend

# Show the plot
plt.tight_layout()
plt.savefig(f'performance_plots/Test_{test_num}_Power_{P}_sparc_R_{R1}_L_{L1}_M_{M1}_ldpc_Rs_{R2}_L_{L2}_M_{M2}_Rl_{R3}.png')
plt.show()