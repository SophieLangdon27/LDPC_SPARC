import sys
import os

# Get the parent directory of the current file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np 
from sparc_sophie.sparc_sim_sophie import sparc_sim_sophie, sparc_ldpc_sim_sophie, sparc_ldpc_sim_sophie_re_run
import matplotlib.pyplot as plt # type: ignore

run_sim = True 
run_plots = True

if run_sim: 
    test_num = 13
    code_params_sparc   = { 'P': 15.0,    # Average codeword symbol power constraint
                            'R': 1.0,     # Rate
                            'L': 995,    # Number of sections
                            'M': 512}      # Columns per section
    code_params_ldpc    = { 'P': 15.0,    # Average codeword symbol power constraint
                            'R': 1.109,     # Rate
                            'L': 1103,    # Number of sections
                            'M': 512}      # Columns per section
    # Currently c.K = 1620 
    ldpc_params   = {'standard': '802.11n',
                    'rate'     : '5/6', 
                    'z'        :  81,
                    'int_rate' :  5/6,
                    'mults'    :  3}
    P = code_params_sparc['P']
    decode_params = {'t_max': 25}  # Maximum number of iterations
    num_of_runs   = 3             # Number of encoding/decoding trials
    rng = np.random.RandomState(seed=None) # Random number generator
    # By having the seed as None the random number generator is different each time

    num_vars    = 10
    var_start   = 1 
    var_stop    = 8
    ber_store_sparc  = np.zeros((num_vars, num_of_runs))
    ber_store_sparc_ldpc = np.zeros((num_vars, num_of_runs))
    ber_store_re_run = np.zeros((num_vars, num_of_runs))
    ber_store_averages = np.zeros((num_vars, 3))
    awgn_var_store = np.linspace(var_start, var_stop, num_vars)
    snr_store = P/awgn_var_store

    for i in range(num_of_runs):
        rng_seed = rng.randint(0, 2**31-1, size=2).tolist()    # This generates two random integers, not sure why
        for v in range(num_vars):  
            _,_,_,_,_,_, ber_sparc = sparc_sim_sophie(code_params_sparc, decode_params, awgn_var_store[v], rng_seed) 
            _, _, _, ber_sparc_ldpc = sparc_ldpc_sim_sophie(code_params_ldpc, ldpc_params, decode_params, awgn_var_store[v], rng_seed)
            _, _, ber_re_run = sparc_ldpc_sim_sophie_re_run(code_params_ldpc, ldpc_params, decode_params, awgn_var_store[v], rng_seed)
            ber_store_sparc[v][i] = ber_sparc       
            ber_store_sparc_ldpc[v][i] = ber_sparc_ldpc 
            ber_store_re_run[v][i] = ber_re_run      
            print(f"Run {i+1}: Var {v+1}/{num_vars}")

    # Average over runs for each variance
    for j in range(num_vars): 
        ber_store_averages[j][0] = np.mean(ber_store_sparc[j])
        ber_store_averages[j][1] = np.mean(ber_store_sparc_ldpc[j])
        ber_store_averages[j][2] = np.mean(ber_store_re_run[j])

    ber_store_transpose = ber_store_averages.T

    R1, L1, M1 = map(code_params_sparc.get,['R','L','M']) 
    R2, L2, M2 = map(code_params_ldpc.get,['R','L','M'])
    R3 = round(ldpc_params['int_rate'], 2)
    np.savez(f'performance_plot_arrays/Test_{test_num}_Power_{P}_sparc_R_{R1}_L_{L1}_M_{M1}_ldpc_Rs_{R2}_L_{L2}_M_{M2}_Rl_{R3}_re_run.npz', ber_store = ber_store_transpose, snr_store = snr_store)

if run_plots: 
    plt.figure(figsize=(15,4))
    plt.plot(snr_store, ber_store_transpose[0], marker='o', label='SPARC')
    plt.plot(snr_store, ber_store_transpose[1], marker='o', label='SPARC+LDPC')
    plt.plot(snr_store, ber_store_transpose[2], marker='o', label='SPARC+LDPC+RE_RUN')

    # Add title and labels
    plt.title('Line Plot of Bit Error Rate against SNR')
    plt.xlabel('SNR')
    plt.ylabel('BER')
    plt.legend()  # Show legend

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'performance_plots/Test_{test_num}_Power_{P}_sparc_R_{R1}_L_{L1}_M_{M1}_ldpc_Rs_{R2}_L_{L2}_M_{M2}_Rl_{R3}_re_run.png')
    plt.show()