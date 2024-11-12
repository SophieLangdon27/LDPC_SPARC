import sys
import os

# Get the parent directory of the current file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np 
from sparc_public_sophie.sparc_sim_sophie import sparc_sim_sophie
import matplotlib.pyplot as plt

code_params   = {'P': 15.0,    # Average codeword symbol power constraint
                 'R': 1.3,     # Rate
                 'L': 1000,    # Number of sections
                 'M': 32}      # Columns per section
P = code_params['P']
decode_params = {'t_max': 25}  # Maximum number of iterations
num_of_runs   = 10             # Number of encoding/decoding trials
rng = np.random.RandomState(seed=None) # Random number generator
# By having the seed as None the random number generator is different each time

num_vars    = 40
var_start   = 1.5
var_stop    = 4
nmse_store  = np.zeros((num_of_runs, num_vars))
awgn_var_store = np.linspace(var_start, var_stop, num_vars)
snr_store = P/awgn_var_store

for i in range(num_of_runs):
    rng_seed = rng.randint(0, 2**31-1, size=2).tolist()    # This generates two random integers, not sure why
    for v in range(num_vars):  
        bits_i,_,_,T,nmse,_ = sparc_sim_sophie(code_params, decode_params, awgn_var_store[v], rng_seed) 
        nmse_store[i][v] = nmse[T]

plt.figure(figsize=(15,4))
for i in range(num_of_runs): 
    plt.plot(snr_store, nmse_store[i], marker='o', label=f'Row {i + 1}')

# Add title and labels
plt.title('Line Plot of Each Row in Data Array Against X Array')
plt.xlabel('SNR')
plt.ylabel('NMSE')
plt.legend()  # Show legend

# Show the plot
plt.tight_layout()
plt.show()

