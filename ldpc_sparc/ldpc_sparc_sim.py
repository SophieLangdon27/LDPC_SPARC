import sys
import os

# Get the parent directory of the current file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np 
from sparc_public_sophie.sparc_sim_sophie import sparc_ldpc_sim_sophie

awgn_var      = 0.5
code_params   = {'P': 15.0,    # Average codeword symbol power constraint
                 'R': 1.3,     # Rate
                 'L': 162,    # Number of sections
                 'M': 16}      # Columns per section
ldpc_params   = {'standard': '802.11n',
                 'rate'    : '1/2', 
                 'z'       : 27}
decode_params = {'t_max': 25}  # Maximum number of iterations
rng = np.random.RandomState(seed=None) # Random number generator
# By having the seed as None the random number generator is different each time
rng_seed = rng.randint(0, 2**31-1, size=2).tolist()

bits_i, bits_o, beta, T, nmse = sparc_ldpc_sim_sophie(code_params, ldpc_params, decode_params, awgn_var, rng_seed)

print("bits_i, bits_o \n", bits_i[0:8], "\n", bits_o[0:8])


