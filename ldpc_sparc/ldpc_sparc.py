import sys
import os

# Get the parent directory of the current file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np 
from sparc_public_copy import sparc_posterior_probs
from ldpc.ldpc_copy import code
import matplotlib.pyplot as plt
import time

# Generating Posterior Probabilities for SPARC
# Starting with an LDPC code we will turn it into SPARC 
# In this code it generates a random vecotr of bits to turn into a sparse vector and send, for us we don't want it to be random.
# This will then be sent over the channel and posterior probabilities will be calculated

awgn_var      = 1.0            # AWGN channel noise variance
code_params   = {'P': 15.0,    # Average codeword symbol power constraint
                 'R': 1.3,     # Rate
                 'L': 324,    # Number of sections
                 'M': 16}      # Columns per section
decode_params = {'t_max': 25}  # Maximum number of iterations
rng = np.random.RandomState(seed=None) # Random number generator
rng_seed      = rng.randint(0, 100, size=2).tolist()   #TODO: Changed the bounds, I don't understand what this is 
L,M = map(code_params.get,['L','M'])

# Currently the code below can only take k = 648 input bits and output 1296 bits as an ldpc codeword
raw_input_bits = np.random.randint(0, 2, size=648)
c = code('802.16', '1/2', 54)
ldpc_vec = c.encode(raw_input_bits)
ldpc_vec = ldpc_vec.astype(bool)

assert ldpc_vec.size == L*np.log2(M)

posterior_probs = sparc_posterior_probs(code_params, decode_params, awgn_var, ldpc_vec, rng_seed)

print(posterior_probs[0:16])

# Turning posterior probabilities from AMP to LDPC
for i in range(L):
    