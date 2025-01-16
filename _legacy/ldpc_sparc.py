import sys
import os

# Get the parent directory of the current file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np 
from ldpc_jossy.py.ldpc import code
import matplotlib.pyplot as plt # type: ignore
from sparc_sophie import sparc_posterior_probs
import time

import warnings

# Generating Posterior Probabilities for SPARC
# Starting with an LDPC code we will turn it into SPARC 
# In this code it generates a random vecotr of bits to turn into a sparse vector and send, for us we don't want it to be random.
# This will then be sent over the channel and posterior probabilities will be calculated

awgn_var      = 1.0            # AWGN channel noise variance
code_params   = {'P': 15.0,    # Average codeword symbol power constraint
                 'R': 1.3,     # Rate
                 'L': 162,    # Number of sections
                 'M': 16}      # Columns per section
decode_params = {'t_max': 25}  # Maximum number of iterations
rng = np.random.RandomState(seed=None) # Random number generator
rng_seed = rng.randint(0, 2**31-1, size=2).tolist()   #TODO: Changed the bounds, I don't understand what this is 
L,M = map(code_params.get,['L','M'])
logM = int(np.log2(M))

# Currently the code below can only take k = 324 input bits and output 648 bits as an ldpc codeword
raw_input_bits = np.random.randint(0, 2, size=324)
print("Message bits [0:4] ", raw_input_bits[0:4], "\n")
c = code('802.11n', '1/2', 27)
ldpc_vec = c.encode(raw_input_bits)
# print("ldpc encoded bits [0:4] systematic ", ldpc_vec[0:4], "\n")
ldpc_vec = ldpc_vec.astype(bool)

assert ldpc_vec.size == L*logM

posterior_probs = sparc_posterior_probs(code_params, decode_params, awgn_var, ldpc_vec, rng_seed)

# print("posterior probs of sparse bits ", posterior_probs[0:16])

posterior_probs_sectioned = posterior_probs.reshape(L, M)
ldpc_probs = np.zeros((L, logM)) 

# Turning posterior probabilities from AMP to LDPC 
for l in range(L):
    for i in range(logM): 
        b = logM - 1 - i
        k = 0
        while k < M: 
            for j in range(k, k+pow(2,i)):
                ldpc_probs[l][b] += posterior_probs_sectioned[l][j]
            k = k + pow(2, i+1)

ldpc_probs = ldpc_probs.reshape(L*logM)

# print("posterior probs of being 0 of ldpc bits ", ldpc_probs[0:4], "\n")

# We have postierior probabilites we need to convert these to LLRs so that we can further decode. 
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in log.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")
    
    # The line where the warning is raised
    LLR = np.log(ldpc_probs)- np.log(1-ldpc_probs)

# print("LLR [0:4] ", LLR[0:4], "\n")

# set -inf and +inf to real numbers with v large magnitude
# LLR = np.nan_to_num(LLR)
large_negative = -100.0
large_positive = 100.0
# Replace inf and -inf
LLR = np.where(LLR == np.inf, large_positive, LLR)
LLR = np.where(LLR == -np.inf, large_negative, LLR)

# print("LLR [0:4] ", LLR[0:4], "\n")

# LLR = LLR / 20 
# print("LLRs / 10 [0:4] ", LLR[0:4], "\n")

app, it = c.decode(LLR)
print("LLR [0:4] after ldpc decoding ", app[0:4], " iterations ", it, "\n") 

# Back to postierior probabilites 
# decoded_msg = np.exp(app)/ (1+np.exp(app))
# print("Probs [0:4] after ldpc decoding ", decoded_msg[0:4], "\n") 

# Hard decoding 
hard_bools = (app < 0)
hard_decoded = [int(bool_val) for bool_val in hard_bools]
print("hard_decoded [0:4] after ldpc decoding ", hard_decoded[0:4], "\n") 

        
    