import sys
import os

# Get the parent directory of the current file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
from ldpc_jossy.py.ldpc import code 

def param_calc(R, mults, percent_protected, M, standard, ldpc_rate, int_rate, z): 
      c = code(standard, ldpc_rate, z)
      logM = np.log2(M)
      k_ldpc = c.K * mults 
      # Encoded ldpc bit length 
      n_ldpc = int(k_ldpc / (int_rate))
      assert n_ldpc % logM == 0
      # Unprotected bit length (both multiples of logM)
      unprotected_bits = int((k_ldpc*(1-percent_protected))/percent_protected)
      unprotected_bits = np.ceil(unprotected_bits / logM) * logM

      L_sparc_ldpc_logM = n_ldpc + unprotected_bits
      L_sparc_ldpc = int(L_sparc_ldpc_logM / logM)
      L_unprotected = int(unprotected_bits / logM)
      L_ldpc = L_sparc_ldpc - L_unprotected

      assert L_sparc_ldpc_logM >= n_ldpc
      k = k_ldpc + unprotected_bits 
      n = int(k / R)
      updated_rate = k / n
      L_sparc = int(k // logM) # Integer division finds the multiples of logM in k so that k_sparc is roughly the same. 
      R_sparc_ldpc = L_sparc_ldpc_logM / n

      lengths = { 'cK' : c.K,
                  'k_ldpc' : k_ldpc, 
                  'n_ldpc' : n_ldpc,
                  'mults'  : mults,
                  'L_unprotected' : L_unprotected, 
                  'L_ldpc' : L_ldpc}

      return L_sparc, R_sparc_ldpc, L_sparc_ldpc, lengths, updated_rate




