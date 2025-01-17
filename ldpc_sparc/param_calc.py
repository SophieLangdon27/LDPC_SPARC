import numpy as np
from ldpc_jossy.py.ldpc import code 

def param_calc(R, mults, L_sparc_ldpc, M, standard, ldpc_rate, int_rate, z): 
      c = code(standard, ldpc_rate, z)
      logM = np.log2(M)
      k_ldpc = c.K * mults 
      n_ldpc = k_ldpc / (int_rate)
      L_sparc_ldpc_logM = L_sparc_ldpc * logM
      assert L_sparc_ldpc_logM > n_ldpc
      unprotected_bits = L_sparc_ldpc_logM - n_ldpc
      k = k_ldpc + unprotected_bits 
      n = k / R
      L_sparc = k / logM
      R_sparc_ldpc = L_sparc_ldpc_logM / n
      protected_percent = (k_ldpc / k) * 100 
   
      L_parity = (n_ldpc-k_ldpc)/logM
      L_user = L_sparc_ldpc - L_parity
      L_protected = k_ldpc/logM
      L_unprotected = L_user - L_protected 
      L_ldpc = int(L_protected + L_parity)

      lengths = { 'cK' : c.K,
                  'k_ldpc' : k_ldpc, 
                  'n_ldpc' : n_ldpc,
                  'L_parity' : L_parity,
                  'L_user' : L_user,
                  'L_protected' : L_protected,
                  'L_unprotected' : L_unprotected, 
                  'L_ldpc' : L_ldpc}
      
      return L_sparc, R_sparc_ldpc, lengths



