'''
Includes channel model and full simulation for each function -- encoding, transmitting + decoding. 
Returns bit error rates. 
'''

import numpy as np
from sparc_sophie.sparc_new import sparc_ldpc_encode, sparc_ldpc_decode, naively_integrated_decoder, bit_err_rate

def sparc_ldpc_sim(sparc_params, ldpc_params, lengths, ldpc_bool, decode_params, awgn_var, rand_seed=None): 
    '''
    Simulated encoding, transmitting and decoding 
    '''

    bits_i, beta0, x, A     = sparc_ldpc_encode(sparc_params, ldpc_params, lengths, ldpc_bool, rand_seed)
    y                       = awgn_channel(x, awgn_var, rand_seed) # Produces the received vector after the channel 
    bits_o                  = sparc_ldpc_decode(y, sparc_params, ldpc_params, decode_params, ldpc_bool, lengths, A)
                                        
    ber = bit_err_rate(bits_i, bits_o)
    
    return bits_i, bits_o, ber

def sparc_ldpc_naive_sim(sparc_params, ldpc_params, lengths, ldpc_bool, decode_params, awgn_var, rand_seed=None): 
    '''
    Simulated encoding, transmitting and decoding 
    '''

    bits_i, beta0, x, A     = sparc_ldpc_encode(sparc_params, ldpc_params, lengths, ldpc_bool, rand_seed)
    y                       = awgn_channel(x, awgn_var, rand_seed) # Produces the received vector after the channel 
    bits_o                  = naively_integrated_decoder(y, sparc_params, ldpc_params, decode_params, A)
                                        
    ber = bit_err_rate(bits_i, bits_o)
    
    return bits_i, bits_o, ber


######## Channel models ########

def awgn_channel(input_array, awgn_var, rand_seed):
    '''
    Add Gaussian noise of mean 0 variance awgn_var.

    '''

    assert input_array.ndim == 1, 'input array must be one-dimensional'
    assert awgn_var >= 0

    rng = np.random.RandomState(rand_seed)
    n   = input_array.size

    return input_array + np.sqrt(awgn_var)*rng.randn(n)

