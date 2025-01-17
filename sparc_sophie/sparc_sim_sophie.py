'''
Includes channel model and full simulation for each function -- encoding, transmitting + decoding. 
Returns bit error rates. 
'''

import numpy as np
from sparc_sophie.sparc_sophie import sparc_ldpc_encode, sparc_ldpc_decode, sparc_encode, sparc_decode, sparc_ldpc_decode_2, bit_err_rate

def sparc_sim_sophie(code_params, decode_params, awgn_var, rand_seed=None): 
    # Currently cheating as encoder directly passes fast transforms Ab and Az to
    # the deocder. (Decoder doesn't use random seed to generate fast transform.)

    # Simulation
    # bits_i = input bits (message), beta0 = sparse vector, x = A.beta, Ab = A in the form to multiply beta, 
    # Az = A transpose in the form to multiply a vector
    bits_i, beta0, x, Ab, Az = sparc_encode(code_params, awgn_var, rand_seed)
    y                        = awgn_channel(x, awgn_var, rand_seed) # Produces the received vector after the channel 
    bits_o, beta, T, nmse, expect = sparc_decode(y, code_params, decode_params,
                                                 awgn_var, rand_seed, beta0, Ab, Az)
    ber = bit_err_rate(bits_i, bits_o)
    
    return bits_i, bits_o, ber

def sparc_ldpc_sim_sophie(code_params, ldpc_params, decode_params, awgn_var, lengths, rand_seed=None): 
    bits_i, beta0, x, Ab, Az = sparc_ldpc_encode(code_params, ldpc_params, awgn_var, lengths, rand_seed)
    y                        = awgn_channel(x, awgn_var, rand_seed) # Produces the received vector after the channel
    bits_o, T   = sparc_ldpc_decode(y, code_params, ldpc_params, decode_params, 
                                                 awgn_var, rand_seed, beta0, lengths, Ab, Az)
    ber = bit_err_rate(bits_i, bits_o)
    
    return bits_i, bits_o, ber

def sparc_ldpc_sim_sophie_re_run(code_params, ldpc_params, decode_params, awgn_var, lengths, rand_seed=None): 
    bits_i, beta0, x, Ab, Az = sparc_ldpc_encode(code_params, ldpc_params, awgn_var, lengths, rand_seed)
    y                        = awgn_channel(x, awgn_var, rand_seed) # Produces the received vector after the channel
    bits_o                   = sparc_ldpc_decode_2(y, code_params, ldpc_params, decode_params, 
                                                 awgn_var, rand_seed, beta0, lengths, Ab, Az)
    ber = bit_err_rate(bits_i, bits_o)
    
    return bits_i, bits_o, ber


######## Channel models ########

def awgn_channel(input_array, awgn_var, rand_seed):
    '''
    Adds Gaussian noise to input array

    Real input_array:
        Add Gaussian noise of mean 0 variance awgn_var.

    Complex input_array:
        Add complex Gaussian noise. Indenpendent Gaussian noise of mean 0
        variance awgn_var/2 to each dimension.
    '''

    assert input_array.ndim == 1, 'input array must be one-dimensional'
    assert awgn_var >= 0

    rng = np.random.RandomState(rand_seed)
    n   = input_array.size

    # Note I have changed this to float64 from original code as it was throwing an error 
    if input_array.dtype == np.float64:  
        return input_array + np.sqrt(awgn_var)*rng.randn(n)

    elif input_array.dtype == np.complex:
        return input_array + np.sqrt(awgn_var/2)*(rng.randn(n)+1j* rng.randn(n))

    else:
        raise Exception("Unknown input type '{}'".format(input_array.dtype))

