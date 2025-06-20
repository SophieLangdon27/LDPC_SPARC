import sys
import os

# Get the parent directory of the current file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
from ldpc_jossy.py.ldpc import code 
from sparc_sophie.sparc_new import *

def sparc_ldpc_sim(sparc_params, ldpc_params, lengths, ldpc_bool, decode_params, awgn_var, rand_seed=None): 
    '''
    Simulated encoding, transmitting and decoding 
    '''

    bits_i, total_bits, beta0, x, A     = sparc_ldpc_encode(sparc_params, ldpc_params, lengths, ldpc_bool, rand_seed)
    y                       = awgn_channel(x, awgn_var, rand_seed) # Produces the received vector after the channel 
    bits_o                  = sparc_ldpc_decode(y, sparc_params, ldpc_params, decode_params, ldpc_bool, lengths, A)
                                        
    ber = bit_err_rate(bits_i, bits_o)
    
    return bits_i, bits_o, ber

def sparc_ldpc_sim_loop(sparc_params, ldpc_params, lengths, ldpc_bool, decode_params, awgn_var, rand_seed=None): 
    '''
    Simulated encoding, transmitting and decoding 
    '''

    bits_i, total_bits, beta0, x, A     = sparc_ldpc_encode(sparc_params, ldpc_params, lengths, ldpc_bool, rand_seed)
    y                       = awgn_channel(x, awgn_var, rand_seed) # Produces the received vector after the channel 
    bits_o                  = sparc_ldpc_decode_loop(y, sparc_params, ldpc_params, decode_params, A)
                                        
    ber = bit_err_rate(bits_i, bits_o)
    
    return bits_i, bits_o, ber

def sparc_ldpc_sim_test(sparc_params, ldpc_params, lengths, ldpc_bool, decode_params, awgn_var, rand_seed=None): 
    '''
    Simulated encoding, transmitting and decoding 
    '''

    bits_i, total_bits, beta0, x, A     = sparc_ldpc_encode(sparc_params, ldpc_params, lengths, ldpc_bool, rand_seed)
    y                       = awgn_channel(x, awgn_var, rand_seed) # Produces the received vector after the channel 
    bits_o_bp, bits_o_no_bp             = sparc_ldpc_decode_test_3(y, sparc_params, ldpc_params, decode_params, ldpc_bool, A)
                                        
    ber_1 = bit_err_rate(bits_i, bits_o_bp)
    ber_2 = bit_err_rate(bits_i, bits_o_no_bp)
    
    return bits_i, bits_o_bp, ber_1, ber_2

def sparc_ldpc_naive_sim(sparc_params, ldpc_params, lengths, ldpc_bool, decode_params, awgn_var, rand_seed=None): 
    '''
    Simulated encoding, transmitting and decoding 
    '''

    bits_i, total_bits, beta0, x, A     = sparc_ldpc_encode(sparc_params, ldpc_params, lengths, ldpc_bool, rand_seed)
    y                       = awgn_channel(x, awgn_var, rand_seed) # Produces the received vector after the channel 
    bits_o                  = naively_integrated_decoder(y, sparc_params, ldpc_params, decode_params, A)
                                        
    ber = bit_err_rate(bits_i, bits_o)
    
    return bits_i, bits_o, ber

def sparc_ldpc_naive_sim_posteriors(sparc_params, ldpc_params, lengths, ldpc_bool, decode_params, awgn_var, rand_seed=None): 
    '''
    Simulated encoding, transmitting and decoding 
    '''

    bits_i, total_bits, beta0, x, A     = sparc_ldpc_encode(sparc_params, ldpc_params, lengths, ldpc_bool, rand_seed)
    y                       = awgn_channel(x, awgn_var, rand_seed) # Produces the received vector after the channel 
    bits_o                  = naively_integrated_decoder_posteriors(y, sparc_params, ldpc_params, decode_params, A)
                                        
    ber = bit_err_rate(bits_i, bits_o)
    
    return bits_i, bits_o, ber

def no_onsager_sim(sparc_params, ldpc_params, lengths, ldpc_bool, decode_params, awgn_var, rand_seed=None): 
    '''
    Simulated encoding, transmitting and decoding 
    '''

    bits_i, total_bits, beta0, x, A     = sparc_ldpc_encode(sparc_params, ldpc_params, lengths, ldpc_bool, rand_seed)
    y                       = awgn_channel(x, awgn_var, rand_seed) # Produces the received vector after the channel 
    bits_o                  = no_onsager_decoder(y, sparc_params, ldpc_params, decode_params, A)
                                        
    ber = bit_err_rate(bits_i, bits_o)
    
    return bits_i, bits_o, ber

def naive_sim_test(sparc_params, ldpc_params, lengths, ldpc_bool, decode_params, awgn_var, rand_seed=None): 

    bits_i, total_bits, beta0, x, A     = sparc_ldpc_encode(sparc_params, ldpc_params, lengths, ldpc_bool, rand_seed)
    y                       = awgn_channel(x, awgn_var, rand_seed) # Produces the received vector after the channel 
    decoded_user_bits_arr             = naively_integrated_test_4(y, sparc_params, ldpc_params, decode_params, lengths, A)

    ber_1 = bit_err_rate(bits_i, decoded_user_bits_arr[0])
    ber_2 = bit_err_rate(bits_i, decoded_user_bits_arr[1])
    ber_3 = bit_err_rate(bits_i, decoded_user_bits_arr[2])
    ber_4 = bit_err_rate(bits_i, decoded_user_bits_arr[3])
    ber_5 = bit_err_rate(bits_i, decoded_user_bits_arr[4])
    ber_6 = bit_err_rate(bits_i, decoded_user_bits_arr[5])
    ber_7 = bit_err_rate(bits_i, decoded_user_bits_arr[6])
    ber_8 = bit_err_rate(bits_i, decoded_user_bits_arr[7])
    ber_9 = bit_err_rate(bits_i, decoded_user_bits_arr[8])
    ber_10 = bit_err_rate(bits_i, decoded_user_bits_arr[9])
    ber_11 = bit_err_rate(bits_i, decoded_user_bits_arr[10])
    ber_12 = bit_err_rate(bits_i, decoded_user_bits_arr[11])
    ber_13 = bit_err_rate(bits_i, decoded_user_bits_arr[12])
    ber_14 = bit_err_rate(bits_i, decoded_user_bits_arr[13])
    ber_15 = bit_err_rate(bits_i, decoded_user_bits_arr[15])
    ber_16 = bit_err_rate(bits_i, decoded_user_bits_arr[15])
    
    return bits_i, ber_1, ber_2, ber_3, ber_4, ber_5, ber_6, ber_7, ber_8, ber_9, ber_10, ber_11, ber_12, ber_13, ber_14, ber_15, ber_16

def naive_posteriors_sim_test(sparc_params, ldpc_params, lengths, ldpc_bool, decode_params, awgn_var, rand_seed=None): 

    bits_i, total_bits, beta0, x, A     = sparc_ldpc_encode(sparc_params, ldpc_params, lengths, ldpc_bool, rand_seed)
    y                       = awgn_channel(x, awgn_var, rand_seed) # Produces the received vector after the channel 
    decoded_user_bits_arr             = naively_integrated_decoder_posteriors_test(y, sparc_params, ldpc_params, decode_params, lengths, A)

    ber_1 = bit_err_rate(bits_i, decoded_user_bits_arr[0])
    ber_2 = bit_err_rate(bits_i, decoded_user_bits_arr[1])
    ber_3 = bit_err_rate(bits_i, decoded_user_bits_arr[2])
    ber_4 = bit_err_rate(bits_i, decoded_user_bits_arr[3])
    ber_5 = bit_err_rate(bits_i, decoded_user_bits_arr[4])
    ber_6 = bit_err_rate(bits_i, decoded_user_bits_arr[5])
    ber_7 = bit_err_rate(bits_i, decoded_user_bits_arr[6])
    ber_8 = bit_err_rate(bits_i, decoded_user_bits_arr[7])
    ber_9 = bit_err_rate(bits_i, decoded_user_bits_arr[8])
    ber_10 = bit_err_rate(bits_i, decoded_user_bits_arr[9])
    ber_11 = bit_err_rate(bits_i, decoded_user_bits_arr[10])
    # ber_12 = bit_err_rate(bits_i, decoded_user_bits_arr[11])
    # ber_13 = bit_err_rate(bits_i, decoded_user_bits_arr[12])
    # ber_14 = bit_err_rate(bits_i, decoded_user_bits_arr[13])
    # ber_15 = bit_err_rate(bits_i, decoded_user_bits_arr[15])
    # ber_16 = bit_err_rate(bits_i, decoded_user_bits_arr[15])
    
    return bits_i, ber_1, ber_2, ber_3, ber_4, ber_5, ber_6, ber_7, ber_8, ber_9, ber_10, ber_11#, ber_12, ber_13, ber_14, ber_15, ber_16

def sparc_ldpc_integrated_sim(sparc_params, ldpc_params, lengths, ldpc_bool, decode_params, awgn_var, rand_seed=None): 
    '''
    Simulated encoding, transmitting and decoding 
    '''

    bits_i, total_bits, beta0, x, A     = sparc_ldpc_encode(sparc_params, ldpc_params, lengths, ldpc_bool, rand_seed)
    y                       = awgn_channel(x, awgn_var, rand_seed) # Produces the received vector after the channel 
    bits_o                  = integrated_decoder(y, sparc_params, ldpc_params, decode_params, A)
                                        
    ber = bit_err_rate(bits_i, bits_o)
    
    return bits_i, bits_o, ber

def sparc_ldpc_integrated_posteriors_sim(sparc_params, ldpc_params, lengths, ldpc_bool, decode_params, awgn_var, rand_seed=None): 
    '''
    Simulated encoding, transmitting and decoding 
    '''

    bits_i, total_bits, beta0, x, A     = sparc_ldpc_encode(sparc_params, ldpc_params, lengths, ldpc_bool, rand_seed)
    y                       = awgn_channel(x, awgn_var, rand_seed) # Produces the received vector after the channel 
    bits_o                  = integrated_decoder_posteriors(y, sparc_params, ldpc_params, decode_params, A)
                                        
    ber = bit_err_rate(bits_i, bits_o)
    
    return bits_i, bits_o, ber

def integrated_sim_test(sparc_params, ldpc_params, lengths, ldpc_bool, decode_params, awgn_var, rand_seed=None): 

    bits_i, total_bits, beta0, x, A     = sparc_ldpc_encode(sparc_params, ldpc_params, lengths, ldpc_bool, rand_seed)
    y                       = awgn_channel(x, awgn_var, rand_seed) # Produces the received vector after the channel 
    decoded_user_bits_arr             = integrated_decoder_test_2(y, sparc_params, ldpc_params, decode_params, lengths, A)

    ber_1 = bit_err_rate(bits_i, decoded_user_bits_arr[0])
    ber_2 = bit_err_rate(bits_i, decoded_user_bits_arr[1])
    ber_3 = bit_err_rate(bits_i, decoded_user_bits_arr[2])
    ber_4 = bit_err_rate(bits_i, decoded_user_bits_arr[3])
    ber_5 = bit_err_rate(bits_i, decoded_user_bits_arr[4])
    ber_6 = bit_err_rate(bits_i, decoded_user_bits_arr[5])
    ber_7 = bit_err_rate(bits_i, decoded_user_bits_arr[6])
    ber_8 = bit_err_rate(bits_i, decoded_user_bits_arr[7])
    ber_9 = bit_err_rate(bits_i, decoded_user_bits_arr[8])
    ber_10 = bit_err_rate(bits_i, decoded_user_bits_arr[9])
    ber_11 = bit_err_rate(bits_i, decoded_user_bits_arr[10])
    
    return bits_i, ber_1, ber_2, ber_3, ber_4, ber_5, ber_6, ber_7, ber_8, ber_9, ber_10, ber_11

def sparc_ldpc_integrated_sim_test_2(sparc_params, ldpc_params, lengths, ldpc_bool, decode_params, awgn_var, rand_seed=None): 
    '''
    Simulated encoding, transmitting and decoding 
    '''

    bits_i, total_bits, beta0, x, A     = sparc_ldpc_encode(sparc_params, ldpc_params, lengths, ldpc_bool, rand_seed)
    y                       = awgn_channel(x, awgn_var, rand_seed) # Produces the received vector after the channel 
    bits_o                  = integrated_decoder_naive_test(y, sparc_params, ldpc_params, decode_params, A)
                                        
    ber = bit_err_rate(bits_i, bits_o)
    
    return bits_i, bits_o, ber

def sparc_ldpc_integrated_no_bp(sparc_params, ldpc_params, lengths, ldpc_bool, decode_params, awgn_var, rand_seed=None): 
    
    bits_i, total_bits, beta0, x, A     = sparc_ldpc_encode(sparc_params, ldpc_params, lengths, ldpc_bool, rand_seed)
    y                       = awgn_channel(x, awgn_var, rand_seed) # Produces the received vector after the channel 
    hard_decision_bits, ldpc_bits_out                  = naively_integrated_test(y, sparc_params, ldpc_params, decode_params, A)
                                        
    ber = bit_err_rate(total_bits, ldpc_bits_out)
    
    return bits_i, ldpc_bits_out, ber

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

