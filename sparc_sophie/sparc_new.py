import sys
import os

# Get the parent directory of the current file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
from ldpc_jossy.py.ldpc import code 

### Main encode/decode functions -------------------------------------------------------------------------------------------------------------
''' Each function is either an decode or a test. Tests either have the same functionality with different code to prove correctness
    or outputs an additional object like a probe. They have been copied from a decode func so not to change the original code. '''

def sparc_ldpc_encode(sparc_params, ldpc_params, lengths, ldpc_bool, rand_seed): 
    ''' 
    Encodes random message vector to an LDPC codeword then SPARC codeword.
    ''' 

    # Calculated user bits length 
    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    logM = int(np.log2(M))
    if (ldpc_bool): 
        L_unprotected, k_ldpc, mults = lengths['L_unprotected'], lengths['k_ldpc'], lengths['mults']
        unprotected_bit_len = int(L_unprotected*logM)
        user_bits_len = int(k_ldpc + unprotected_bit_len)
    else: 
        user_bits_len = L*logM

    # Generate user bits 
    rng = np.random.default_rng(rand_seed)
    user_bits = rng.integers(0, 2, size=user_bits_len)

    # LDPC code 
    if (ldpc_bool): 
        total_bits = encode_ldpc(user_bits, ldpc_params, mults, unprotected_bit_len)
    else: 
        total_bits = user_bits.astype(bool)

    encoded_bit_len = total_bits.size
    assert encoded_bit_len == L*logM

    # Convert bits to message vector
    n = int(encoded_bit_len/R)
    P_l = P/L
    beta0 = bin_arr_2_msg_vector(total_bits, M, n, P_l)

    A = create_design_matrix(L, M, n, rand_seed) 
    x = np.dot(A, beta0)

    return user_bits, total_bits, beta0, x, A

def sparc_ldpc_decode(y, sparc_params, ldpc_params, decode_params, ldpc_bool, lengths, A):
    '''
    Decodes using AMP fully then BP fully if LDPC.
    '''
    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    sqrt_nP_l = np.sqrt(n * (P/L))

    # Run AMP fully 
    beta_soft_estimate, s = sparc_amp(y, sparc_params, decode_params, A)
    
    if (ldpc_bool): 
        c = code(ldpc_params["standard"], ldpc_params["rate"], ldpc_params["z"])
        L_unprotected = lengths['L_unprotected']
        unprotected_sparse_len = int(L_unprotected*M)
        protected_beta_estimate   = beta_soft_estimate[unprotected_sparse_len:]

        unprotected = msg_vector_map_estimator(s, M, sqrt_nP_l)[:unprotected_sparse_len]
        unprotected_bits_out = msg_vector_2_bin_arr(unprotected, M)

        bp_probs = beta_estimate_to_bp_probs(protected_beta_estimate, L, M, sqrt_nP_l)
        _, protected_bits_out = ldpc_bp(bp_probs, c, 200, True)
    
        bits_out = np.concatenate((unprotected_bits_out, protected_bits_out))

    else: 
        unprotected = msg_vector_map_estimator(s, M, sqrt_nP_l)
        bits_out = msg_vector_2_bin_arr(unprotected, M)

    return bits_out

def sparc_ldpc_decode_loop(y, sparc_params, ldpc_params, decode_params, A):
    '''
    Decodes using AMP fully then BP fully if LDPC then AMP again .
    '''
    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    sqrt_nP_l = np.sqrt(n * (P/L))

    # Run AMP fully 
    beta_soft_estimate, s = sparc_amp(y, sparc_params, decode_params, A)
    bp_probs = beta_estimate_to_bp_probs(beta_soft_estimate, L, M, sqrt_nP_l)
    c = code(ldpc_params["standard"], ldpc_params["rate"], ldpc_params["z"])
    ldpc_probs, _ = ldpc_bp(bp_probs, c, 200, False)
    post_bp_beta = bp_output_to_beta_estimate(ldpc_probs, L, M, sqrt_nP_l)
    new_y = np.dot(A, post_bp_beta)
    _, s_2 = sparc_amp_termination(new_y, sparc_params, decode_params, A)

    hard_decoded_beta = msg_vector_map_estimator(s_2, M, sqrt_nP_l)
    ldpc_bits_out = msg_vector_2_bin_arr(hard_decoded_beta, M)

    assert len(ldpc_bits_out) % c.N == 0
    num_blocks = len(ldpc_bits_out) / c.N
    ldpc_bits_out = np.array_split(ldpc_bits_out, num_blocks)
    user_bits = []
    for chunk in ldpc_bits_out: 
        user_bits.append(chunk[c.K])
    user_bits = np.array(user_bits)
    user_bits = user_bits.flatten()

    return user_bits

def sparc_ldpc_decode_test(y, sparc_params, ldpc_params, decode_params, ldpc_bool, A):
    '''
    Does the exact same function as sparc_ldpc_decode but with the sparc_amp_single_it in a loop
    and ldpc_bp split into loops of 5 iterations.
    '''

    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    sqrt_nP_l = np.sqrt(n * (P/L))

    c = code(ldpc_params["standard"], ldpc_params["rate"], ldpc_params["z"])
    t_max = decode_params['t_max']

    # AMP
    beta = np.zeros(L*M)
    z = 0
    tau_sqr = 1
    AT = A.T

    for i in range(t_max): 
        beta, z, tau_sqr = sparc_amp_single_it(sparc_params, y, A, AT, beta, z, tau_sqr)

    # BP 
    ldpc_probs = beta_estimate_to_bp_probs(beta, L, M, sqrt_nP_l)
    eps = 1e-15  # Small constant to prevent log(0)
    ldpc_probs = np.clip(ldpc_probs, eps, 1 - eps)
    LLR = np.log(ldpc_probs) - np.log(1 - ldpc_probs)

    num_blocks = len(LLR) / c.N
    LLR = np.array_split(LLR, num_blocks)
    app_cut = []
    for i in range(40): 
        for chunk in range(len(LLR)): 
            LLR[chunk] = c.decode(LLR[chunk], 5)[0]

    for chunk in LLR: 
        decoded = c.decode(chunk, 5)[0]
        app_cut.append(decoded[:c.K])
    app_cut = np.array(app_cut)
    app_cut = app_cut.flatten()

    hard_bools = (app_cut < 0)
    hard_decision_bits = np.array([int(bool_val) for bool_val in hard_bools])

    return hard_decision_bits

def sparc_ldpc_decode_test_2(y, sparc_params, ldpc_params, decode_params, ldpc_bool, A):
    '''
    Attempt to make it more similar to the normal naively integrated one. Similar to sparc_ldpc_decode_test, has a loop for
    amp_single_it and for bp but takes the soft bp output turns into a beta estimate and decides the user bits from here. 
    '''
    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    sqrt_nP_l = np.sqrt(n * (P/L))

    c = code(ldpc_params["standard"], ldpc_params["rate"], ldpc_params["z"])
    t_max = decode_params['t_max']

    # AMP
    beta = np.zeros(L*M)
    z = 0
    tau_sqr = 1
    AT = A.T

    for i in range(t_max): 
        beta, z, tau_sqr = sparc_amp_single_it(sparc_params, y, A, AT, beta, z, tau_sqr)

    # BP 
    ldpc_probs = beta_estimate_to_bp_probs(beta, L, M, sqrt_nP_l)
    eps = 1e-15  # Small constant to prevent log(0)
    ldpc_probs = np.clip(ldpc_probs, eps, 1 - eps)
    LLR = np.log(ldpc_probs) - np.log(1 - ldpc_probs)

    num_blocks = len(LLR) / c.N
    LLR = np.array_split(LLR, num_blocks)
    app = []
    for i in range(40): 
        for chunk in range(len(LLR)): 
            LLR[chunk] = c.decode(LLR[chunk], 5)[0]

    app = np.array(LLR)
    app = app.flatten()

    ldpc_probs = (np.exp(app))/(1+np.exp(app))
    beta_estimate = bp_output_to_beta_estimate(ldpc_probs, L, M, sqrt_nP_l)
    beta = msg_vector_map_estimator(beta_estimate, M, sqrt_nP_l)
    ldpc_bits_out = msg_vector_2_bin_arr(beta, M)
    ldpc_bits_split = np.array_split(ldpc_bits_out, num_blocks)
    bits_out = []
    for chunk in ldpc_bits_split: 
        bits_out.append(chunk[:c.K])

    bits_out = np.array(bits_out)

    return bits_out

def sparc_ldpc_decode_test_3(y, sparc_params, ldpc_params, decode_params, ldpc_bool, A): 
    '''
    Same as sparc_ldpc_decode but also does no BP, it just uses the fact that the ldpc code is systematic.
    '''
    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    sqrt_nP_l = np.sqrt(n * (P/L))

    # Run AMP fully 
    beta_soft_estimate, s = sparc_amp(y, sparc_params, decode_params, A)
    
    c = code(ldpc_params["standard"], ldpc_params["rate"], ldpc_params["z"])

    bp_probs = beta_estimate_to_bp_probs(beta_soft_estimate, L, M, sqrt_nP_l)
    _, bits_out, no_bp_bits = ldpc_bp_test(bp_probs, c, 200, True)


    return bits_out, no_bp_bits

def no_onsager_decoder(y, sparc_params, ldpc_params, decode_params, A): 
    '''
    Combines the decoders but doesn't use an Onsager term to correct dependencies just to see the effect. 
    '''

    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    sqrt_nP_l = np.sqrt(n * (P/L))
    c = code(ldpc_params["standard"], ldpc_params["rate"], ldpc_params["z"])
    t_max = decode_params['t_max']

    # Initial AMP + BP
    beta = np.zeros(L*M)
    z = 0
    tau_sqr = 1
    AT = A.T
    for i in range(t_max): 
        beta, z, tau_sqr = amp_no_onsager(sparc_params, y, A, AT, beta, z, tau_sqr)
        ldpc_probs = beta_estimate_to_bp_probs(beta, L, M, sqrt_nP_l)
        if (i != t_max-1): 
            ldpc_probs, _ = ldpc_bp(ldpc_probs, c, 6, False)
            beta = bp_output_to_beta_estimate(ldpc_probs, L, M, sqrt_nP_l)
        else: 
            _, hard_decision_bits = ldpc_bp(ldpc_probs, c, 200, True)

    return hard_decision_bits

def naively_integrated_decoder(y, sparc_params, ldpc_params, decode_params, A): 
    '''
    Aims to combine AMP with BP. AMP single iteration followed by x bp iterations i times. Finished by f bp iterations.
    '''

    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    sqrt_nP_l = np.sqrt(n * (P/L))
    c = code(ldpc_params["standard"], ldpc_params["rate"], ldpc_params["z"])
    t_max = decode_params['t_max']

    # Initial AMP + BP
    beta = np.zeros(L*M)
    z = 0
    tau_sqr = 1
    AT = A.T
    for i in range(t_max): 
        beta, z, tau_sqr = sparc_amp_single_it(sparc_params, y, A, AT, beta, z, tau_sqr)
        ldpc_probs = beta_estimate_to_bp_probs(beta, L, M, sqrt_nP_l)
        if (i != t_max-1): 
            ldpc_probs, _ = ldpc_bp(ldpc_probs, c, 6, False)
            beta = bp_output_to_beta_estimate(ldpc_probs, L, M, sqrt_nP_l)
        else: 
            _, hard_decision_bits = ldpc_bp(ldpc_probs, c, 200, True)

    return hard_decision_bits

def naively_integrated_test(y, sparc_params, ldpc_params, decode_params, A): 
    '''
    Originally only the decided user bits are given, here there is also the option of no final round of BP, the beta estimate is
    used to decode and produce the ldpc bits. 
    '''
    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    sqrt_nP_l = np.sqrt(n * (P/L))
    c = code(ldpc_params["standard"], ldpc_params["rate"], ldpc_params["z"])
    t_max = decode_params['t_max']

    # Initial AMP + BP
    beta = np.zeros(L*M)
    z = 0
    tau_sqr = 1
    AT = A.T
    for i in range(t_max): 
        beta, z, tau_sqr = sparc_amp_single_it(sparc_params, y, A, AT, beta, z, tau_sqr)
        ldpc_probs = beta_estimate_to_bp_probs(beta, L, M, sqrt_nP_l)
        if (i != t_max-1): 
            ldpc_probs, _ = ldpc_bp(ldpc_probs, c, 5, False)
            beta = bp_output_to_beta_estimate(ldpc_probs, L, M, sqrt_nP_l)
        else: 
            _, hard_decision_bits = ldpc_bp(ldpc_probs, c, 200, True)

    beta_decision = msg_vector_map_estimator(beta, M, sqrt_nP_l)
    ldpc_bits_out = msg_vector_2_bin_arr(beta_decision, M)

    return hard_decision_bits, ldpc_bits_out

def naively_integrated_test_2(y, sparc_params, ldpc_params, decode_params, lengths, A):
    '''
    Outputs the decoded user bits before and after BP in specified iterations of AMP.
    '''
    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    sqrt_nP_l = np.sqrt(n * (P/L))
    c = code(ldpc_params["standard"], ldpc_params["rate"], ldpc_params["z"])   
    t_max = decode_params['t_max']

    mults = lengths['mults']
    user_bits_len = c.K * mults
    decoded_user_bits_arr = np.zeros((6,user_bits_len))

    # Initial AMP + BP
    beta = np.zeros(L*M)
    z = 0
    tau_sqr = 1
    AT = A.T
    store_idx = 0
    for i in range(t_max): 
        beta, z, tau_sqr = sparc_amp_single_it(sparc_params, y, A, AT, beta, z, tau_sqr)
        ldpc_probs = beta_estimate_to_bp_probs(beta, L, M, sqrt_nP_l)
        if i in [10,11,12]: 
            decoded_user_bits_arr[store_idx] = ldpc_probs_to_user_bits(ldpc_probs, c)
            store_idx += 1
        ldpc_probs, _ = ldpc_bp(ldpc_probs, c, 6, False)
        if i in [10,11,12]:
            decoded_user_bits_arr[store_idx] = ldpc_probs_to_user_bits(ldpc_probs, c)
            store_idx += 1
        beta = bp_output_to_beta_estimate(ldpc_probs, L, M, sqrt_nP_l)

    return decoded_user_bits_arr

def naively_integrated_test_3(y, sparc_params, ldpc_params, decode_params, lengths, A):
    '''
    Outputs the decoded user bits before MMSE, before BP (After MMSe), and after BP in  2 specified iterations of AMP.
    '''
    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    sqrt_nP_l = np.sqrt(n * (P/L))
    c = code(ldpc_params["standard"], ldpc_params["rate"], ldpc_params["z"])   
    t_max = decode_params['t_max']

    mults = lengths['mults']
    user_bits_len = c.K * mults
    decoded_user_bits_arr = np.zeros((6,user_bits_len))

    # Initial AMP + BP
    beta = np.zeros(L*M)
    z = 0
    tau_sqr = 1
    AT = A.T
    store_idx = 0
    for i in range(t_max): 
        beta, z, tau_sqr, user_bits_before = sparc_amp_single_it_test(sparc_params, y, A, AT, beta, z, tau_sqr, c)
        ldpc_probs = beta_estimate_to_bp_probs(beta, L, M, sqrt_nP_l)
        if i in [9,10]: 
            decoded_user_bits_arr[store_idx] = user_bits_before
            decoded_user_bits_arr[store_idx+1] = ldpc_probs_to_user_bits(ldpc_probs, c)
        ldpc_probs, _ = ldpc_bp(ldpc_probs, c, 6, False)
        if i in [9,10]:
            decoded_user_bits_arr[store_idx+2] = ldpc_probs_to_user_bits(ldpc_probs, c)
            store_idx += 3
        beta = bp_output_to_beta_estimate(ldpc_probs, L, M, sqrt_nP_l)

    return decoded_user_bits_arr

def naively_integrated_test_4(y, sparc_params, ldpc_params, decode_params, lengths, A):
    '''
    Outputs the decoded user bits at each iteration of AMP before BP in specified iterations of AMP.
    '''
    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    sqrt_nP_l = np.sqrt(n * (P/L))
    c = code(ldpc_params["standard"], ldpc_params["rate"], ldpc_params["z"])   
    t_max = decode_params['t_max']

    mults = lengths['mults']
    user_bits_len = c.K * mults
    decoded_user_bits_arr = np.zeros((16,user_bits_len))

    # Initial AMP + BP
    beta = np.zeros(L*M)
    z = 0
    tau_sqr = 1
    AT = A.T
    for i in range(t_max): 
        beta, z, tau_sqr = sparc_amp_single_it(sparc_params, y, A, AT, beta, z, tau_sqr)
        ldpc_probs = beta_estimate_to_bp_probs(beta, L, M, sqrt_nP_l)
        if i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]: 
            decoded_user_bits_arr[i] = ldpc_probs_to_user_bits(ldpc_probs, c)
        ldpc_probs, _ = ldpc_bp(ldpc_probs, c, 6, False)
        beta = bp_output_to_beta_estimate(ldpc_probs, L, M, sqrt_nP_l)

    return decoded_user_bits_arr

def naively_integrated_decoder_posteriors(y, sparc_params, ldpc_params, decode_params, A): 
    '''
    Aims to combine AMP with BP. AMP single iteration followed by x bp iterations i times. Finished by f bp iterations.
    '''

    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    sqrt_nP_l = np.sqrt(n * (P/L))
    c = code(ldpc_params["standard"], ldpc_params["rate"], ldpc_params["z"])
    t_max = decode_params['t_max']

    # Initial AMP + BP
    beta = np.zeros(L*M)
    z = 0
    tau_sqr = 1
    AT = A.T
    for i in range(t_max): 
        beta, z, tau_sqr = sparc_amp_single_it(sparc_params, y, A, AT, beta, z, tau_sqr)
        ldpc_probs = beta_estimate_to_bp_probs(beta, L, M, sqrt_nP_l)
        if (i != t_max-1): 
            ldpc_probs, _ = ldpc_bp(ldpc_probs, c, 6, False)
            old_estimate = bp_output_to_beta_estimate(ldpc_probs, L, M, sqrt_nP_l)
            gamma = old_estimate / sqrt_nP_l
            alpha = beta / sqrt_nP_l
            beta = update_using_bp_probs(gamma, alpha, sqrt_nP_l, M)
        else: 
            _, hard_decision_bits = ldpc_bp(ldpc_probs, c, 200, True)

    return hard_decision_bits

def naively_integrated_decoder_posteriors_test(y, sparc_params, ldpc_params, decode_params, lengths, A): 
    '''
    Aims to combine AMP with BP. AMP single iteration followed by x bp iterations i times. Finished by f bp iterations.
    '''

    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    sqrt_nP_l = np.sqrt(n * (P/L))
    c = code(ldpc_params["standard"], ldpc_params["rate"], ldpc_params["z"])
    t_max = decode_params['t_max']

    mults = lengths['mults']
    user_bits_len = c.K * mults
    decoded_user_bits_arr = np.zeros((11,user_bits_len))

    # Initial AMP + BP
    beta = np.zeros(L*M)
    z = 0
    tau_sqr = 1
    AT = A.T
    for i in range(t_max): 
        beta, z, tau_sqr = sparc_amp_single_it(sparc_params, y, A, AT, beta, z, tau_sqr)
        ldpc_probs = beta_estimate_to_bp_probs(beta, L, M, sqrt_nP_l)
        if i in [0,1,2,3,4,5,6,7,8,9,10]: 
            decoded_user_bits_arr[i] = ldpc_probs_to_user_bits(ldpc_probs, c)
        ldpc_probs, _ = ldpc_bp(ldpc_probs, c, 6, False)
        gamma = bp_output_to_beta_estimate(ldpc_probs, L, M, sqrt_nP_l)
        beta = update_using_bp_probs(gamma, beta, sqrt_nP_l, M)

    return decoded_user_bits_arr

def integrated_decoder(y, sparc_params, ldpc_params, decode_params, A): 
    '''
    Integrated the decoders but with the updated formula. 
    '''
    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    P_l = P/L
    c = code(ldpc_params["standard"], ldpc_params["rate"], ldpc_params["z"])
    t_max = decode_params['t_max']
    num_its = 6
    num_its_final = 200
    S_k = S_k_mapping(M)

    beta = np.zeros(L*M)
    z = 0
    differentiated_eta = np.zeros(n)
    AT = A.T
    for t in range(t_max):  
        if t != 0: 
            differentiated_eta = differentiated_eta_calc(beta, vk, vk_0, alpha, tau_sqr, L, M, S_k, n, P_l)
        z = y - (np.dot(A, beta)) + (z/n)*(np.sum(differentiated_eta))
        s = np.dot(AT, z) + beta 
        tau_sqr = np.sum(z ** 2) / n    
        if (t != t_max-1):
            hard_decision = False
            alpha, vk_0, vk, beta, _ = eta(s, tau_sqr, n, P_l, M, L, c, num_its, num_its_final, hard_decision)
        else: 
            hard_decision = True 
            _, _, _, _, hard_decision_bits= eta(s, tau_sqr, n, P_l, M, L, c, num_its, num_its_final, hard_decision)

    return hard_decision_bits

def integrated_decoder_naive_test(y, sparc_params, ldpc_params, decode_params, A): 
    '''
    Integrated the decoders but with the updated formula. 
    '''
    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    P_l = P/L
    c = code(ldpc_params["standard"], ldpc_params["rate"], ldpc_params["z"])
    t_max = decode_params['t_max']
    num_its = 6
    num_its_final = 200
    S_k = S_k_mapping(M)

    beta = np.zeros(L*M)
    z = 0
    onsager_term = np.zeros(n)
    AT = A.T
    for t in range(t_max):  
        if t != 0: 
            onsager_term = (z / tau_sqr) * (P - ((np.sum(beta ** 2))/n) )
        z = y - (np.dot(A, beta)) + onsager_term
        s = np.dot(AT, z) + beta 
        tau_sqr = np.sum(z ** 2) / n    
        if (t != t_max-1):
            hard_decision = False
            alpha, vk_0, vk, beta, _ = eta(s, tau_sqr, n, P_l, M, L, c, num_its, num_its_final, hard_decision)
        else: 
            hard_decision = True 
            _, _, _, _, hard_decision_bits= eta(s, tau_sqr, n, P_l, M, L, c, num_its, num_its_final, hard_decision)

    return hard_decision_bits

def integrated_decoder_test_0(y, sparc_params, ldpc_params, decode_params, A): 
    '''
    Same as naively_integrated_test, this function also has the option of skipping final BP and using the beta estimate
    to get the decoded ldpc_bits. 
    '''
    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    P_l = P/L
    sqrt_nP_l = np.sqrt(n*P_l)
    c = code(ldpc_params["standard"], ldpc_params["rate"], ldpc_params["z"])
    t_max = decode_params['t_max']
    num_its = 5
    num_its_final = 5
    S_k = S_k_mapping(M)

    beta = np.zeros(L*M)
    z = 0
    differentiated_eta = np.zeros(n)
    AT = A.T
    for t in range(t_max):  
        if t != 0: 
            differentiated_eta = differentiated_eta_calc(beta, vk, vk_0, alpha, tau_sqr, L, M, S_k, n, P_l)
        z = y - (np.dot(A, beta)) + z*(np.sum(differentiated_eta)/n)
        s = np.dot(AT, z) + beta 
        tau_sqr = np.sum(z ** 2) / n    
        if (t != t_max-1):
            hard_decision = False
            alpha, vk_0, vk, beta, _ = eta(s, tau_sqr, n, P_l, M, L, c, num_its, num_its_final, hard_decision)
        else: 
            hard_decision = True 
            _, _, _, _, hard_decision_bits= eta(s, tau_sqr, n, P_l, M, L, c, num_its, num_its_final, hard_decision)

    beta_decision = msg_vector_map_estimator(beta, M, sqrt_nP_l)
    ldpc_bits_out = msg_vector_2_bin_arr(beta_decision, M)

    return hard_decision_bits, ldpc_bits_out

def integrated_decoder_test(y, sparc_params, ldpc_params, decode_params, lengths, A): 
    '''
    Outputs the decoded user bits before and after mmse estimation after BP for any two
    consecutive AMP iteraions. 
    '''
    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    P_l = P/L
    c = code(ldpc_params["standard"], ldpc_params["rate"], ldpc_params["z"])
    t_max = decode_params['t_max']
    num_its = 5
    S_k = S_k_mapping(M)

    mults = lengths['mults']
    user_bits_len = c.K * mults
    decoded_user_bits_arr = np.zeros((6,user_bits_len))
    store_idx = 0 

    beta = np.zeros(L*M)
    z = 0
    differentiated_eta = np.zeros(n)
    AT = A.T
    for t in range(t_max):  
        if t != 0: 
            differentiated_eta = differentiated_eta_calc(beta, vk, vk_0, alpha, tau_sqr, L, M, S_k, n, P_l)
        z = y - (np.dot(A, beta)) + z*(np.sum(differentiated_eta)/n)
        s = np.dot(AT, z) + beta 
        tau_sqr = np.sum(z ** 2) / n    
        alpha, vk_0, vk, beta, user_bits_before, user_bits_middle, user_bits_after = eta_test(s, tau_sqr, n, P_l, M, L, c, num_its)
        if t in [4, 5]: 
            decoded_user_bits_arr[store_idx] = user_bits_before
            decoded_user_bits_arr[store_idx+1] = user_bits_middle
            decoded_user_bits_arr[store_idx+2] = user_bits_after
            store_idx += 3 

    return decoded_user_bits_arr
    
def integrated_decoder_test_2(y, sparc_params, ldpc_params, decode_params, lengths, A): 
    '''
    Outputs the decoded user bits before each AMP iteration for specified iterations
    '''
    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    P_l = P/L
    c = code(ldpc_params["standard"], ldpc_params["rate"], ldpc_params["z"])
    t_max = decode_params['t_max']
    num_its = 5
    S_k = S_k_mapping(M)

    mults = lengths['mults']
    user_bits_len = c.K * mults
    decoded_user_bits_arr = np.zeros((11,user_bits_len))

    beta = np.zeros(L*M)
    z = 0
    differentiated_eta = np.zeros(n)
    AT = A.T
    for t in range(t_max):  
        if t != 0: 
            differentiated_eta = differentiated_eta_calc(beta, vk, vk_0, alpha, tau_sqr, L, M, S_k, n, P_l)
        z = y - (np.dot(A, beta)) + z*(np.sum(differentiated_eta)/n)
        s = np.dot(AT, z) + beta 
        tau_sqr = np.sum(z ** 2) / n    
        alpha, vk_0, vk, beta, user_bits_before, user_bits_middle, user_bits_after = eta_test(s, tau_sqr, n, P_l, M, L, c, num_its)
        # if t in [0,1,2,3,4,5,6,7,8,9,10]: 
        if t in [11,12,13,14,15,16,17,18,19,20,21]: 
            decoded_user_bits_arr[t-11] = user_bits_before

    return decoded_user_bits_arr

def integrated_decoder_test_3(y, sparc_params, ldpc_params, decode_params, A): 
    '''
    Integrated decoder but with no bp  
    '''
    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    P_l = P/L
    c = code(ldpc_params["standard"], ldpc_params["rate"], ldpc_params["z"])
    t_max = decode_params['t_max']
    num_its = 5
    num_its_final = 200
    S_k = S_k_mapping(M)

    beta = np.zeros(L*M)
    z = 0
    differentiated_eta = np.zeros(n)
    AT = A.T
    for t in range(t_max+1):  
        if t != 0: 
            differentiated_eta = differentiated_eta_calc(beta, vk, vk_0, alpha, tau_sqr, L, M, S_k, n, P_l)
        z = y - (np.dot(A, beta)) + z*(np.sum(differentiated_eta)/n)
        s = np.dot(AT, z) + beta 
        tau_sqr = np.sum(z ** 2) / n    
        if (t != t_max-1):
            hard_decision = False
            alpha, vk_0, vk, beta, _ = eta_test_2(s, tau_sqr, n, P_l, M, L, c, num_its, num_its_final, hard_decision)
        else: 
            hard_decision = True 
            _, _, _, _, hard_decision_bits= eta_test_2(s, tau_sqr, n, P_l, M, L, c, num_its, num_its_final, hard_decision)

    return hard_decision_bits

def integrated_decoder_posteriors(y, sparc_params, ldpc_params, decode_params, A): 
    '''
    Integrated the decoders but with the updated formula using the new method of improving not replacing beta from bp probs. 
    '''
    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    P_l = P/L
    c = code(ldpc_params["standard"], ldpc_params["rate"], ldpc_params["z"])
    t_max = decode_params['t_max']
    num_its = 6
    num_its_final = 200
    S_k = S_k_mapping(M)

    beta = np.zeros(L*M)
    z = 0
    differentiated_eta = np.zeros(n)
    AT = A.T
    for t in range(t_max):  
        if t != 0: 
            differentiated_eta = differentiated_eta_calc_posteriors(gamma, beta, vk, vk_0, alpha, tau_sqr, L, M, S_k, n, P_l)
        z = y - (np.dot(A, beta)) + z*(np.sum(differentiated_eta)/n)
        s = np.dot(AT, z) + beta 
        tau_sqr = np.sum(z ** 2) / n    
        if (t != t_max-1):
            hard_decision = False
            alpha, vk_0, vk, beta, gamma, _ = eta_posteriors(s, tau_sqr, n, P_l, M, L, c, num_its, num_its_final, hard_decision)
        else: 
            hard_decision = True 
            _, _, _, _, _, hard_decision_bits= eta_posteriors(s, tau_sqr, n, P_l, M, L, c, num_its, num_its_final, hard_decision)

    return hard_decision_bits

######## AMP + BP decoder ######## --------------------------------------------------------------------------------------------------------------

def eta(s, tau_sqr, n, P_l, M, L, c, num_its, num_its_final, hard_decision_bool): 
    #(Gone over)
    sqrt_nP_l = np.sqrt(n*P_l)

    # Step One (Expectation): 
    weighted_alpha = msg_vector_mmse_estimator(s, tau_sqr, n, P_l, M)
    alpha = weighted_alpha/sqrt_nP_l
    assert len(weighted_alpha) == L*M

    # Step Two (Conversion to codeword bit-wise probabilites): 
    vk_0 = beta_estimate_to_bp_probs(weighted_alpha, L, M, sqrt_nP_l)
    assert len(vk_0) == L * np.log2(M)

    if hard_decision_bool: 
        # Step Three (Belief Propagation): 
        vk, hard_decision_bits = ldpc_bp(vk_0, c, num_its_final, hard_decision_bool)

        # Step Four (Conversion to beta section-wise probabilites): 
        beta = np.zeros(L*M)
    else: 
        # Step Three (Belief Propagation): 
        vk, hard_decision_bits = ldpc_bp(vk_0, c, num_its, hard_decision_bool)
        # Step Four (Conversion to beta section-wise probabilites): 
        beta = bp_output_to_beta_estimate(vk, L, M, sqrt_nP_l)
    assert len(beta) == L*M
    
    return alpha, vk_0, vk, beta, hard_decision_bits

def eta_test(s, tau_sqr, n, P_l, M, L, c, num_its): 
    
    sqrt_nP_l = np.sqrt(n*P_l)

    # Step One (Expectation): 
    decoded_beta = msg_vector_map_estimator(s, M, sqrt_nP_l)
    decoded_ldpc_bits = msg_vector_2_bin_arr(decoded_beta, M)
    user_bits_before = ldpc_bits_to_user_bits(decoded_ldpc_bits, c)

    weighted_alpha = msg_vector_mmse_estimator(s, tau_sqr, n, P_l, M)
    alpha = weighted_alpha/sqrt_nP_l
    assert len(weighted_alpha) == L*M

    # Step Two (Conversion to codeword bit-wise probabilites): 
    vk_0 = beta_estimate_to_bp_probs(weighted_alpha, L, M, sqrt_nP_l)
    user_bits_middle = ldpc_probs_to_user_bits(vk_0, c)
    assert len(vk_0) == L * np.log2(M)

    # Step Three (Belief Propagation): 
    vk, _ = ldpc_bp(vk_0, c, num_its, False)
    user_bits_after = ldpc_probs_to_user_bits(vk, c)
    # Step Four (Conversion to beta section-wise probabilites): 
    beta = bp_output_to_beta_estimate(vk, L, M, sqrt_nP_l)
    assert len(beta) == L*M
    
    return alpha, vk_0, vk, beta, user_bits_before, user_bits_middle, user_bits_after

def eta_test_2(s, tau_sqr, n, P_l, M, L, c, num_its, num_its_final, hard_decision_bool): 

    sqrt_nP_l = np.sqrt(n*P_l)

    # Step One (Expectation): 
    weighted_alpha = msg_vector_mmse_estimator(s, tau_sqr, n, P_l, M)
    alpha = weighted_alpha/sqrt_nP_l
    assert len(weighted_alpha) == L*M

    # Step Two (Conversion to codeword bit-wise probabilites): 
    vk_0 = beta_estimate_to_bp_probs(weighted_alpha, L, M, sqrt_nP_l)
    assert len(vk_0) == L * np.log2(M)

    if hard_decision_bool: 
        # Step Three (Belief Propagation): 
        vk, hard_decision_bits = ldpc_bp(vk_0, c, num_its_final, hard_decision_bool)

        # Step Four (Conversion to beta section-wise probabilites): 
        beta = np.zeros(L*M)
    else: 
        # Step Three (Belief Propagation): 
        vk = vk_0
        hard_decision_bits = 0 
        # Step Four (Conversion to beta section-wise probabilites): 
        beta = bp_output_to_beta_estimate(vk, L, M, sqrt_nP_l)
    assert len(beta) == L*M
    
    return alpha, vk_0, vk, beta, hard_decision_bits

def eta_posteriors(s, tau_sqr, n, P_l, M, L, c, num_its, num_its_final, hard_decision_bool): 

    sqrt_nP_l = np.sqrt(n*P_l)

    # Step One (Expectation): 
    weighted_alpha = msg_vector_mmse_estimator(s, tau_sqr, n, P_l, M)
    alpha = weighted_alpha/sqrt_nP_l
    assert len(weighted_alpha) == L*M

    # Step Two (Conversion to codeword bit-wise probabilites): 
    vk_0 = beta_estimate_to_bp_probs(weighted_alpha, L, M, sqrt_nP_l)
    assert len(vk_0) == L * np.log2(M)

    if hard_decision_bool: 
        # Step Three (Belief Propagation): 
        vk, hard_decision_bits = ldpc_bp(vk_0, c, num_its_final, hard_decision_bool)

        # Step Four (Conversion to beta section-wise probabilites): 
        beta = np.zeros(L*M)
        gamma = np.zeros(L*M)
    else: 
        # Step Three (Belief Propagation): 
        vk, hard_decision_bits = ldpc_bp(vk_0, c, num_its, hard_decision_bool)
        # Step Four (Conversion to beta section-wise probabilites): 
        old_estimate = bp_output_to_beta_estimate(vk, L, M, sqrt_nP_l)
        gamma = old_estimate / sqrt_nP_l
        beta = update_using_bp_probs(gamma, alpha, sqrt_nP_l, M)
    assert len(beta) == L*M
    
    return alpha, vk_0, vk, beta, gamma, hard_decision_bits

def differentiated_eta_calc(beta, vk, vk_0, alpha, tau_sqr, L, M, S_k, n, P_l): 
    #(Gone over --- Seems to be taking really long)
    logM = int(np.log2(M))
    vk_sectioned = vk.reshape(L,logM)
    alpha_sectioned = alpha.reshape(L,M)
    main_term = np.zeros((L,M))
    for l in range(L): 
        for i in range(M): 
            binary_num = format(i,f"0{logM}b")
            for k in range(logM): 
                if (binary_num[k] == '1'): 
                    main_term[l][i] -= vk_sectioned[l][k] * sub_term(vk_0, alpha_sectioned, tau_sqr, l, k, i, S_k, n, P_l, L, M)
                else: 
                    main_term[l][i] += (1 - vk_sectioned[l][k]) * sub_term(vk_0, alpha_sectioned, tau_sqr, l, k, i, S_k, n, P_l, L, M)

    main_term = main_term.reshape(L*M)

    return beta * main_term 

def differentiated_eta_calc_posteriors(gamma, beta, vk, vk_0, alpha, tau_sqr, L, M, S_k, n, P_l): 

    sqrt_nP_l = np.sqrt(n*P_l)
    logM = int(np.log2(M))
    vk_sectioned = vk.reshape(L,logM)
    alpha_sectioned = alpha.reshape(L,M)
    main_term = np.zeros((L,M))
    for l in range(L): 
        for i in range(M): 
            binary_num = format(i,f"0{logM}b")
            for k in range(logM): 
                if (binary_num[k] == '1'): 
                    main_term[l][i] -= vk_sectioned[l][k] * sub_term(vk_0, alpha_sectioned, tau_sqr, l, k, i, S_k, n, P_l, L, M)
                else: 
                    main_term[l][i] += (1 - vk_sectioned[l][k]) * sub_term(vk_0, alpha_sectioned, tau_sqr, l, k, i, S_k, n, P_l, L, M)

    main_term = main_term.reshape(L*M)

    alpha_dash = alpha * (sqrt_nP_l / tau_sqr) * (1 - alpha)
    gamma_dash = gamma * main_term 
    top = alpha * gamma 
    bot = top.reshape(-1, M).sum(axis=1).repeat(M)
    top_dash = (alpha_dash*gamma)+(alpha*gamma_dash)
    bot_dash = top_dash.reshape(-1, M).sum(axis=1).repeat(M)
    eta_dash = (sqrt_nP_l * ((top_dash*bot) - (top*bot_dash))) / (bot**2)

    return eta_dash 

def sub_term(vk_0, alpha_sectioned, tau_sqr, l, k, i, S_k, n, P_l, L, M): 
    sum_term = 0
    q_to_sum_over = S_k[k]
    for q in q_to_sum_over: 
        if q == i: 
            sum_term += alpha_sectioned[l][q] * (np.sqrt(n*P_l)/tau_sqr) * (1 - alpha_sectioned[l][q])
        else: 
            sum_term += alpha_sectioned[l][q] * (np.sqrt(n*P_l)/tau_sqr) * (-alpha_sectioned[l][i])  # Changed from q to i here
        
    vk_0_sectioned = vk_0.reshape(L,int(np.log2(M)))
    val = np.clip(vk_0_sectioned[l][k], 1e-10, 1 - 1e-10)

    return (1 / (val * (1 - val))) * sum_term

def sparc_amp(y, sparc_params, decode_params, A): 
    '''
    The normal AMP algorithm.
    ''' #(Tested)
    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    P_l = P / L
    AT = A.T

    t_max = decode_params['t_max']

    # Initialise variables 
    beta = np.zeros(L*M)
    z = y
    atol = 2*np.finfo(np.float64).resolution

    for t in range(t_max): 
        if t > 0:
            Ab = np.dot(A, beta)
            Onsager = (z / tau_sqr) * (P - ((np.sum(beta ** 2))/n) )
            z = y - Ab + Onsager 
        
        ATz = np.dot(AT, z)
        s = beta + ATz
        tau_sqr = np.sum(z ** 2) / n 
        beta = msg_vector_mmse_estimator(s, tau_sqr, n, P_l, M)

    return beta, s 

def sparc_amp_loop(y, beta, sparc_params, decode_params, A): 
    '''
    The normal AMP algorithm.
    ''' #(Tested)
    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    P_l = P / L
    AT = A.T

    t_max = decode_params['t_max']

    # Initialise variables 
    z = y
    atol = 2*np.finfo(np.float64).resolution

    for t in range(t_max): 
        if t > 0:
            Ab = np.dot(A, beta)
            Onsager = (z / tau_sqr) * (P - ((np.sum(beta ** 2))/n) )
            z = y - Ab + Onsager 
        
        ATz = np.dot(AT, z)
        s = beta + ATz
        tau_sqr = np.sum(z ** 2) / n 
        beta = msg_vector_mmse_estimator(s, tau_sqr, n, P_l, M)

    return beta, s

def sparc_amp_termination(y, sparc_params, decode_params, A): 
    '''
    The normal AMP algorithm.
    ''' #(Tested)
    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    P_l = P / L
    AT = A.T

    t_max = decode_params['t_max']

    # Initialise variables 
    beta = np.zeros(L*M)
    z = y
    atol = 2*np.finfo(np.float64).resolution

    for t in range(t_max): 
        if t > 0:
            Ab = np.dot(A, beta)
            Onsager = (z / tau_sqr) * (P - ((np.sum(beta ** 2))/n) )
            z = y - Ab + Onsager 
        
        ATz = np.dot(AT, z)
        s = beta + ATz
        tau_sqr = np.sum(z ** 2) / n 
        beta_termination = beta 
        beta, termination = msg_vector_mmse_estimator_termination(s, tau_sqr, n, P_l, M)
        if termination: 
            beta = beta_termination
            break

    return beta, s 

def sparc_amp_single_it(sparc_params, y, A, AT, beta, z, tau_sqr):
    #(Tested)

    P, L, M = sparc_params['P'], sparc_params['L'], sparc_params['M']
    P_l = P/L
    n = len(y)
    Ab = np.dot(A, beta)
    Onsager = (z / tau_sqr) * (P - ((np.sum(beta ** 2))/n) )
    z = y - Ab + Onsager

    ATz = np.dot(AT, z)
    s = beta + ATz
    tau_sqr = np.sum(z ** 2) / n 
    beta = msg_vector_mmse_estimator(s, tau_sqr, n, P_l, M)

    return beta, z, tau_sqr

def sparc_amp_single_it_test(sparc_params, y, A, AT, beta, z, tau_sqr, c):
    #(Tested)

    P, L, M = sparc_params['P'], sparc_params['L'], sparc_params['M']
    P_l = P/L
    n = len(y)
    Ab = np.dot(A, beta)
    Onsager = (z / tau_sqr) * (P - ((np.sum(beta ** 2))/n) )
    z = y - Ab + Onsager

    ATz = np.dot(AT, z)
    s = beta + ATz
    tau_sqr = np.sum(z ** 2) / n 

    sqrt_nP_l = np.sqrt(n*P_l)
    decoded_beta = msg_vector_map_estimator(s, M, sqrt_nP_l)
    decoded_ldpc_bits = msg_vector_2_bin_arr(decoded_beta, M)
    user_bits_before = ldpc_bits_to_user_bits(decoded_ldpc_bits, c)

    beta = msg_vector_mmse_estimator(s, tau_sqr, n, P_l, M)

    return beta, z, tau_sqr, user_bits_before

def amp_no_onsager(sparc_params, y, A, AT, beta, z, tau_sqr):

    P, L, M = sparc_params['P'], sparc_params['L'], sparc_params['M']
    P_l = P/L
    n = len(y)
    Ab = np.dot(A, beta)
    z = y - Ab

    ATz = np.dot(AT, z)
    s = beta + ATz
    tau_sqr = np.sum(z ** 2) / n 
    beta = msg_vector_mmse_estimator(s, tau_sqr, n, P_l, M)

    return beta, z, tau_sqr

def update_using_bp_probs(gamma, alpha, sqrt_nP_l, M): 
    '''
    Takes the current beta estimate and uses the new beta estimate from BP (gamma) to update it. 
    '''

    top = (alpha)*(gamma)
    bot = top.reshape(-1, M).sum(axis=1).repeat(M)

    return (sqrt_nP_l * (top / bot))

def msg_vector_mmse_estimator(s, tau_sqr, n, P_l, M):
    '''
    MMSE (Bayes optimal) estimator of message vector of SPARC in
    (possibly complex) independent additive Gaussian noise.

    s  : effective observation in Gaussian noise (length L*M vector)
    tau_sqr: the noise variance (length L*M vector)
    L  : number of sections (1 non-zero entry per section)
    M  : number of entries per section
    '''

    # The current method of preventing overflow is:
    # 1) subtract the maximum entry of array x from all entries before taking
    #    np.exp(x). Note: the largest exponent that np.float64 can handle is
    #    roughly 709.
    #
    # Perhaps I can avoid tau becoming too small by changing how I do early
    # termination in the AMP algorithm.

    x   = np.sqrt(n*P_l) * (s / tau_sqr) 
    if (np.any(x-x.max() >= 708)) or (np.any(x-x.max() <= -800)): 
        print("Possible overflow from exponent in mmse_estimator")
    top = np.exp(x - x.max(), dtype=np.float64)
    bot = top.reshape(-1, M).sum(axis=1).repeat(M)

    # Cast back to normal float or complex data types
    return ((np.sqrt(n*P_l)) * (top / bot)).astype(np.float64)

def msg_vector_mmse_estimator_termination(s, tau_sqr, n, P_l, M):
    '''
    MMSE (Bayes optimal) estimator of message vector of SPARC in
    (possibly complex) independent additive Gaussian noise.

    s  : effective observation in Gaussian noise (length L*M vector)
    tau_sqr: the noise variance (length L*M vector)
    L  : number of sections (1 non-zero entry per section)
    M  : number of entries per section
    '''

    # The current method of preventing overflow is:
    # 1) subtract the maximum entry of array x from all entries before taking
    #    np.exp(x). Note: the largest exponent that np.float64 can handle is
    #    roughly 709.
    #
    # Perhaps I can avoid tau becoming too small by changing how I do early
    # termination in the AMP algorithm.
    termination = False
    x   = np.sqrt(n*P_l) * (s / tau_sqr) 
    if (np.any(x-x.max() >= 700)) or (np.any(x-x.max() <= -800)): 
        print("Possible overflow from exponent in mmse_estimator")
        termination = True
    top = np.exp(x - x.max(), dtype=np.float64)
    bot = top.reshape(-1, M).sum(axis=1).repeat(M)

    # Cast back to normal float or complex data types
    if termination: 
        return s, termination 
    return ((np.sqrt(n*P_l)) * (top / bot)).astype(np.float64), termination

def msg_vector_map_estimator(s, M, sqrt_nP_l):
    '''
    MAP estimator of message vector of SPARC in independent
    additive white Gaussian noise.

    s  : effective observation in Gaussian noise (length L*M vector)
    M  : number of entries per section
    '''

    L = s.size // M # Number of sections

    beta = np.zeros_like(s, dtype=float).reshape(L,-1)

    idxs = s.reshape(L,-1).argmax(axis=1)
    beta[np.arange(L), idxs] = sqrt_nP_l


    return beta.ravel()

def beta_estimate_to_bp_probs(beta, L, M, sqrt_nP_l): 
    '''
    Takes the posterior probabilities given by AMP (beta) and coverts to ldpc probs. 
    Where the ldpc probs are the probabilites of that bit being a 0. 
    '''
    logM = int(np.log2(M))
    # Turns amp posterior probs into ldpc 
    beta_estimate_sectioned = beta.reshape(L, M)
    ldpc_probs = np.zeros((L, logM)) 
    for l in range(L):
        for i in range(logM): 
            b = logM - 1 - i
            k = 0
            while k < M: 
                for j in range(k, k+pow(2,i)):
                    ldpc_probs[l][b] += (beta_estimate_sectioned[l][j] / sqrt_nP_l)
                k = k + pow(2, i+1)

    ldpc_probs = ldpc_probs.reshape(L*logM)

    return ldpc_probs

def S_k_mapping(M): 
    '''
    Produces a list of logM lists, each of size 2^{logM-1}. Each list is the indicies of the non-zero
    terms that correspond to the kth ldpc bit being a zero. 

    E.g 0 -> 1  0
        1 -> 0  1 
        k -> q0 q1  So this would be [[0]]
    '''

    logM = int(np.log2(M))
    S_k = [[] for _ in range(logM)]
    for i in range(logM): 
        b = logM - 1 - i
        k = 0
        while k < M: 
            for j in range(k, k+pow(2,i)): 
                S_k[b].append(j)
            k = k + pow(2,i+1)

    return S_k

def ldpc_bp(ldpc_probs, c, num_its, hard_decision_bool):
    '''
    Takes in ldpc_probs and decodes using bp, returns the probabilites.
    '''

    eps = 1e-15  # Small constant to prevent log(0)
    ldpc_probs = np.clip(ldpc_probs, eps, 1 - eps)
    LLR = np.log(ldpc_probs) - np.log(1 - ldpc_probs)

    assert len(LLR) % c.N == 0
    num_blocks = len(LLR) / c.N
    LLR = np.array_split(LLR, num_blocks)
    app = []
    app_cut = []
    for chunk in LLR: 
        decoded = c.decode(chunk, num_its)[0]
        app.append(decoded)
        app_cut.append(decoded[:c.K])
    app = np.array(app)
    app_cut = np.array(app_cut)
    app = app.flatten()
    app_cut = app_cut.flatten()

    if (hard_decision_bool): 
        hard_bools = (app_cut < 0)
        hard_decision_bits = np.array([int(bool_val) for bool_val in hard_bools])
        ldpc_probs = 0 
    else: 
        ldpc_probs = (np.exp(app))/(1+np.exp(app))
        hard_decision_bits = 0 

    return ldpc_probs, hard_decision_bits

def ldpc_bp_test(ldpc_probs, c, num_its, hard_decision_bool):
    '''
    Does the same as ldpc_bp but also just extracts the user bits using the fact its a systematic code.
    '''

    eps = 1e-15  # Small constant to prevent log(0)
    ldpc_probs = np.clip(ldpc_probs, eps, 1 - eps)
    LLR = np.log(ldpc_probs) - np.log(1 - ldpc_probs)

    assert len(LLR) % c.N == 0
    num_blocks = len(LLR) / c.N
    LLR = np.array_split(LLR, num_blocks)
    app = []
    app_cut = []
    no_bp_cut = []
    for chunk in LLR: 
        no_bp_cut.append(chunk[:c.K])
        decoded = c.decode(chunk, num_its)[0]
        app.append(decoded)
        app_cut.append(decoded[:c.K])
    app = np.array(app)
    app_cut = np.array(app_cut)
    no_bp_cut = np.array(no_bp_cut)
    app = app.flatten()
    app_cut = app_cut.flatten()
    no_bp_cut = no_bp_cut.flatten()

    if (hard_decision_bool): 
        hard_bools = (app_cut < 0)
        hard_bools_no_bp = (no_bp_cut < 0)
        hard_decision_bits = np.array([int(bool_val) for bool_val in hard_bools])
        no_bp_bits = np.array([int(bool_val) for bool_val in hard_bools_no_bp])
        ldpc_probs = 0 
    else: 
        ldpc_probs = (np.exp(app))/(1+np.exp(app))
        hard_decision_bits = 0 

    return ldpc_probs, hard_decision_bits, no_bp_bits

def ldpc_probs_to_user_bits(ldpc_probs, c): 

    num_blocks = len(ldpc_probs) / c.N
    ldpc_probs_split = np.array_split(ldpc_probs, num_blocks)
    user_bits_probs = []
    for chunk in ldpc_probs_split: 
        user_bits_probs.append(chunk[:c.K])
    user_bits_probs = np.array(user_bits_probs)
    user_bits_probs = user_bits_probs.flatten()

    decoded_user_bits = (user_bits_probs < 0.5).astype(int)

    return decoded_user_bits

def ldpc_bits_to_user_bits(ldpc_bits, c): 

    num_blocks = len(ldpc_bits) / c.N
    ldpc_bits_split = np.array_split(ldpc_bits, num_blocks)
    user_bits = []
    for chunk in ldpc_bits_split: 
        user_bits.append(chunk[:c.K])
    user_bits = np.array(user_bits)
    user_bits = user_bits.flatten()

    return user_bits

def bp_output_to_beta_estimate(ldpc_probs, L, M, sqrt_nP_l): 
    '''
    For beta estimate [0] being the non-zero we have the ldpc bits of all 1s. 
    Therefore as the ldpc_probs are for it being a zero. We want to multiply (1-p) for all p.
    To do this the beta index we are looking at is turned into a binary num and the p's are 
    multiplied together accordingly. 
    '''

    logM = int(np.log2(M))
    ldpc_probs_sectioned = ldpc_probs.reshape(L,logM)
    amp_probs = np.ones((L,M))
    for l in range(L): 
        for i in range(M): 
            binary_num = format(i,f"0{logM}b")
            # The probability is the product of p if binary_num = 0 or (1-p) if binary_num = 1
            for j in range(logM): 
                amp_probs[l][i] = amp_probs[l][i]*ldpc_probs_sectioned[l][j] if (binary_num[j] == '0') else amp_probs[l][i]*(1-ldpc_probs_sectioned[l][j])

    amp_probs = amp_probs.reshape(L*M)
    return amp_probs * sqrt_nP_l


######## Design Matrix ######### --------------------------------------------------------------------------------------------------------------------

def create_design_matrix(L, M, n, rand_seed):
    ''' 
    Constructs the design matrix and functions to multiply 
    a vector by the matrix and its transpose. 
    ''' #(Tested)

    rng = np.random.default_rng(rand_seed) 
    ML = M * L 
    scale = 1/np.sqrt(n)
    A = rng.normal(loc=0, scale=scale, size=(n, ML))
    return A 

######## Message vector operations ######## ----------------------------------------------------------------------------------------------------

def bin_arr_2_msg_vector(bin_arr, M, n, P_l):
    '''
    Convert binary array (numpy.ndarray) to SPARC message vector

    M: entries per section of SPARC message vector
    ''' #(Tested)

    logM = int(np.log2(M))
    bin_arr_size = bin_arr.size
    assert bin_arr_size % logM == 0
    L = bin_arr_size // logM # Num of sections

    msg_vector = np.zeros(L*M)

    for l in range(L):
        idx = bin_arr_2_int(bin_arr[l*logM : (l+1)*logM])
        val = np.sqrt(n*P_l)
        msg_vector[l*M + idx] = val

    return msg_vector

def msg_vector_2_bin_arr(msg_vector, M):
    '''
    Convert SPARC message vector to binary array (numpy.ndarray)

    M: entries per section of SPARC message vector
    '''
    assert type(msg_vector) == np.ndarray
    assert type(M)==int and M>0 
    assert msg_vector.size % M == 0
    logM = int(round(np.log2(M)))
    L = msg_vector.size // M

    sec_size = logM

    msg_reshape  = msg_vector.reshape(L,M)
    idxs1, idxs2 = np.nonzero(msg_reshape)
    assert np.array_equal(idxs1, np.arange(L)) # Exactly 1 nonzero in each row

    bin_arr = np.zeros(L*sec_size, dtype='bool')
    for l in range(L):
        bin_arr[l*sec_size : l*sec_size+logM] = int_2_bin_arr(idxs2[l], logM)

    return bin_arr

def encode_ldpc(user_bits, ldpc_params, mults, unprotected_bit_len): 
    '''
    Takes user_bits in, seperates the unprotected bits, and encodes the 
    remaining bits, outputs both concatenated. 
    ''' #(Tested)

    c = code(ldpc_params["standard"], ldpc_params["rate"], ldpc_params["z"])
    ldpc_bits = user_bits[unprotected_bit_len:]
    unprotected_bits = user_bits[:unprotected_bit_len].astype(bool)

    ldpc_bits = np.array_split(ldpc_bits, mults)
    for i in range(mults): 
        ldpc_bits[i] = c.encode(ldpc_bits[i])
    ldpc_bits = np.concatenate(ldpc_bits).astype(bool)
    total_bits = np.concatenate((unprotected_bits, ldpc_bits))

    return total_bits

######## Binary operations ######## --------------------------------------------------------------------------------------------------------

def bin_arr_2_int(bin_array):
    '''
    Binary array (numpy.ndarray) to integer
    '''
    assert bin_array.dtype == 'bool'
    k = bin_array.size
    assert 0 < k < 64 # Ensures non-negative integer output
    return bin_array.dot(1 << np.arange(k)[::-1])

def int_2_bin_arr(integer, arr_length):
    '''
    Integer to binary array (numpy.ndarray) of length arr_length
    NB: only works for non-negative integers
    '''
    assert integer>=0
    return np.array(list(np.binary_repr(integer, arr_length))).astype('bool')

def bit_err_rate(bits_in, bits_out): 
    '''
    Calculates bit error rate from bits in and out 
    '''

    assert len(bits_in == bits_out)
    ber = np.sum(bits_in != bits_out) / len(bits_in)

    return ber

