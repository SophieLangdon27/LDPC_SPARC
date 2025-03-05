import sys
import os

# Get the parent directory of the current file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
from ldpc_jossy.py.ldpc import code 

### Main encode/decode functions -------------------------------------------------------------------------------------------------------------

def sparc_ldpc_encode(sparc_params, ldpc_params, lengths, ldpc_bool, rand_seed): 
    ''' 
    Encodes random message vector to an LDPC codeword then SPARC codeword.
    Note this is only for partially protected codes. 
    ''' #(Tested)

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
    n = int(round(encoded_bit_len/R))
    P_l = P/L
    beta0 = bin_arr_2_msg_vector(total_bits, M, n, P_l)

    # Update code_params
    R_actual = encoded_bit_len / n      # Actual rate
    sparc_params.update({'n':n, 'R_actual':R_actual})

    A = create_design_matrix(L, M, n, rand_seed) 
    x = np.dot(A, beta0)

    return user_bits, beta0, x, A

def sparc_ldpc_decode(y, sparc_params, ldpc_params, decode_params, ldpc_bool, lengths, A):
    '''
    Decodes using AMP fully then BP fully.
    '''
    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    sqrt_nP_l = np.sqrt(n * (P/L))

    # Run AMP fully 
    beta_soft_estimate, s = sparc_amp(y, sparc_params, decode_params, A)
    
    if (ldpc_bool): 
        c = code(ldpc_params["standard"], ldpc_params["rate"], ldpc_params["z"])
        L_unprotected, L_ldpc = map(lengths.get,['L_unprotected','L_ldpc'])
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

def naively_integrated_decoder(y, sparc_params, ldpc_params, decode_params, A): 

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
        beta, z, tau_sqr = sparc_amp_single_run(sparc_params, y, A, AT, beta, z, tau_sqr)
        ldpc_probs = beta_estimate_to_bp_probs(beta, L, M, sqrt_nP_l)
        if (i != t_max-1): 
            ldpc_probs, _ = ldpc_bp(ldpc_probs, c, 6, False)
            beta = bp_output_to_beta_estimate(ldpc_probs, L, M, sqrt_nP_l)
        else: 
            _, hard_decision_bits = ldpc_bp(ldpc_probs, c, 6, True)

    return hard_decision_bits


def integrated_decoder(y, sparc_params, ldpc_params, decode_params, A): 

    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    P_l = P/L
    c = code(ldpc_params["standard"], ldpc_params["rate"], ldpc_params["z"])
    t_max = decode_params['t_max']
    num_runs = 6 

    # Initial AMP + BP
    beta = np.zeros(L*M)
    z = 0
    differentiated_eta = np.zeros(n)
    AT = A.T
    for t in range(t_max):  
        if t != 0: 
            differentiated_eta = differentiated_eta(beta, vk, vk_0, alpha, tau_sqr)
        z = y - (np.dot(A, beta)) + z*(np.sum(differentiated_eta)/n)
        s = np.dot(AT, z) + beta 
        tau_sqr = np.sum(z ** 2) / n    
        if (t != t_max-1):
            hard_decision = False
            alpha, vk_0, vk, beta, _ = eta(s, tau_sqr, n, P_l, M, L, c, num_runs, hard_decision)
        else: 
            hard_decision = True 
            _, _, _, _, hard_decision_bits= eta(s, tau_sqr, n, P_l, M, L, c, num_runs, hard_decision)

    return hard_decision_bits




######## AMP + BP decoder ######## --------------------------------------------------------------------------------------------------------------

def eta(s, tau_sqr, n, P_l, M, L, c, num_runs, hard_decision_bool): 
    
    sqrt_nP_l = np.sqrt(n*P_l)

    # Step One (Expectation): 
    weighted_alpha = msg_vector_mmse_estimator(s, tau_sqr, n, P_l, M)
    alpha = weighted_alpha/sqrt_nP_l

    # Step Two (Conversion to codeword bit-wise probabilites): 
    vk_0 = beta_estimate_to_bp_probs(weighted_alpha, L, M, sqrt_nP_l)

    # Step Three (Belief Propagation): 
    vk, hard_decision_bits = ldpc_bp(vk_0, c, num_runs, hard_decision_bool)

    # Step Four (Conversion to beta section-wise probabilites): 
    if hard_decision_bool: 
        beta = np.zeros(L*M)
    else: 
        beta = bp_output_to_beta_estimate(vk, L, M, sqrt_nP_l)
    
    return alpha, vk_0, vk, beta, hard_decision_bits

def differentiated_eta(beta, vk, vk_0, alpha, tau_sqr, L, M, S_k, n, P_l): 

    logM = int(np.log2(M))
    vk_sectioned = vk.reshape(L,logM)
    main_term = np.zeros((L,M))
    for l in range(L): 
        for i in range(M): 
            binary_num = format(i,f"0{logM}b")
            # The probability is the product of p if binary_num = 0 or (1-p) if binary_num = 1
            for k in range(logM): 
                if (binary_num[k] == '0'): 
                    main_term[l][i] -= vk_sectioned[l][k] * sub_term(vk_0, alpha, tau_sqr, l, k, i, S_k, n, P_l)
                else: 
                    main_term[l][i] += (1 - vk_sectioned[l][k]) * sub_term(vk_0, alpha, tau_sqr, l, k, i, S_k, n, P_l)

    amp_probs = amp_probs.reshape(L*M)


    return beta * main_term 

def sub_term(vk_0, alpha, tau_sqr, l, k, i, S_k, n, P_l): 

    sum_term = 0
    q_to_sum_over = S_k[k]
    for q in range(q_to_sum_over): 
        if q == i: 
            sum_term += alpha[l][q] * (np.sqrt(n*P_l)/tau_sqr) * (1 - alpha[l][q])
        else: 
            sum_term += alpha[l][q] * (np.sqrt(n*P_l)/tau_sqr) * (-alpha[l][q])
        
    return (1 / (vk_0[l][k] * (1 - vk_0[l][k]))) * sum_term

def sparc_amp(y, sparc_params, decode_params, A): 
    '''
    The normal AMP algorithm.
    ''' #(Tested)
    P, R, L, M = sparc_params['P'], sparc_params['R'], sparc_params['L'], sparc_params['M']
    n = len(y)
    P_l = P / L
    AT = A.T

    t_max, rtol = decode_params['t_max'], decode_params['rtol']

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

def sparc_amp_single_run(sparc_params, y, A, AT, beta, z, tau_sqr):
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

def msg_vector_mmse_estimator(s, tau_sqr, n, P_l, M):
    '''
    MMSE (Bayes optimal) estimator of message vector of SPARC in
    (possibly complex) independent additive Gaussian noise.

    s  : effective observation in Gaussian noise (length L*M vector)
    tau: the noise variance (length L*M vector)
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
    top = np.exp(x - x.max(), dtype=np.float64)
    bot = top.reshape(-1, M).sum(axis=1).repeat(M)

    # Cast back to normal float or complex data types
    return ((np.sqrt(n*P_l)) * (top / bot)).astype(np.float64)

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

def ldpc_bp(ldpc_probs, c, num_runs, hard_decision_bool):
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
        decoded = c.decode(chunk, num_runs)[0]
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

