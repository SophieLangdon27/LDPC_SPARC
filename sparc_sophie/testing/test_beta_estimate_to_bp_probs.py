import numpy as np 

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

def test_beta_estimate_to_bp_probs(): 
    # Known with certainty 
    beta = np.array([1,0,0,0, 0,0,1,0, 1,0,0,0])
    L = 3
    M = 4
    sqrt_nP_l = 1
    test_1 = beta_estimate_to_bp_probs(beta, L, M, sqrt_nP_l)
    ans_1 = np.array([1,1, 0,1, 1,1])
    assert np.array_equal(test_1, ans_1)

    # The probability of each term being non-zero is a mutually exclusive event so the probabilities must add to one. 
    beta = np.array([0.7,0.1,0.1,0.1, 0.1,0.1,0.7,0.1, 0.7,0.1,0.1,0.1])
    L = 3
    M = 4
    sqrt_nP_l = 1
    test_2 = beta_estimate_to_bp_probs(beta, L, M, sqrt_nP_l)
    test_2 = np.where(test_2 < 0.5, 1, 0)
    ans_2 = np.array([0,0, 1,0, 0,0])
    assert np.array_equal(test_2, ans_2)

    beta = np.array([0.5,0.2,0.1,0.1, 0.1,0.1,0.7,0.1, 0.2,0.4,0.2,0.2])
    L = 3
    M = 4
    sqrt_nP_l = 1
    test_2 = beta_estimate_to_bp_probs(beta, L, M, sqrt_nP_l)
    test_2 = np.where(test_2 < 0.5, 1, 0)
    ans_2 = np.array([0,0, 1,0, 0,1])
    assert np.array_equal(test_2, ans_2)