import numpy as np 

######## Differentiated eta ######## --------------------------------------------------------------------------------------------------------------
'''
Takes inputs beta, vk, vk_0, alpha, tau_sqr for the equation and computes the 
derivative of eta(s) with respect to s_i.
'''

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
                if (binary_num[k] == '0'): 
                    main_term[l][i] -= vk_sectioned[l][k] * differentiated_LLR_calc(vk_0, alpha_sectioned, tau_sqr, l, k, i, S_k, n, P_l, L, M)
                else: 
                    main_term[l][i] += (1 - vk_sectioned[l][k]) * differentiated_LLR_calc(vk_0, alpha_sectioned, tau_sqr, l, k, i, S_k, n, P_l, L, M)

    main_term = main_term.reshape(L*M)

    return beta * main_term 

######## Differentiated LLR  ######## --------------------------------------------------------------------------------------------------------------
'''
Takes inputs vk_0, alpha, tau_sqr and calculated the derivative of the LLR with respect to s_i
in a specific section and for a specific index in that section. 
'''

def vk_0_term_calc(vk_0, l, k, L, M): 
    vk_0_sectioned = vk_0.reshape(L,int(np.log2(M)))
    return 1 / (vk_0_sectioned[l][k] * (1 - vk_0_sectioned[l][k]))

def sum_term_calc(alpha_sectioned, tau_sqr, l, k, i, S_k, n, P_l): 

    sum_term = 0
    q_to_sum_over = S_k[k]
    for q in q_to_sum_over: 
        if q == i: 
            sum_term += alpha_sectioned[l][q] * (np.sqrt(n*P_l)/tau_sqr) * (1 - alpha_sectioned[l][q])
        else: 
            sum_term += alpha_sectioned[l][q] * (np.sqrt(n*P_l)/tau_sqr) * (-alpha_sectioned[l][q])    
    
    return sum_term 

def test_sum_term_calc(): 
    n, P_l = 2, 2
    tau_sqr = 0.25
    l, k, i = 0, 0, 0
    S_k = [[0,1],[0,2]]
    alpha_sectioned = np.array([[0.7,0.1,0.1,0.1]])
    sum_term = sum_term_calc(alpha_sectioned, tau_sqr, l, k, i, S_k, n, P_l)
    assert sum_term == 1.6


def differentiated_LLR_calc(vk_0, alpha_sectioned, tau_sqr, l, k, i, S_k, n, P_l, L, M): 
    
    sum_term = sum_term_calc(alpha_sectioned, tau_sqr, l, k, i, S_k, n, P_l)
    vk_0_term = vk_0_term_calc(vk_0, l, k, L, M)

    return vk_0_term * sum_term

