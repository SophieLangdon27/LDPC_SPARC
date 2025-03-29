import numpy as np 

######## S_k Mapping ######## --------------------------------------------------------------------------------------------------------------
'''
For a ldpc codeword bit index, k, the set of sparse section-wise indices, q, that correspond to k being a zero. 
0  0 -> 1  0  0  0
0  1 -> 0  1  0  0
1  0 -> 0  0  1  0
1  1 -> 0  0  0  1
k0 k1   q0 q1 q2 q3

So for k0, S_k: q0, q1

This is done section wise and only dependant on M. The output is logM (k) lists of length 2^{logM-1}
'''

def S_k_mapping(M): 

    logM = int(np.log2(M))
    S_k = [[] for _ in range(logM)]
    for i in range(logM): 
        k = logM - 1 - i
        b = 0
        while b < M: 
            for q in range(b, b+pow(2,i)): 
                S_k[k].append(q)
            b = b + pow(2,i+1)

    return S_k

def test_S_k_mapping(): 
    test_1 = S_k_mapping(4)
    assert test_1 == [[0,1],[0,2]]

    test_2 = S_k_mapping(8)
    assert test_2 == [[0,1,2,3],[0,1,4,5],[0,2,4,6]]

    test_3 = S_k_mapping(16)
    assert test_3 == [[0,1,2,3,4,5,6,7],[0,1,2,3,8,9,10,11],[0,1,4,5,8,9,12,13],[0,2,4,6,8,10,12,14]]

def S_k_mapping_2(M): 

    logM = int(np.log2(M))
    S_k = np.ones((logM,pow(2,(logM-1))))
    for m in range(logM): 
        k = logM - 1 - m
        b = 0
        i = 0 
        while b < M: 
            for j in range(b, b+pow(2,m)): 
                S_k[k][i] = j
                i += 1
            b = b + pow(2,m+1)

    return S_k

def test_S_k_mapping_2(): 
    test_1 = S_k_mapping_2(4)
    assert np.array_equal(test_1, ([[0,1],[0,2]]))

    test_2 = S_k_mapping_2(8)
    assert np.array_equal(test_2, ([[0,1,2,3],[0,1,4,5],[0,2,4,6]]))

    test_3 = S_k_mapping_2(16)
    assert np.array_equal(test_3, ([[0,1,2,3,4,5,6,7],[0,1,2,3,8,9,10,11],[0,1,4,5,8,9,12,13],[0,2,4,6,8,10,12,14]]))