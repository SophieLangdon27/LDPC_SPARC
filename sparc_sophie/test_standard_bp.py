'''
JUST SENDS ACROSS AN LDPC CODEWORD, ADDS NOISE AND DECODES USING BP. THE DIFFERENCE BETWEEN DOING BP 
AND JUST TAKING THE FIRST K BITS IS MINIMAL. 

'''

import sys
import os

# Get the parent directory of the current file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
from ldpc_jossy.py.ldpc import code 
import matplotlib.pyplot as plt

#####################       EXTRA FUNCTIONS         #####################

def bit_err_rate(bits_in, bits_out): 
    '''
    Calculates bit error rate from bits in and out 
    '''

    assert len(bits_in == bits_out)
    ber = np.sum(bits_in != bits_out) / len(bits_in)

    return ber

#####################       TEST CODE         #####################

standard = '802.11n'
ldpc_rate = '5/6'
int_rate = 5/6
z = 81
c = code(standard, ldpc_rate, z)

mults = 3 
k = c.K * mults
user_bits = np.random.randint(0, 2, size=k)

ldpc_bits = np.array_split(user_bits, mults)
for i in range(mults): 
    ldpc_bits[i] = c.encode(ldpc_bits[i])
ldpc_bits = np.concatenate(ldpc_bits)
mapped_ldpc_bits = np.where(ldpc_bits == 0, -1, ldpc_bits)

n_ldpc = int(k * (1/int_rate))
logM = 6
assert n_ldpc % logM == 0

num_snrs = 10
snr_start = 1
snr_stop = 6
snr_store = np.linspace(snr_start, snr_stop, num_snrs)
P = 1.0
awgn_var_store = P/snr_store

def sim(awgn_var, c, user_bits):
    noise = np.sqrt(awgn_var)*np.random.randn(ldpc_bits.size)
    noisy_ldpc_bits = mapped_ldpc_bits + noise 
    LLR =  (-1)*(noisy_ldpc_bits) / (awgn_var)

    assert len(LLR) % c.N == 0
    num_blocks = len(LLR) / c.N
    LLR = np.array_split(LLR, num_blocks)
    app_cut = []
    no_bp_cut = []
    for chunk in LLR: 
        decoded = c.decode(chunk, 400)[0]
        app_cut.append(decoded[:c.K])
        no_bp_cut.append(chunk[:c.K])
    app_cut = np.array(app_cut)
    app_cut = app_cut.flatten()
    no_bp_cut = np.array(no_bp_cut)
    no_bp_cut = no_bp_cut.flatten()

    hard_bools = (app_cut < 0)
    hard_decision_bits = np.array([int(bool_val) for bool_val in hard_bools])

    no_bp_bools = (no_bp_cut < 0)
    no_bp_bits = np.array([int(bool_val) for bool_val in no_bp_bools])

    ber_1 = bit_err_rate(user_bits, hard_decision_bits)
    ber_2 = bit_err_rate(user_bits, no_bp_bits)
    return ber_1, ber_2


ber_store_1 = np.zeros(num_snrs)
ber_store_2 = np.zeros(num_snrs)

for v in range(num_snrs):  
    ber_store_1[v], ber_store_2[v]  = sim(awgn_var_store[v], c, user_bits)
    print("Step ", v+1)

plt.figure(figsize=(15,4))
plt.plot(snr_store, ber_store_1, marker='o', label='bp')
plt.plot(snr_store, ber_store_2, marker='o', label='no_bp')

# Add title and labels
plt.title('Line Plot of Bit Error Rate against SNR')
plt.xlabel('SNR')
plt.ylabel('BER')
plt.legend()  # Show legend

# Show the plot
plt.tight_layout()
plt.show()
