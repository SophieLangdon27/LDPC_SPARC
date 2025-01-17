'''
My own test of Jossy's LDPC library 

'''

import sys
import os

# Get the parent directory of the current file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np 
from ldpc_jossy.py.ldpc import code

c = code('802.11n','5/6',81)
print("Standard ", c.standard, ", Rate ", c.rate, ", z ", c.z, ", k ", c.K, ", n", c.N, "\n")

raw_input_bits = np.random.randint(0, 2, c.K)
x = c.encode(raw_input_bits)
# H = c.pcmat()
# assert np.count_nonzero(np.mod(np.dot(x,np.transpose(H)),2)) == 0
print("Bits ", raw_input_bits[0:4], "\n")

# What is the below giving LLRs?
y = np.array(10*(0.5-x), dtype = float) 
app, it = c.decode(y, 'sumprod') 
print("Iterations ", it, "\n")

d = np.array(app < 0, dtype=int)
print("Decoded ", d[0:4], "\n")

# u = np.random.randint(0,2,K)
#         x = mycode.encode(u)
#         assert np.count_nonzero(np.mod(np.dot(x,np.transpose(H)),2)) == 0
#         # modulate and amplify
#         y = np.array(10*(.5-x), dtype=float)
#         app,it = mycode.decode(y,'sumprod')
#         assert it == 0
#         xh = np.array(app<0, dtype=int)
#         assert np.count_nonzero(xh != x) == 0
#         app,it = mycode.decode(y,'sumprod2')
#         assert it == 0
#         xh = np.array(app<0, dtype=int)
#         assert np.count_nonzero(xh != x) == 0
