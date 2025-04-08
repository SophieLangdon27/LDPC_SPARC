import sys
import os

# Get the parent directory of the current file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)  

from ldpc_jossy.py.ldpc import code 

standard = '802.11n'
ldpc_rate = '5/6'
z = 81

c = code(standard, ldpc_rate, z)
print("c.K: ", c.K)