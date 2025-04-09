import sys
import os

# Get the parent directory of the current file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)  

from ldpc_jossy.py.ldpc import code 

standard = '802.16'
ldpc_rate = '5/6'
z = 90

c = code(standard, ldpc_rate, z)
print("c.K: ", c.K)