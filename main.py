import torch
import numpy as np
from dataclasses import dataclass, field

import Bit_functions
@dataclass
class CommunicationParams:
    K: int = 5 # number of devices
    M: int = 10 # number of tx symbols of each device
    R: int = 10 # number of symbol slots

    P0: int = 10
    sigma: float = 1

params = CommunicationParams()

### Bit-slicing & quantization parameter ###

b = torch.tensor([1, 1, 2])
B = b.sum() * 2
x_min = -16  # lower bound of tx symbols
x_max = 15  # upper bound of tx symbols

L = b.numel() # number of bit segments
delta = (x_max - x_min) / 2 ** B # quantization density


x= torch.empty(1).uniform_(x_min, x_max)

q = Bit_functions.quantization(x, x_min, x_max, B)
x_hat = Bit_functions.inv_quantization(q, x_min, x_max, B)
print(x,x_hat)


q_l = Bit_functions.bit_slicing(q, b)
x_hat = Bit_functions.reconstruction(q_l, b, x_min, delta)
print(q_l)

q_l_bit = []
for idx, dec in enumerate(q_l):
    print(b[idx])
    q_l_bit.append(Bit_functions.decimal_to_binary_auto(dec, b[idx]))

q_tmp = torch.cat(q_l_bit)
print(q_tmp)

q_odd = torch.zeros(B//2)
q_even = torch.zeros(B//2)

idx_odd = 0
idx_even = 0
for idx in range(B): # remember python starts from index 0
    if idx % 2 == 0:
        q_odd[idx_even] = q_tmp[idx]
        idx_even += 1
    if idx % 2 != 0:
        q_even[idx_odd] = q_tmp[idx]
        idx_odd += 1

print(q_odd,q_even)

q_odd_dec = Bit_functions.bit_segments_to_ints(q_odd, b)
q_even_dec = Bit_functions.bit_segments_to_ints(q_even, b)
print(q_odd_dec)
print(q_even_dec)

m = Bit_functions.constellation_mapper(q_odd_dec, q_even_dec, b)
print(m)