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
B = 4 # quantization bits
b = torch.tensor([1, 1, 2])
x_min = -16  # lower bound of tx symbols
x_max = 15  # upper bound of tx symbols

L = b.numel() # number of bit segments
delta = (x_max - x_min) / 2 ** B

x= torch.empty(1).uniform_(x_min, x_max)

q = Bit_functions.quantization(x, x_min, x_max, B)
x_hat = Bit_functions.inv_quantization(q, x_min, x_max, B)
print(x,x_hat)
print(q)

q_l = Bit_functions.bit_slicing(q, b)
x_hat = Bit_functions.reconstruction(q_l, b, x_min, delta)

print(q_l)

print(x_hat)

