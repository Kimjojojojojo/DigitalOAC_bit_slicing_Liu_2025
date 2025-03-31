import torch
import numpy as np
from dataclasses import dataclass, field

import Functions
import Bit_functions
import Single_device


@dataclass
class CommunicationParams:
    K: int = 4 # number of devices
    M: int = 10 # number of tx symbols of each device
    R: int = 10 # number of symbol slots

    P0: int = 10
    sigma: float = 1
    g_th: float = 0.3

params = CommunicationParams()

### Bit-slicing & quantization parameter ###
M = 4  # number of data in each transmission
N = M//2
b = torch.tensor([1, 1, 2])
B = b.sum()
x_min = -16  # lower bound of tx symbols
x_max = 15  # upper bound of tx symbols

L = b.numel() # number of bit segments
delta = (x_max - x_min) / 2 ** B # quantization density
Ei_g_th, _ = Functions.exponential_integration(params.g_th) # reference [30]
rho_0 = params.P0 / (params.M * Ei_g_th)
d = torch.zeros(L)  # distance between each point of the constellation
for l in range(L):
    d[l] = torch.sqrt(6 / (4 ** b[l] - 1))

# process of each device : from generate symbol to constellation mapping and aggregation
x = torch.empty(params.K, M).uniform_(x_min, x_max)
x_hat = torch.empty(params.K, M)
m_k = []
for k in range(params.K):
    print(f'K = {k}')
    k_th_symbols = Single_device.symbol_converter(x[k], x_min, x_max, M, B, b, L)
    m_k.append(k_th_symbols)
    print(m_k[k])

# aggregation
s = torch.zeros(N,L, dtype=torch.cfloat)
for n in range(N):
    for l in range(L):
        tmp = 0
        for k in range(params.K):
            tmp += m_k[k][n][l]
        s[n][l] = tmp



z = Functions.generate_rayleigh(N, L, params.sigma)

# aggregation
r = s
print(r)
# equalizing
s_hat = r/rho_0

u_hat_ml = torch.zeros(M,L) # decoding
for m in range(M):
    for l in range(L):
        if m % 2 == 0:
            u_hat_ml[m][l] = torch.real(s_hat[m//2][l])/d[l] + (2**b[l]-1)/(2*params.K)
        if m % 2 == 1:
            u_hat_ml[m][l] = torch.imag(s_hat[(m-1)//2][l])/d[l] + (2**b[l]-1)/(2*params.K)

print(u_hat_ml)

u_hat_m = Bit_functions.inv_bit_slicing(u_hat_ml, M, b) # decoding

y_hat_m = Bit_functions.reconstruction_y(u_hat_m, params, M, x_min, delta)
print(torch.sum(x,dim=0))
print(y_hat_m)
### requiring MAP detecion ###

### ###





