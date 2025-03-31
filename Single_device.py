import torch
import numpy as np
from dataclasses import dataclass, field

import Bit_functions

def symbol_converter(x, x_min, x_max, M, B, b, L):
    delta = (x_max - x_min) / 2 ** B  # quantization density

    q = [] # integers which indicates quantized index
    for m in range(M):
        q.append(Bit_functions.quantization(x[m], x_min, x_max, B))

    #x_hat = Bit_functions.inv_quantization(q, x_min, x_max, B)

    q_l = [] # quantized index being bit-sliced
    for m in range(M):
        q_l.append(Bit_functions.bit_slicing(q[m], b))

    #x_hat = Bit_functions.reconstruction(q_l, b, x_min, delta)

    m_vec = [] # constellation mapping
    for n in range(M//2):
        m_n = []
        for l in range(L):
            d_l = torch.sqrt(6.0 / (4 ** b[l] - 1))
            common_part = (2 ** b[l] - 1) / 2
            real_part = (q_l[2 * n][l] - common_part) * d_l
            imag_part = (q_l[2 * n + 1][l] - common_part) * d_l
            m_nl = torch.complex(real_part, imag_part)
            m_n.append(m_nl)
        m_vec.append(m_n)


    #print(m)

    return m_vec