import torch
import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from dataclasses import dataclass

import Bit_functions

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
@dataclass
class CommunicationParams:
    K: int = 5
    M: int = 10
    R: int = 10
    P0: int = 10
    sigma: float = 1

params = CommunicationParams()

# Bit-slicing & quantization parameters
b = torch.tensor([1, 1, 2])
B = int(b.sum()) * 2
x_min = -16
x_max = 15
L = b.numel()
delta = (x_max - x_min) / 2 ** B

# 결과 저장용 리스트
m0_list = []
m1_list = []
m2_list = []

N = 500  # 반복 횟수

for _ in range(N):
    x = torch.empty(1).uniform_(x_min, x_max)

    q = Bit_functions.quantization(x, x_min, x_max, B)
    x_hat = Bit_functions.inv_quantization(q, x_min, x_max, B)

    q_l = Bit_functions.bit_slicing(q, b)
    x_hat = Bit_functions.reconstruction(q_l, b, x_min, delta)

    q_l_bit = []
    for idx, dec in enumerate(q_l):
        q_l_bit.append(Bit_functions.decimal_to_binary_auto(dec, b[idx]))

    q_tmp = torch.cat(q_l_bit)

    q_odd = torch.zeros(B // 2)
    q_even = torch.zeros(B // 2)

    idx_odd = 0
    idx_even = 0
    for idx in range(B):
        if idx % 2 == 0:
            q_odd[idx_even] = q_tmp[idx]
            idx_even += 1
        else:
            q_even[idx_odd] = q_tmp[idx]
            idx_odd += 1

    q_odd_dec = Bit_functions.bit_segments_to_ints(q_odd, b)
    q_even_dec = Bit_functions.bit_segments_to_ints(q_even, b)

    m = Bit_functions.constellation_mapper(q_odd_dec, q_even_dec, b)

    m0_list.append(m[0])
    m1_list.append(m[1])
    m2_list.append(m[2])

# 텐서 → 넘파이 변환
m0_np = torch.stack(m0_list).numpy()
m1_np = torch.stack(m1_list).numpy()
m2_np = torch.stack(m2_list).numpy()

# 시각화 (matplotlib)
plt.figure(figsize=(8, 6))
plt.scatter(m0_np.real, m0_np.imag, label='m[0]', alpha=0.6, s=20)
plt.scatter(m1_np.real, m1_np.imag, label='m[1]', alpha=0.6, s=20)
plt.scatter(m2_np.real, m2_np.imag, label='m[2]', alpha=0.6, s=20)

plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.title("Constellation Diagram of m[0], m[1], m[2]")
plt.grid(True)
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()
