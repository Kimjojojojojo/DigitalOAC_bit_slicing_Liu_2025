import torch
import math

def quantization(x, x_min, x_max, B): # Input : real number x; Output : quantization index q
    delta = (x_max - x_min) / 2 ** B
    offset = 0 - x_min
    x_offset = x + offset
    q = torch.floor(x_offset / delta)

    return q

def inv_quantization(q,  x_min, x_max, B): # Input : quantization index q; Output : estimation of original number x_hat
    delta = (x_max - x_min) / 2 ** B
    offset = 0 - x_min
    x_hat = q * delta + delta / 2 - offset

    return x_hat

def bit_slicing(q, b): # Input : quantization index q; siliced quantization index q_l
    b_IQ = b * 2 # to consider odd and even bits
    L = b.numel()
    c_l = torch.zeros(L + 1)
    for l in range(1, L + 1):
        for i in range(l):
            c_l[l] += b_IQ[i]
    q_l = torch.zeros(L)
    for l in range(1, L + 1):
        q_l[l - 1] = torch.floor(q / 2 ** c_l[l - 1]) - 2 ** b_IQ[l - 1] * torch.floor(q / 2 ** c_l[l])

    return q_l

def reconstruction(q_l, b, x_min, delta):
    b_IQ = b * 2 # to consider odd and even bits
    L = b.numel()
    x_hat = 0
    c_l = torch.zeros(L + 1)
    for l in range(1, L + 1):
        for i in range(l):
            c_l[l] += b_IQ [i]

    for l in range(1, L + 1):
        x_hat += 2 ** c_l[l - 1] * q_l[l - 1] * delta
    x_hat = x_hat + delta / 2 + x_min

    return x_hat

import torch


def decimal_to_binary_auto(x, b) -> torch.Tensor: # Input : number x; Output : binary
    """
    10진수 x를 B = 2b 비트 이진수 텐서로 변환 (MSB first).
    항상 길이 B짜리 결과 반환.
    """
    B = 2 * b
    x_int = int(x)

    if x_int < 0:
        raise ValueError("음수는 지원하지 않아요 😢")

    bits = [(x_int >> i) & 1 for i in reversed(range(B))]  # MSB → LSB
    return torch.tensor(bits, dtype=torch.int)

def bit_segments_to_ints(x, b): # Input : q_odd or q_even bits; Output : int number from segemented bits
    start = 0
    result = []

    for length in b:
        bits = x[start:start + length].int()
        # 2진수 → 10진수
        val = 0
        for i, bit in enumerate(reversed(bits)):
            val += bit.item() * (2 ** i)
        result.append(val)
        start += length

    result_tensor = torch.tensor(result)

    return result_tensor

def constellation_mapper(q_odd_dec, q_even_dec, b): # Input : bit coordination tensor; Output : constellation point tensor
    L = b.numel()  # number of bit segments
    d = torch.zeros(L)  # distance between each point of the constellation
    for l in range(L):
        d[l] = torch.sqrt(6 / (4 ** b[l] - 1))

    m = torch.zeros(L, dtype=torch.cfloat)
    for l in range(L):
        m[l] = (q_odd_dec[l] - (2**b[l] - 1)/2) * d[l] + 1j*(q_even_dec[l] - (2**b[l] - 1)/2) * d[l]

    return m
