import torch

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
    L = b.numel()
    c_l = torch.zeros(L + 1)
    for l in range(1, L + 1):
        for i in range(l):
            c_l[l] += b[i]
    print(c_l)
    q_l = torch.zeros(L)
    for l in range(1, L + 1):
        q_l[l - 1] = torch.floor(q / 2 ** c_l[l - 1]) - 2 ** b[l - 1] * torch.floor(q / 2 ** c_l[l])

    return q_l

def reconstruction(q_l, b, x_min, delta):
    L = b.numel()
    x_hat = 0
    c_l = torch.zeros(L + 1)
    for l in range(1, L + 1):
        for i in range(l):
            c_l[l] += b[i]

    for l in range(1, L + 1):
        x_hat += 2 ** c_l[l - 1] * q_l[l - 1] * delta
    x_hat = x_hat + delta / 2 + x_min

    return x_hat