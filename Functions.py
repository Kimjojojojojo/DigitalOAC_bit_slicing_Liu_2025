import torch

def generate_rayleigh(row, col: int, sigma) -> torch.Tensor:
    """
    Rayleigh fading을 따르는 복소수 채널 K개 생성
    각 채널은 h ~ CN(0,1) → Rayleigh 분포 magnitude
    """
    real = sigma * torch.randn(row, col) / torch.sqrt(torch.tensor(2.0))
    imag = sigma * torch.randn(row, col) / torch.sqrt(torch.tensor(2.0))
    h = torch.complex(real, imag)

    return h

def exponential_integration(k, steps=100000):
    # 단순한 사다리꼴 적분법으로 torch 버전 구현
    x = torch.linspace(k, 100.0, steps)  # 무한대를 대체하는 큰 값 (100 이상이면 충분)
    dx = x[1] - x[0]
    y = (1 / x) * torch.exp(-x)
    result = torch.trapz(y, dx=dx)

    return result.item(), 0  # torch에선 에러 추정치 없음