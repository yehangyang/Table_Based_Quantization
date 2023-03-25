import torch


def quant_min(bit: int):
    return -(1 << (bit-1))


def quant_max(bit: int):
    return (1 << (bit-1))-1


def quantize(x: torch.Tensor, scale: float):
    return (x/scale).to(torch.int8)


def dequantize(x: torch.Tensor, scale: float):
    return x*scale
