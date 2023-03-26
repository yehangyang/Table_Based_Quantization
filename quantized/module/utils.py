import torch


def quant_min(bit: int, narrow: bool = False):
    return -(1 << (bit - 1)) + 1 if narrow else -(1 << (bit - 1))


def quant_max(bit: int):
    return (1 << (bit - 1)) - 1


def quantize(x: torch.Tensor, scale: float, bit: int, narrow: bool = False):
    quantized_x = torch.clamp(x / scale, quant_min(bit, narrow), quant_max(bit))
    return quantized_x.to(torch.int8)


def dequantize(x: torch.Tensor, scale: float):
    return x * scale
