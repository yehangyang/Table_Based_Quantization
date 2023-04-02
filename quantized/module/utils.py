import torch


def quant_min(bit: int, narrow: bool = False, unsign: bool = False):
    if unsign:
        return 0
    else:
        return -(1 << (bit - 1)) + 1 if narrow else -(1 << (bit - 1))


def quant_max(bit: int, unsign: bool = False):
    return (1 << bit) - 1 if unsign else (1 << (bit - 1)) - 1


def quantize(x: torch.Tensor, scale: float, bit: int, narrow: bool = False, unsign: bool = False):
    quantized_x = torch.clamp(x / scale, quant_min(bit, narrow, unsign), quant_max(bit, unsign))
    return quantized_x.to(torch.uint8 if unsign else torch.int8)


def dequantize(x: torch.Tensor, scale: float):
    return x * scale
