import torch


def quant_max(bit: int, unsign: bool):
    return (1 << bit) - 1 if unsign else (1 << (bit - 1)) - 1


class QuantConfig:

    def __init__(self, bit: int, narrow: bool, unsign: bool, amax: float = None, scale: float = None) -> None:
        if bit > 8:
            assert (unsign is False)
        else:
            assert (unsign in (False, True))

        assert (any((amax, scale)))

        self._bit = bit
        self._narrow = narrow
        self._unsign = unsign
        self._amax = amax
        self._scale = scale

    @property
    def quant_min(self):
        if self._unsign:
            return 0
        else:
            return -(1 << (self._bit - 1)) + 1 if self._narrow else -(1 << (self._bit - 1))

    @property
    def quant_max(self):
        return quant_max(self._bit, self._unsign)

    @property
    def scale(self):
        return self._scale if self._scale else self._amax / self.quant_max

    @property
    def dtype(self):
        if self._bit <= 8:
            return torch.uint8 if self._unsign else torch.int8
        elif self._bit <= 16:
            return torch.int16
        elif self._bit <= 32:
            return torch.int32
        else:
            return torch.int64

    @property
    def range(self):
        return torch.arange(self.quant_min, self.quant_max + 1, dtype=self.dtype)

    def set_scale(self, scale: float) -> None:
        self._scale = scale

    def randint(self, shape):
        return torch.randint(self.quant_min, self.quant_max + 1, shape, dtype=self.dtype)

    def quantize(self, x: torch.Tensor):
        quantized_x = torch.clamp(x / self.scale, self.quant_min, self.quant_max)
        return quantized_x.to(self.dtype)

    def dequantize(self, x: torch.Tensor):
        return x * self.scale
