import torch


class QuantConfig:

    def __init__(self, amax: float, bit: int, narrow: bool, unsign: bool) -> None:
        if bit > 8:
            assert (unsign == False)
        else:
            assert (unsign in (False, True))

        self._amax = amax
        self._bit = bit
        self._narrow = narrow
        self._unsign = unsign

    @property
    def quant_min(self):
        if self._unsign:
            return 0
        else:
            return -(1 << (self._bit - 1)) + 1 if self._narrow else -(1 << (self._bit - 1))

    @property
    def quant_max(self):
        return (1 << self._bit) - 1 if self._unsign else (1 << (self._bit - 1)) - 1

    @property
    def scale(self):
        return self._amax / self.quant_max

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

    def randint(self, shape):
        return torch.randint(self.quant_min, self.quant_max + 1, shape, dtype=self.dtype)

    def quantize(self, x: torch.Tensor):
        quantized_x = torch.clamp(x / self.scale, self.quant_min, self.quant_max)
        return quantized_x.to(self.dtype)

    def dequantize(self, x: torch.Tensor):
        return x * self.scale
