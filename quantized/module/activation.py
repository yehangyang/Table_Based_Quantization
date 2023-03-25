from typing import Callable
import torch
from .utils import quant_min, quant_max


class _TwoSideGradientEqualsZero(torch.nn.Module):
    def __init__(self, func: Callable, high: float, bit: int, is_symmetric: bool = True) -> None:
        super().__init__()
        self.__input_scale = high/quant_max(bit)
        input_q = torch.arange(
            quant_min(bit), quant_max(bit)+1, dtype=torch.int8)
        input_f = input_q*self.__input_scale

        output_f = func(input_f)
        self.__output_scale = output_f.max()/quant_max(bit)
        output_q = output_f/self.__output_scale
        output_q = output_q.to(torch.int8)
        self._table = torch.cat(
            (output_q[(quant_max(bit)+1):], output_q[:(quant_max(bit)+1)]))

    def forward(self, x: torch.Tensor):
        y = self._table[x.int()]
        return y

    @property
    def input_scale(self):
        return self.__input_scale

    @property
    def output_scale(self):
        return self.__output_scale


class Sigmoid(_TwoSideGradientEqualsZero):
    def __init__(self, high: float, bit: int, is_symmetric: bool = True) -> None:
        super().__init__(torch.sigmoid, high, bit, is_symmetric)


class TanH(_TwoSideGradientEqualsZero):
    def __init__(self,  high: float, bit: int, is_symmetric: bool = True) -> None:
        super().__init__(torch.tanh, high, bit, is_symmetric)
