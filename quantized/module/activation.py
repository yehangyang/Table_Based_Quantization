from typing import Callable, Tuple
import torch
from .utils import quant_min, quant_max, quantize


class _SymmetryQuant(torch.nn.Module):

    def __init__(self,
                 func: Callable,
                 input_amax: float,
                 bit: int,
                 narrow: bool = False,
                 output_amax: float = None) -> None:
        """Initialize quant-input to quant-output mapping table for symmetry quantization.

        Args:
            func (Callable): corresponding standard floating-point function
            input_amax (float): the amax of input for quantization
            bit (int): the bit number
            narrow (bool, optional): True: quant_min = -2^(bit - 1) + 1. Defaults to False, quant_min = -2^(bit - 1)
            output_amax (float, optional): the amax of output for quantization.
                                           Defaults to None, the amax = amax(nonlinear(DQ(quant_input)))
        """
        super().__init__()
        # (input_quant) -> DQ -> (input_float)
        self.__input_scale = input_amax / quant_max(bit)
        input_quant = torch.arange(quant_min(bit, narrow), quant_max(bit) + 1, dtype=torch.int8)
        input_float = input_quant * self.__input_scale

        # (input_float) -> float_func -> Q -> (output_quant)
        output_float = func(input_float)
        output_amax = output_amax if output_amax else torch.absolute(output_float).max()
        self.__output_scale = output_amax / quant_max(bit)
        output_quant = quantize(output_float, self.__output_scale, bit, narrow)

        # adjust sequence of output_quant for easier retrieve
        index = quant_max(bit) if narrow else quant_max(bit) + 1
        self._table = torch.cat((output_quant[index:], output_quant[:index]))

    def forward(self, x: torch.Tensor):
        y = self._table[x.to(torch.int64)]
        return y

    @property
    def input_scale(self):
        return self.__input_scale

    @property
    def output_scale(self):
        return self.__output_scale


class Sigmoid(_SymmetryQuant):

    def __init__(self, input_amax: float, bit: int, narrow: bool = False) -> None:
        super().__init__(torch.sigmoid, input_amax, bit, narrow, 1.0)


class TanH(_SymmetryQuant):

    def __init__(self, input_amax: float, bit: int, narrow: bool = False) -> None:
        super().__init__(torch.tanh, input_amax, bit, narrow, 1.0)
