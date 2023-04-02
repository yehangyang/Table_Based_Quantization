from typing import Callable, Tuple
import torch
from .utils import QuantConfig


class _SymmetryQuantTable(torch.nn.Module):

    def __init__(self,
                 func: Callable,
                 input_bit: int,
                 input_amax: float,
                 input_unsign: bool,
                 output_bit: int,
                 output_amax: float = None,
                 output_unsign: bool = False,
                 narrow: bool = False) -> None:
        """Initialize quant-input to quant-output mapping table for symmetry quantization.

        Args:
            func (Callable): corresponding standard floating-point function
            input_amax (float): the amax of input for quantization
            bit (int): the bit number
            narrow (bool, optional): True: quant_min = -2^(bit - 1) + 1. Defaults to False, quant_min = -2^(bit - 1)
            output_amax (float, optional): the amax of output for quantization.
                                           Defaults to None, the amax = amax(nonlinear(DQ(quant_input)))
            output_unsign(bool, optional): True: quant_output is in unsign integer, False is Default, in sign integer
        """
        super().__init__()
        assert (input_bit <= 8)

        # (input_quant) -> DQ -> (input_float)
        self.input_qconfig = QuantConfig(input_amax, input_bit, narrow, input_unsign)
        input_quant = self.input_qconfig.range
        input_float = self.input_qconfig.dequantize(input_quant)

        # (input_float) -> float_func -> Q -> (output_quant)
        output_float = func(input_float)
        output_amax = output_amax if output_amax else torch.absolute(output_float).max()
        self.output_qconfig = QuantConfig(output_amax, output_bit, narrow, output_unsign)
        output_quant = self.output_qconfig.quantize(output_float)

        # adjust sequence of output_quant for easier retrieve
        if input_unsign:
            self._table = output_quant
        else:
            index = self.input_qconfig.quant_max if narrow else self.input_qconfig.quant_max + 1
            self._table = torch.cat((output_quant[index:], output_quant[:index]))

    def forward(self, x: torch.Tensor):
        y = self._table[x.to(torch.int64)]
        return y


class Sigmoid(_SymmetryQuantTable):

    def __init__(self,
                 input_bit: int,
                 input_amax: float,
                 input_unsign: bool,
                 output_bit: int,
                 output_amax: float = None,
                 output_unsign: bool = False,
                 narrow: bool = False,
                 *args,
                 **kwargs) -> None:
        super().__init__(torch.sigmoid, input_bit, input_amax, input_unsign, output_bit, output_amax, output_unsign,
                         narrow)


class TanH(_SymmetryQuantTable):

    def __init__(self,
                 input_bit: int,
                 input_amax: float,
                 input_unsign: bool,
                 output_bit: int,
                 output_amax: float = None,
                 narrow: bool = False,
                 *args,
                 **kwargs) -> None:
        super().__init__(torch.tanh, input_bit, input_amax, input_unsign, output_bit, output_amax, False, narrow)
