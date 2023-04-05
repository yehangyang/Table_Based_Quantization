from typing import Callable
import torch
from .utils import QuantConfig, quant_max


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
            input_bit (int): bit number of input for quantization
            input_amax (float): amax of input for quantization
            input_unsign (bool): True: quant_input is in unsign integer, False is Default, in sign integer
            output_bit (int): bit number of output for quantization
            output_amax (float, optional): the amax of output for quantization.
                                           Defaults to None, = amax(nonlinear(DQ(quant_input)))
            output_unsign (bool, optional): True: quant_input is in unsign integer, False is Default, in sign integer. Defaults to False.
            narrow (bool, optional): True: quant_min = -2^(bit - 1) + 1, False: quant_min = -2^(bit - 1). Defaults to False.
        """
        super().__init__()
        assert (input_bit <= 8)

        # (input_quant) -> DQ -> (input_float)
        self.input_qconfig = QuantConfig(bit=input_bit, narrow=narrow, unsign=input_unsign, amax=input_amax)
        input_quant = self.input_qconfig.range
        input_float = self.input_qconfig.dequantize(input_quant)

        # (input_float) -> float_func -> Q -> (output_quant)
        output_float = func(input_float)
        output_amax = output_amax if output_amax else torch.absolute(output_float).max()
        self.output_qconfig = QuantConfig(bit=output_bit, narrow=narrow, unsign=output_unsign, amax=output_amax)
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


class Softmax(torch.nn.Module):

    def __init__(self,
                 dim_len: int,
                 input_bit: int,
                 input_amax: float,
                 input_unsign: bool,
                 output_bit: int,
                 output_amax: float,
                 output_unsign: bool = True,
                 acc_bit: int = 16,
                 narrow: bool = False,
                 dim: int = None) -> None:
        super().__init__()

        assert (input_bit <= 8)

        self._dim = dim if dim else -1
        self._dim_len = dim_len

        self.input_qconfig = QuantConfig(bit=input_bit, narrow=narrow, unsign=input_unsign, amax=input_amax)
        self.output_qconfig = QuantConfig(bit=output_bit, narrow=narrow, unsign=output_unsign, amax=output_amax)

        # (input_quant) -> minus_max -> DQ -> float_func -> (exp_float)
        input_quant = self.input_qconfig.range.to(torch.int32)
        input_quant_minus_max = input_quant - self.input_qconfig.quant_max
        input_float_minus_max = self.input_qconfig.dequantize(input_quant_minus_max)
        exp_float = torch.exp(input_float_minus_max)

        acc_quant_max = quant_max(bit=acc_bit, unsign=False)

        # denominator
        denominator_scale = 1 / (acc_quant_max // dim_len)  # denominator allowed min quant scale
        self.denominator_element_qconfig = QuantConfig(bit=acc_bit, narrow=False, unsign=False, scale=denominator_scale)
        # (exp_float) -> Q -> (denominator_element_quant)
        denominator_element_quant = self.denominator_element_qconfig.quantize(exp_float)

        # numerator
        numerator_bit = acc_bit + output_bit
        numerator_scale = denominator_scale * self.output_qconfig.scale
        self.numerator_qconfig = QuantConfig(bit=numerator_bit, narrow=False, unsign=False, scale=numerator_scale)
        # (exp_float) -> Q -> (numerator_quant)
        numerator_quant = self.numerator_qconfig.quantize(exp_float)

        # adjust sequence of output_quant for easier retrieve
        if input_unsign:
            self._denominator_element_table = denominator_element_quant
            self._numerator_table = numerator_quant
        else:
            index = self.input_qconfig.quant_max if narrow else self.input_qconfig.quant_max + 1
            self._denominator_element_table = torch.cat(
                (denominator_element_quant[index:], denominator_element_quant[:index]))
            self._numerator_table = torch.cat((numerator_quant[index:], numerator_quant[:index]))

    def forward(self, x: torch.Tensor):
        denominator_element = self._denominator_element_table[x.to(torch.int64)]
        denominator = torch.sum(denominator_element, dim=self._dim)

        numerator = self._numerator_table[x.to(torch.int64)]

        y = numerator / denominator
        y = torch.clamp(y, self.output_qconfig.quant_min, self.output_qconfig.quant_max)
        y = y.to(self.output_qconfig.dtype)
        return y
