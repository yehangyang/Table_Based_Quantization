import unittest
from typing import Callable, Tuple
from tqdm import tqdm

import torch
from quantized.module import utils
from quantized.module.activation import _SymmetryQuantTable, Sigmoid, TanH, TanHHalfTable, Softmax


def __check_symmetric_quant_table(quant_cls: _SymmetryQuantTable,
                                  float_func: Callable,
                                  input_bit: int,
                                  input_amax: float,
                                  input_unsign: bool,
                                  output_bit: int,
                                  output_amax: float = None,
                                  output_unsign: bool = False,
                                  narrow: bool = False) -> bool:
    """Check whether the output of quant_cls is correct

    Args:
        quant_cls (_SymmetryQuantTable): an symmetric quantization operator
        float_func (Callable): ground truth function in floating-point
        input_amax (float): the amax of input for quantization
        bit (int): the bit number of quantization
        narrow (bool): Ture: quant_min = -2^(bit - 1) + 1, False: quant_min = 2^(bit - 1)
        output_amax (float): the amax of output for quantization
        output_unsign (bool): True: quant_output is in unsign integer, False: in sign integer

    Returns:
        bool: Ture: all elements of quantization function output are correct, False: any elements is wrong
    """

    input_shape = (1, 128)

    quant_func = quant_cls(input_bit, input_amax, input_unsign, output_bit, output_amax, output_unsign, narrow)
    input_qconfig: utils.QuantConfig = quant_func.input_qconfig
    output_qconfig: utils.QuantConfig = quant_func.output_qconfig

    quant_input = input_qconfig.randint(input_shape)

    # (quant_input) -> quant_func -> (quant_output)
    quant_output = quant_func(quant_input)

    # (quant_input) -> DQ -> float_func -> Q -> (quant_output)
    ground_truth_float_input = input_qconfig.dequantize(quant_input)
    ground_truth_float_output = float_func(ground_truth_float_input)
    ground_truth_quant_output = output_qconfig.quantize(ground_truth_float_output)

    # every element should be the same
    return (quant_output == ground_truth_quant_output).all()


def _check_symmetric_quant_table(quant_cls: _SymmetryQuantTable,
                                 float_func: Callable,
                                 input_bit_range: Tuple[int],
                                 input_amax_range: Tuple[float],
                                 input_unsign_range: Tuple[bool],
                                 output_bit_range: Tuple[int],
                                 output_amax_range: Tuple[float],
                                 output_unsign_range: Tuple[bool] = (False,)):
    for _ in tqdm(range(100), desc=f'Testing {quant_cls}'):
        for input_bit in input_bit_range:
            for input_amax in input_amax_range:
                for input_unsign in input_unsign_range:
                    for output_bit in output_bit_range:
                        for output_amax in output_amax_range:
                            for output_unsign in output_unsign_range:
                                for narrow in (True, False):
                                    if not __check_symmetric_quant_table(quant_cls, float_func, input_bit, input_amax,
                                                                         input_unsign, output_bit, output_amax,
                                                                         output_unsign, narrow):
                                        print(f'input_bit = {input_bit}, input_amax = {input_amax}, input_unsign = {input_unsign}, '\
                                              f'output_bit = {output_bit}, output_amax = {output_amax}, output_unsign = {output_unsign}, '\
                                              f'narrow = {narrow} is FAIL!')
                                        return False

    return True


def __check_symmetric_quant_table_softmax(dim_len: int,
                                          input_bit: int,
                                          input_amax: float,
                                          input_unsign: bool,
                                          output_bit: int,
                                          output_amax: float = None,
                                          output_unsign: bool = False,
                                          narrow: bool = False) -> bool:
    """Check whether the output of quant_cls is correct

    Args:
        quant_cls (_SymmetryQuantTable): an symmetric quantization operator
        float_func (Callable): ground truth function in floating-point
        input_amax (float): the amax of input for quantization
        bit (int): the bit number of quantization
        narrow (bool): Ture: quant_min = -2^(bit - 1) + 1, False: quant_min = 2^(bit - 1)
        output_amax (float): the amax of output for quantization
        output_unsign (bool): True: quant_output is in unsign integer, False: in sign integer

    Returns:
        bool: Ture: all elements of quantization function output are correct, False: any elements is wrong
    """

    input_shape = (1, 1000)

    quant_func = Softmax(dim_len=dim_len,
                         input_bit=input_bit,
                         input_amax=input_amax,
                         input_unsign=input_unsign,
                         output_bit=output_bit,
                         output_amax=output_amax,
                         output_unsign=output_unsign,
                         narrow=narrow)
    input_qconfig: utils.QuantConfig = quant_func.input_qconfig
    output_qconfig: utils.QuantConfig = quant_func.output_qconfig

    quant_input = input_qconfig.randint(input_shape)

    # (quant_input) -> quant_func -> (quant_output)
    quant_output = quant_func(quant_input)

    # (quant_input) -> DQ -> float_func -> Q -> (quant_output)
    ground_truth_float_input = input_qconfig.dequantize(quant_input)
    ground_truth_float_output = torch.softmax(ground_truth_float_input, -1)
    ground_truth_quant_output = output_qconfig.quantize(ground_truth_float_output)

    # every element should be the same
    max_absolute_error = torch.max(torch.absolute(quant_output - ground_truth_quant_output))
    return max_absolute_error


def _check_symmetric_quant_table_softmax(dim_len_range: Tuple[int],
                                         input_bit_range: Tuple[int],
                                         input_amax_range: Tuple[float],
                                         input_unsign_range: Tuple[bool],
                                         output_bit_range: Tuple[int],
                                         output_amax_range: Tuple[float],
                                         output_unsign_range: Tuple[bool] = (False,)):
    for dim_len in tqdm(range(dim_len_range[0], dim_len_range[1]), desc=f'Testing {Softmax}'):
        for input_bit in input_bit_range:
            for input_amax in input_amax_range:
                for input_unsign in input_unsign_range:
                    for output_bit in output_bit_range:
                        for output_amax in output_amax_range:
                            for output_unsign in output_unsign_range:
                                for narrow in (True, False):
                                    max_absolute_error = __check_symmetric_quant_table_softmax(
                                        dim_len, input_bit, input_amax, input_unsign, output_bit, output_amax,
                                        output_unsign, narrow)
                                    if max_absolute_error > 1:
                                        print(f'dim_len = {dim_len}, input_bit = {input_bit}, input_amax = {input_amax}, input_unsign = {input_unsign}, '\
                                                f'output_bit = {output_bit}, output_amax = {output_amax}, output_unsign = {output_unsign}, '\
                                                f'narrow = {narrow} max_absolute_error is {max_absolute_error}!')
                                        return False

    return True


class TestActivation(unittest.TestCase):

    # def test_sigmoid(self):
    #     self.assertEqual(
    #         _check_symmetric_quant_table(quant_cls=Sigmoid,
    #                                      float_func=torch.sigmoid,
    #                                      input_bit_range=(8, 4),
    #                                      input_amax_range=(4, 5, 6, 7, 8),
    #                                      input_unsign_range=(False, True),
    #                                      output_bit_range=(8, 4),
    #                                      output_amax_range=(0.5, 1, 1.5, None),
    #                                      output_unsign_range=(True, False)), True)

    def test_tanh(self):
        self.assertEqual(
            _check_symmetric_quant_table(quant_cls=TanH,
                                         float_func=torch.tanh,
                                         input_bit_range=(8, 4),
                                         input_amax_range=(2, 3, 4, 5, 6),
                                         input_unsign_range=(False, True),
                                         output_bit_range=(8, 4),
                                         output_amax_range=(0.5, 1, 1.5, None),
                                         output_unsign_range=(False,)), True)

    # def test_tanh_half_table(self):
    #     self.assertEqual(
    #         _check_symmetric_quant_table(quant_cls=TanHHalfTable,
    #                                      float_func=torch.tanh,
    #                                      input_bit_range=(8, 4),
    #                                      input_amax_range=(2, 3, 4, 5, 6),
    #                                      input_unsign_range=(False, True),
    #                                      output_bit_range=(8, 4),
    #                                      output_amax_range=(0.5, 1, 1.5, None),
    #                                      output_unsign_range=(False,)), True)

    # def test_softmax(self):
    #     self.assertEqual(
    #         _check_symmetric_quant_table_softmax(dim_len_range=(8, 1024),
    #                                              input_bit_range=(8, 4),
    #                                              input_amax_range=(4, 5, 6, 7, 8),
    #                                              input_unsign_range=(False, True),
    #                                              output_bit_range=(8, 4),
    #                                              output_amax_range=(0.5, 0.75, 1),
    #                                              output_unsign_range=(True, False)), True)


if __name__ == '__main__':
    unittest.main()
