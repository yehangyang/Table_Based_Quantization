import unittest
from typing import Callable, Tuple

import torch
from quantized.module import utils
from quantized.module.activation import _SymmetryQuantTable, Sigmoid, TanH


def __check_symmetric_quant(quant_cls: _SymmetryQuantTable, float_func: Callable, input_amax: float, bit: int,
                            narrow: bool, output_amax: float, output_unsign: bool) -> bool:
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
    quant_input = torch.randint(utils.quant_min(bit, narrow), utils.quant_max(bit) + 1, input_shape, dtype=torch.int8)

    # (quant_input) -> quant_func -> (quant_output)
    quant_func = quant_cls(input_amax, bit, narrow, output_amax, output_unsign)
    quant_output = quant_func(quant_input)

    # (quant_input) -> DQ -> float_func -> Q -> (quant_output)
    ground_truth_float_input = utils.dequantize(quant_input, quant_func.input_scale)
    ground_truth_float_output = float_func(ground_truth_float_input)
    ground_truth_quant_output = utils.quantize(ground_truth_float_output, quant_func.output_scale, bit, narrow,
                                               output_unsign)

    # every element should be the same
    return (quant_output == ground_truth_quant_output).all()


def _check_symmetric_quant(quant_cls: _SymmetryQuantTable,
                           float_func: Callable,
                           input_amax_range: Tuple[float],
                           output_amax_range: Tuple[float],
                           output_unsign_range: Tuple[bool] = (False,)):
    for _ in range(100):
        for input_amax in input_amax_range:
            for bit in (8, 4):
                for narrow in (True, False):
                    for output_amax in output_amax_range:
                        for output_unsign in output_unsign_range:
                            if not __check_symmetric_quant(quant_cls, float_func, input_amax, bit, narrow, output_amax,
                                                           output_unsign):
                                print(f'input_amax = {input_amax}, bit = {bit}, narrow = {narrow} is FAIL!')
                                return False

    return True


class TestActivation(unittest.TestCase):

    def test_sigmoid(self):
        self.assertEqual(
            _check_symmetric_quant(Sigmoid, torch.sigmoid, (4, 5, 6, 7, 8), (0.5, 1, 1.5, None), (True, False)), True)

    def test_tanh(self):
        self.assertEqual(_check_symmetric_quant(TanH, torch.tanh, (2, 3, 4, 5, 6), (0.5, 1, 1.5, None)), True)


if __name__ == '__main__':
    unittest.main()
