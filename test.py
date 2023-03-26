import unittest
from typing import Callable

import torch
from quantized.module import utils
from quantized.module.activation import _SymmetryQuant, Sigmoid, TanH


def __check_symmetric_quant(quant_func_class: _SymmetryQuant, float_func: Callable, input_amax: tuple, bit: int,
                            narrow: bool):
    input_shape = (1, 128)
    quant_input = torch.randint(utils.quant_min(bit, narrow), utils.quant_max(bit) + 1, input_shape, dtype=torch.int8)

    # quant_input -> quant_func -> quant_output
    quant_func = quant_func_class(input_amax, bit, narrow)
    quant_output = quant_func(quant_input)

    # quant_input -> DQ -> float_func -> Q -> quant_output
    ground_truth_float_input = utils.dequantize(quant_input, quant_func.input_scale)
    ground_truth_float_output = float_func(ground_truth_float_input)
    ground_truth_quant_output = utils.quantize(ground_truth_float_output, quant_func.output_scale, bit, narrow)

    # every element should be the same
    return (quant_output == ground_truth_quant_output).all()


def _check_symmetric_quant(quant_func_class: _SymmetryQuant, float_func: Callable, input_amax_range: tuple):
    for _ in range(100):
        for input_amax in input_amax_range:
            for bit in (4, 8):
                for narrow in (True, False):
                    if not __check_symmetric_quant(quant_func_class, float_func, input_amax, bit, narrow):
                        print(f'input_amax = {input_amax}, bit = {bit}, narrow = {narrow} is FAIL!')
                        return False

    return True


class TestActivation(unittest.TestCase):

    def test_sigmoid(self):
        self.assertEqual(_check_symmetric_quant(Sigmoid, torch.sigmoid, (4, 5, 6, 7, 8)), True)

    def test_tanh(self):
        self.assertEqual(_check_symmetric_quant(TanH, torch.tanh, (2, 3, 4, 5, 6)), True)


if __name__ == '__main__':
    unittest.main()
