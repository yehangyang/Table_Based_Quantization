import unittest
from typing import Callable

import torch
from quantized.module import activation, utils


def two_side_template(quant_func_class: activation._TwoSideGradientEqualsZero, float_func: Callable, high_range: tuple):
    for _ in range(100):
        for high in high_range:
            for bit in (4, 8):
                input_shape = (1, 128)
                quant_input = torch.randint(
                    utils.quant_min(bit), utils.quant_max(bit)+1, input_shape)

                # quant_input -> quant_sigmoid -> quant_output
                quant_sigmoid = quant_func_class(high, bit)
                quant_output = quant_sigmoid(quant_input)

                # quant_input -> DQ -> float_sigmoid -> Q -> quant_output
                ground_truth_float_input = utils.dequantize(
                    quant_input, quant_sigmoid.input_scale)
                ground_truth_float_output = float_func(
                    ground_truth_float_input)
                ground_truth_quant_output = utils.quantize(
                    ground_truth_float_output, quant_sigmoid.output_scale)

                # every element should be the same
                if not (quant_output == ground_truth_quant_output).all():
                    print(f'high = {high}, bit = {bit} is FAIL!')
                    return False
    return True


class TestActivation(unittest.TestCase):

    def test_sigmoid(self):
        self.assertEqual(two_side_template(
            activation.Sigmoid, torch.sigmoid, (4, 8)), True)

    def test_tanh(self):
        self.assertEqual(two_side_template(
            activation.TanH, torch.tanh, (4, 8)), True)


if __name__ == '__main__':
    unittest.main()
