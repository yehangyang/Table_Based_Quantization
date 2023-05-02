---
comments: true
---

# 查表法量化 TanH

## 基本信息

表达公式：$y_i = \frac {e^{x_i} - e^{-x_i}} {e^{x_i} + e^{-x_i}}$

函数曲线：

<iframe 
    src="https://www.desmos.com/calculator/xlyvzxvzny?embed" 
    width="800" 
    height="300" 
    style="border: 1px solid #ccc" 
    frameborder=0>
</iframe>

## 常规实现

常规的量化 TanH 可以直接根据《查表法 · 量化激活函数》中描述的实现。这里直接给出功能代码和测试代码。

## 特殊变形

观察 TanH 的函数曲线，可以发现 TanH 是一个中心对称函数。对称本是一种很美好数学特性。借助中心对称这个特性，可以通过一些简单的数学运算换回映射表一半的内存占用。根据 TanH 的数据表达公式 $y_i = \frac {e^{x_i} - e^{-x_i}} {e^{x_i} + e^{-x_i}}$，代入输入量化公式 $x_i = scale_x X_i$，代入输出量化公式 y = sy _ Y 可得，
sy _ Y = (exp(sx _ X) - exp(-sx _ X)) / (exp(sx _ X) + exp(-sx _ X))
Y = (1 / sy) _ (exp(sx _ X) - exp(-sx _ X)) / (exp(sx _ X) + exp(-sx \* X))
即映射表为 X -> Y。假设当前只有 X >= 0 的映射表，其映射关系用 project 表示。那么根据中心对称特性，

- 当 X < 0 时，Y = - project(-X)
- 当 X >= 0 时，Y = project(X)

## 功能代码

```python
class TanHHalfTable(torch.nn.Module):

    def __init__(self,
                 input_bit: int,
                 input_amax: float,
                 input_unsign: bool,
                 output_bit: int,
                 output_amax: float = None,
                 *args,
                 **kwargs) -> None:
        super().__init__()
        assert (input_bit <= 8)
        narrow = True

        # (input_quant) -> DQ -> (input_float)
        self.input_qconfig = QuantConfig(bit=input_bit, narrow=narrow, unsign=input_unsign, amax=input_amax)
        input_quant_positive = torch.arange(0, self.input_qconfig.quant_max + 1, dtype=self.input_qconfig.dtype)

        input_float_positive = self.input_qconfig.dequantize(input_quant_positive)

        # (input_float) -> float_func -> Q -> (output_quant)
        output_float_positive = torch.tanh(input_float_positive)
        output_amax = output_amax if output_amax else torch.absolute(output_float_positive).max()
        self.output_qconfig = QuantConfig(bit=output_bit, narrow=narrow, unsign=False, amax=output_amax)
        output_quant_positive = self.output_qconfig.quantize(output_float_positive)

        self._table = output_quant_positive

    def forward(self, x: torch.Tensor):
        y = torch.zeros_like(x, dtype=self.output_qconfig.dtype)
        positive_mask = x >= 0
        y[positive_mask] = self._table[x[positive_mask].to(torch.int64)]

        negative_mask = x < 0
        y[negative_mask] = -self._table[(-x[negative_mask]).to(torch.int64)]
        return y
```

## 测试代码

```python
def test_tanh_half_table(self):
        self.assertEqual(
            _check_symmetric_quant_table(quant_cls=TanHHalfTable,
                                         float_func=torch.tanh,
                                         input_bit_range=(8, 4),
                                         input_amax_range=(2, 3, 4, 5, 6),
                                         input_unsign_range=(False, True),
                                         output_bit_range=(8, 4),
                                         output_amax_range=(0.5, 1, 1.5, None),
                                         output_unsign_range=(False,)), True)
```

## 常规 v.s. 变形

常规：指常规实现，使用一张完整的映射表

变形：指特殊变形，使用半张映射表，伴随少量简单计算

比较项 常规实现 特殊变形 说明
内存占用 2^input*quant_bit * outut*quant_bit / 8 (Bytes) 2^(input_quant_bit - 1)* outut_quant_bit / 8 (Bytes) 内存占用特殊变形可以节省一半
额外计算 只需要查表 X < 0 的元素，查表前需要取反，查表后需要取反 部分元素多两次取反操作
