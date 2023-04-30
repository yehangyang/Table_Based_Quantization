---
comments: true
---

# 查表法量化激活函数

## 动机

首先交代一下激活函数量化的需求从何而来。为了提高模型推理速度，一些算子被做成量化算子，在整型域上进行计算（e.g. int8 量化，还有更激进的 int4 量化）。通常情况下线性计算实现量化是比较简单的，因为不是本文讨论重点，这里不做赘述。

而在模型设计中，线性层的后面通常会跟一个非线性层（也就是所谓的激活函数）。例如现在有下图（a）这样一个模型。如果只能对线性层做量化，那么只能得到下图（b）这样的效果，量化和非量化算子之间会穿插很多 Q/DQ 节点（也就是量化和浮点间的转化）。而我们希望看到下图（c）的效果，也就是希望激活函数也能进行量化计算。

![half quantization v.s. full quantizaiton](img/half_vs_full.drawio.svg)

观察上图（b），nonlinear 的输出会再次被量化，作为下一个量化 linear 的输入。细想，量化其实同时在边界上和数量上限制了表示范围。以 int8 量化为例，linear 输出的 $X \in [-128, +127]$（边界限制），可由 256 个数表示（数量限制）。此时如果 nonlinear 由浮点实现，则 $X$ 需要被反量化（$x' = scale_x · X$），才能作为浮点 nonlinear 的输入。请不要以为做了反量化变成浮点数了，$x'$ 的边界和数量限制就解除了，它仍旧被限制在 $[-128scale_x, +127scale_x]$，数值表示数量为 256 不变（并且可以和反量化前一一对应）。在浮点域做完 nonlinear 之后，$y = nonlinear(x')$，限制仍然存在（仍然保持一一对应）。为了能成为下一个量化 linear 的输入，需要对 $y$ 量化，$Y = [\frac {y} {scale_y}]$，限制依旧在（一一对应）。费劲周章，最后得到的还是 256 种数值结果，所以我们为什么不借助“一一对应”的传递性，用查表法来得到 nonlinear 的量化结果呢？

总结一下，我们的动机是希望通过量化非线性激活函数来实现全模型的量化（上图（b）到上图（c）的转变），并且根据分析是可以通过查表法来实现量化非线性激活函数，且可以保证无误差。

正如《动机》章节描述，我们将用查表法来实现非线性激活函数的量化。量化非线性激活函数是为了实现图中（c）的 nonlinear(int8) 等效替换（b）的 DQ -> nonlinear(float) -> Q 过程。

![wanted quantization look like](img/wanted_quantization.drawio.svg)

## 测试代码

代码地址

不同激活函数的查表实现可能存在差异，但是测试设计肯定是一致的，所以把测试设计放在功能设计前面。测试时需要保证在相同量化输入下，（b）和（c）两者量化输出的数值是完全相等的。如代码块所示，其中：

* L20 表示（c）的虚线框过程，量化结果为quant_output。
* L23~L25 表示（b）的虚线框过程，量化结果为ground_truth_quant_output。
* L28 是元素比较，quant_output应该与ground_truth_quant_output完全相等。

```python linenums="1"
def __check_symmetric_quant(quant_cls: _SymmetryQuant, float_func: Callable, input_amax: float, bit: int,
                            narrow: bool) -> bool:
    """Check whether the output of quant_cls is correct

    Args:
        quant_cls (_SymmetryQuant): an symmetric quantization operator
        float_func (Callable): ground truth function in floating-point
        input_amax (float): the amax of input for quantization
        bit (int): the bit number of quantization
        narrow (bool): Ture: quant_min = -2^(bit - 1) + 1, False: quant_min = 2^(bit - 1)

    Returns:
        bool: Ture: all elements of quantization function output are correct, False: any elements is wrong
    """
    input_shape = (1, 128)
    quant_input = torch.randint(utils.quant_min(bit, narrow), utils.quant_max(bit) + 1, input_shape, dtype=torch.int8)

    # (quant_input) -> quant_func -> (quant_output)
    quant_func = quant_cls(input_amax, bit, narrow)
    quant_output = quant_func(quant_input)

    # (quant_input) -> DQ -> float_func -> Q -> (quant_output)
    ground_truth_float_input = utils.dequantize(quant_input, quant_func.input_scale)
    ground_truth_float_output = float_func(ground_truth_float_input)
    ground_truth_quant_output = utils.quantize(ground_truth_float_output, quant_func.output_scale, bit, narrow)

    # every element should be the same
    return (quant_output == ground_truth_quant_output).all()
```

## 功能代码

代码地址
功能实现的关键是生成查表法用到的映射表。以对称量化为例，获取映射表 table 的流程如下图所示。

![table based quantization workflow](img/quantization_workflow.drawio.svg)

代码实现如下面代码块所示，其中：

* L21～L33：生成映射表 table 的代码（在__init__函数中可见）
* L36：前向推理时的查表过程

```python linenums="1"
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
```

至此完成非线性激活函数量化查表法的最普适实现，i.e.，适合任何一个非线性激活函数的量化实现（除了 Softmax）。在同一个网络中，可能会包含多个非线性激活函数。如果每个激活函数都使用查表法实现，以 int8 量化为例，每个激活函数的查表实现需要花费 256 Byte 的内存。如果激活函数数量稍多一点，内存占用还是比较厉害的。不过如果我们稍加限制，譬如在同一个网络中，同一种激活函数的输入/输出 scale 能分别限制为相同，或者限制在少量组合中，那么查表法使用的表格数量就能减少。e.g., 一个网络中有 30 个 Sigmoid 激活函数，其中 10 个的输入/输出 scale 是相同的，那么就能共用一个表；如果条件允许，30 个 Sigmoid 的输入/输出 scale 都分别相同，那就都能共用同一个表，内存占用一下子缩小 30 倍。

话说回来，这个内存占用与一个卷积核 [N, C, H, W] = [4, 8, 3, 3] 差不多（同样以 int8 量化来比较），而通常的卷积核是该例子的好几倍。

接下来的小节会根据各激活函数的特性，给出一些特例设计，试图进一步提升推理性能（通常是内存和计算之间的博弈）。还会讨论如何实现 Softmax 的量化。
