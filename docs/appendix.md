# 附录

## 量化术语

* 大写字母表示整数 tensor
* 小写字母表示浮点 tensor
* $scale$ 表示量化缩放系数
* $zero\_point$ 表示量化零点偏移（本文先把问题简单化，只考虑对称量化，不考虑零点偏移）
* $bit$ 表示量化类型位数
* 量化后数据范围 $[Q^{min}, Q^{max}]$
* 对称量化的 $scale = \frac {F^{max}} {Q^{max}}$，其中 $F^{max}$ 表示浮点最大值，由 PTQ 确定，$Q^{max}$ 由 $bit$ 确定
* 量化（Q）：$X_i = clamp([x_i/scale_x]; Q^{min}, Q^{max})$
* 反量化（DQ）：$x'_i = scale_x X_i \approx x_i$
