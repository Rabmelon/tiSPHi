import taichi as ti
import numpy as np
from functools import reduce    # 整数：累加；字符串、列表、元组：拼接。lambda为使用匿名函数
from show import *

ti.init(arch=ti.cpu)

m = ti.field(float, 10)
tmp = ti.Vector([1, 2])

@ti.kernel
def cubic_kernel_derivative(r: ti.types.ndarray()):
    res = ti.Vector([0.0 for _ in range(2)])
    k = 40 / 7 / np.pi
    k *= 6. / 12**2
    r_norm = r.norm()
    q = r_norm / 12
    if r_norm > 1e-5 and q <= 1.0:
        grad_q = r / (r_norm * 12)
        if q <= 0.5:
            res = k * q * (3.0 * q - 2.0) * grad_q
        else:
            factor = 1.0 - q
            res = k * (-factor * factor) * grad_q
    return res

# @ti.kernel
# def foo():
    # print(cubic_kernel_derivative(tmp))

# @ti.kernel
# def test():
    # m[0] = tmp.transpose() @ cubic_kernel_derivative(tmp)
    # print(m)


if __name__ == "__main__":
    print("hallo tiSPHi!")
    # foo()
    # test()
    tmp1 = cubic_kernel_derivative(tmp)
    print(tmp1)
