import taichi as ti
import numpy as np
from functools import reduce    # 整数：累加；字符串、列表、元组：拼接。lambda为使用匿名函数
from show import *

ti.init(arch=ti.cpu)

m = ti.field(float, 10)
v1 = ti.Vector([1, 2])
v2 = ti.Vector([1.5, 2.2])
r = ti.Vector([0.05, 0.02])


@ti.func
def cubic_kernel_derivative(r):
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

@ti.kernel
def foo():
    v_xy = (v1 - v2).dot(r)
    tmp1 = v_xy * cubic_kernel_derivative(r)
    tmp2 = r.transpose() @ cubic_kernel_derivative(v1-v2)
    m[0] = tmp2[0]
    print(v_xy)
    print(cubic_kernel_derivative(r))
    print(tmp1)
    print(tmp2)
    print(m[0])

# @ti.kernel
# def test():
    # m[0] = tmp.transpose() @ cubic_kernel_derivative(tmp)
    # print(m)


if __name__ == "__main__":
    print("hallo tiSPHi!")
    foo()
    # test()
