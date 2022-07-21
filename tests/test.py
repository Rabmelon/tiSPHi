import taichi as ti
import numpy as np

ti.init(debug=True)

@ti.func
def funtest(pos, t):
    res = ti.Matrix([[1, 2*ti.sin(t), 0.5], [-0.333, 1, -ti.sin(t)], [2*pos[0]*ti.sin(2*t), 0, 1.5]])
    return res

@ti.func
def cal_db(a, b):
    dim = 2
    tmp = ti.Matrix([[a[i,j]*b[i,j] for i in range(dim)] for j in range(dim)])
    res = tmp.sum()
    return res

@ti.func
def cal_I1(stress):
    res = stress.trace()
    return res

@ti.func
def cal_sJ2(s):
    res = ti.sqrt(0.5 * (s*s).sum())
    return res

@ti.func
def cal_fDP(I1, sJ2, a_f, k_c):
    res = sJ2 + a_f * I1 - k_c
    return res


@ti.kernel
def foo():
    I1 = cal_I1(stress)
    print("foo 1", I1)
    sJ2 = cal_sJ2(s)
    print("foo 2", sJ2)
    fDP = cal_fDP(I1, sJ2, af, kc)
    print("foo 3", fDP)

if __name__ == "__main__":
    print("hallo test kernel function accuracy!")

    I = ti.Matrix(np.eye(2))
    f_v = ti.Matrix([[1, 2], [3, 4], [9, 1], [2, 2]])
    kd = ti.Vector([2, 3])

    rfvkd = f_v @ kd / 2
    print("rfvkd", rfvkd)

    LLL = ti.Matrix([[1, 2], [3, 4]])

    # DDD = ti.Vector([LLL[i]+LLL[j] for i,j in zip(range(2), range(2))])
    DDD = 0.5 * ti.Matrix([[LLL[i, j]+LLL[j, i] for i in range(2)] for j in range(2)])

    print("DDD", DDD)
    print("DDD", DDD.sum())

    a = ti.Matrix([[1, 2], [3, 4]])
    b = ti.Matrix([[2, 4], [3, 1]])
    # foo()
    print("a*b", (a*b).sum())
    print("bij-bmmdeltaij", b-b.trace()*I)


    stress = ti.Matrix([[1, 2], [3, 4]])
    s = stress - stress.trace() / 3 * I
    af = 0.24
    kc = 2.0
    foo()

    flag_end = 1
