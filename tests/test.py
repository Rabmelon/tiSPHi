import taichi as ti
import numpy as np

ti.init(debug=True)

epsilon = 1e-16
dim = 3
I = ti.Matrix(np.eye(dim))
# I = ti.Matrix([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
alpha_fric = 0.13685771015527778
k_c = 4.402384598296268

@ti.func
def cal_stress_s(stress):
    res = stress - stress.trace() / 3.0 * I
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
def cal_fDP(I1, sJ2):
    res = sJ2 + alpha_fric * I1 - k_c
    return res

@ti.func
def adapt_stress(stress, fDP_old):
    res = stress
    stress_s = cal_stress_s(stress)
    vI1 = cal_I1(stress)
    sJ2 = cal_sJ2(stress_s)
    fDP_new = cal_fDP(vI1, sJ2)
    dfDP = fDP_new - fDP_old
    print("fDP before", fDP_new)
    while fDP_new > epsilon:
        if fDP_new >= sJ2:
            res = adapt_1(res, vI1)
            print("adapt 1", res)
        else:
            res = adapt_2(stress_s, vI1, sJ2)
            print("adapt 2", res)
        stress_s = cal_stress_s(res)
        vI1 = cal_I1(res)
        sJ2 = cal_sJ2(stress_s)
        fDP_new = cal_fDP(vI1, sJ2)
        print("fDP after", fDP_new)
    return res

@ti.func
def adapt_1(stress, I1):
    tmp = (I1-k_c/alpha_fric) / 3.0
    res = stress - tmp * I
    return res

@ti.func
def adapt_2(s, I1, sJ2):
    r = (-I1 * alpha_fric + k_c) / sJ2
    res = r * s + I * I1 / 3.0
    return res

@ti.kernel
def foo():
    stress = adapt_stress(stress, fDP_old)
    # print("after adaptation", stress)

if __name__ == "__main__":
    print("hallo test here!")

    fDP_old = 0.0
    # stress_ini = [6, 2, 3, -1, 0, 0]  # xx, yy, zz, xy, xz, yz # no adapt elas
    # stress_ini = [6.2992, 4, 5, -2, 0, 0]  # xx, yy, zz, xy, xz, yz # no adapt plas
    # stress_ini = [16, 12, 10, -4, 0, 0]  # xx, yy, zz, xy, xz, yz # adapt 1
    # stress_ini = [6, 2, 10, -4, 0, 0]  # xx, yy, zz, xy, xz, yz # adapt 2
    # stress_ini = [24, 15, 0, 0, 0, 0]  # xx, yy, zz, xy, xz, yz # plain strain???

    # stress = ti.Matrix([[6.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [0.0, 0.0, 3.0]]) # no adapt elas
    # stress = ti.Matrix([[6.2993, -2.0, 0.0], [-2.0, 5.488, 0.0], [0.0, 0.0, 5.0]]) # no adapt plas
    stress = ti.Matrix([[16.0, -4.0, 0.0], [-4.0, 12.0, 0.0], [0.0, 0.0, 10.0]]) # adapt 1
    # stress = ti.Matrix([[6.0, -4.0, 0.0], [-4.0, 2.0, 0.0], [0.0, 0.0, 10.0]]) # adapt 2
    print("before adaptation", stress)

    foo()

    flag_end = 1
