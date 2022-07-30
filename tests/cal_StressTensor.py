import taichi as ti
import numpy as np

ti.init(arch=ti.cpu, debug=True, default_fp=ti.f64)

epsilon = 1e-16
dim = 3
I3 = ti.Matrix(np.eye(dim))

@ti.func
def get_f_stress(stress):
    res = ti.Vector([stress[0,0], stress[1,1], stress[0,1], stress[2,2]])
    return res

@ti.func
def get_stress2(stress):
    res = ti.Matrix([[stress[0,0], stress[0,1]], [stress[1,0], stress[1,1]]])
    return res

@ti.func
def get_f_stress3(f_stress):
    res = ti.Matrix([[f_stress[0], f_stress[2], 0.0],
                     [f_stress[2], f_stress[1], 0.0], [0.0, 0.0, f_stress[3]]])
    return res

@ti.func
def cal_stress_s(stress):
    res = stress - stress.trace() / 3.0 * I3
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
def cal_fDP_from_stress(stress):
    stress_s = cal_stress_s(stress)
    vI1 = cal_I1(stress)
    sJ2 = cal_sJ2(stress_s)
    res = sJ2 + alpha_fric * vI1 - k_c
    return res

@ti.func
def adapt_stress(stress):
    # TODO: add a return of the new DP flag and adaptation flag
    # TODO: what is the usage of dfDP?
    res = stress
    stress_s = cal_stress_s(stress)
    vI1 = cal_I1(stress)
    sJ2 = cal_sJ2(stress_s)
    fDP_new = cal_fDP(vI1, sJ2)

    count = 0

    while fDP_new > epsilon:
        if fDP_new > sJ2:
            res = adapt_1(res, vI1)
            print("---- adapt 1 step:", get_f_stress(res))
        else:
            res = adapt_2(stress_s, vI1, sJ2)
            print("---- adapt 2 step:", get_f_stress(res))
        stress_s = cal_stress_s(res)
        vI1 = cal_I1(res)
        sJ2 = cal_sJ2(stress_s)
        fDP_new = cal_fDP(vI1, sJ2)
        print("---- adapt fDP =", fDP_new)

        count = count + 1
        if count >= 10:
            print("---- endless loop of adaptation!", get_f_stress(stress))
            break
        assert count < 10, "---- endless loop of adaptation!"

    return res

@ti.func
def adapt_1(stress, I1):
    tmp = (I1-k_c/alpha_fric) / 3.0
    res = stress - tmp * I3
    return res

@ti.func
def adapt_2(s, I1, sJ2):
    r = (-I1 * alpha_fric + k_c) / sJ2
    res = r * s + I3 * I1 / 3.0
    return res


@ti.kernel
def foo():
    stress = get_f_stress3(f_stress)
    stress_s = cal_stress_s(stress)
    vI1 = cal_I1(stress)
    sJ2 = cal_sJ2(stress_s)
    fDP_old = cal_fDP(vI1, sJ2)
    print("σ =", stress)
    print("s =", stress_s)
    print("I1 =", vI1)
    print("sJ2 =", sJ2)
    print("fDP old =", fDP_old)

@ti.kernel
def adapt():
    stress = get_f_stress3(f_stress)
    stress_new = adapt_stress(stress)
    fDP_new = cal_fDP_from_stress(stress_new)
    print("σ new =", stress_new)
    print("fDP new =", fDP_new)

if __name__ == "__main__":
    print("hallo test here!")
    fric = 21.9 / 180 * np.pi
    coh = 0.0
    alpha_fric = ti.tan(fric) / ti.sqrt(9 + 12 * ti.tan(fric)**2)
    k_c = 3 * coh / ti.sqrt(9 + 12 * ti.tan(fric)**2)

    # f_stress = ti.Vector([-1392.923706, -2863.590332, 0.102193, -1393.044434])
    # f_stress = ti.Vector([-5734.567383, -2583.398926, 126.379333, -2623.165283])
    f_stress = ti.Vector([-1.499807054856, -2.312474820000, -0.000000021633, -0.993532778422])

    foo()
    adapt()

    flag_end = 1
