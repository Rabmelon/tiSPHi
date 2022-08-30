import taichi as ti
import numpy as np

ti.init(debug=True, default_fp=ti.f64)

dim = 3
I3 = ti.Matrix(np.eye(dim))

# repulsive forces
@ti.func
def cal_repulsive_force(r, vsound, dx, h):
    r_norm = r.norm()
    chi = 1.0 - r_norm / (1.5 * dx) if (r_norm >= 0.0 and r_norm < 1.5 * dx) else 0.0
    gamma = r_norm / (0.75 * h)
    f = 0.0
    if gamma > 0 and gamma <= 2 / 3:
        f = 2 / 3
    elif gamma > 2 / 3 and gamma <= 1:
        f = 2 * gamma - 1.5 * gamma**2
    elif gamma > 1 and gamma < 2:
        f = 0.5 * (2 - gamma)**2
    res = 0.01 * vsound**2 * chi * f / (r_norm**2) * r
    return res

@ti.func
def get_f_stress3(f_stress):
    res = ti.Matrix([[f_stress[0], f_stress[2], 0.0], [f_stress[2], f_stress[1], 0.0], [0.0, 0.0, f_stress[3]]])
    return res

@ti.kernel
def foo():
    # stress = get_f_stress3(f_stress)
    # print("σ =", stress)

    F00 = cal_repulsive_force(xi0-xj0, vsound, dx, h)
    print("F00 =", F00)
    F10 = cal_repulsive_force(xi1-xj0, vsound, dx, h)
    print("F10 =", F10)
    F20 = cal_repulsive_force(xi2-xj0, vsound, dx, h)
    print("F20 =", F20)
    F01 = cal_repulsive_force(xi0-xj1, vsound, dx, h)
    print("F01 =", F01)
    F11 = cal_repulsive_force(xi1-xj1, vsound, dx, h)
    print("F11 =", F11)
    F21 = cal_repulsive_force(xi2-xj1, vsound, dx, h)
    print("F21 =", F21)

if __name__ == "__main__":
    print("hallo test here!")

    f_stress = ti.Vector([-0.742128, -1.731632, -0.000000, -0.742128])

    vsound = 53.50462688440333
    dx = 0.002
    h = 0.0024
    xj0 = ti.Vector([0.001, 0.0])
    xj1 = ti.Vector([0.002, 0.0])
    xi0 = ti.Vector([0.001, 0.001])
    xi1 = ti.Vector([0.001, 0.000])
    xi2 = ti.Vector([0.001, -0.00001])

    foo()


    flag_end = 1
