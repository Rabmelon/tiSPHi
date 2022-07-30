import taichi as ti
import numpy as np

ti.init(debug=True, default_fp=ti.f64)

dim = 3
I3 = ti.Matrix(np.eye(dim))

@ti.func
def get_f_stress3(f_stress):
    res = ti.Matrix([[f_stress[0], f_stress[2], 0.0], [f_stress[2], f_stress[1], 0.0], [0.0, 0.0, f_stress[3]]])
    return res

@ti.kernel
def foo():
    stress = get_f_stress3(f_stress)
    print("σ =", stress)

if __name__ == "__main__":
    print("hallo test here!")

    f_stress = ti.Vector([-0.742128, -1.731632, -0.000000, -0.742128])

    # foo()

    a = ti.Matrix([[1, 2], [3, 4]])
    b = 0.5 * ti.Matrix([[a[i, j] - a[j, i] for j in range(2)] for i in range(2)])
    print(b)

    flag_end = 1
