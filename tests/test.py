import taichi as ti
import numpy as np

ti.init(debug=True, default_fp=ti.f64)

@ti.kernel
def foo():
    A = An.inverse()
    print(A)

def Ap(pi):
    Ai = pi @ pi.transpose()
    print("Ai,", Ai)
    return Ai

if __name__ == "__main__":
    print("hallo test here!")

    p1 = ti.Vector([1.0, 8.0, 0.0])
    p2 = ti.Vector([1.0, 4.0, 4.0])
    p3 = ti.Vector([1.0, 0.0, 4.0])
    p4 = ti.Vector([1.0, 8.0, 8.0])

    A1 = Ap(p1)
    A2 = Ap(p2)
    A3 = Ap(p3)
    A4 = Ap(p4)
    An = A1 + A2 + A3 + A4

    print("An,", An)
    foo()


    flag_end = 1
