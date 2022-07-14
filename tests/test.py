import taichi as ti
import numpy as np

ti.init()

@ti.kernel
def foo():
    pass

if __name__ == "__main__":
    print("hallo test kernel function accuracy!")

    a = ti.Matrix([[1, 2], [3, 4]])
    b = ti.Vector([1, 2])

    D = ti.Matrix([[1,2,0,2],[2,1,0,2],[0,0,3,0],[2,2,0,1]])

    c = a@b
    d = a@a

    print(D)
    print(c)
    print(d)

    flag_end = 1
