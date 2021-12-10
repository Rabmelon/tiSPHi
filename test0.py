import taichi as ti
from eng.sph_solver import *

# TODO: Test wcsph code in taichi course and try to understand its parts.

ti.init(arch=ti.cuda)

@ti.kernel
def sayHallo():
    print("Hallo, tiSPHi!")

sayHallo()

case1 = SPHSolver()
