import taichi as ti
import numpy as np
from eng.particle_system import *
from eng.sph_solver import *
from eng.gguishow import *

ti.init()

class TEST(SPHSolver):
    def __init__(self, particle_system, TDmethod, kernel):
        super().__init__(particle_system, 1, kernel)
        print("Hallo, class WCSPH Solver with SE starts to serve!")

    @ti.kernel
    def init_value(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] < 10:
                # self.ps.val[p_i] = self.ps.u[p_i].norm()
                self.ps.val[p_i] = self.pressure[p_i]

    @ti.kernel
    def foo(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            x_i = self.ps.x[p_i]
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]

if __name__ == "__main__":
    print("hallo test kernel function accuracy!")

    screen_to_world_ratio = 4   # exp: world = (150, 100), ratio = 4, screen res = (600, 400)
    rec_world = [135, 105]   # a rectangle world start from (0, 0) to this pos
    particle_radius = 2
    case1 = ParticleSystem(rec_world, particle_radius)
    case1.add_cube(lower_corner=[0, 0], cube_size=[80, 80], material=1)

    tmp = TEST()

    tmp.foo()

    flag_end = 1
