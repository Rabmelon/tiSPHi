import taichi as ti
import numpy as np
from eng.particle_system import *
from eng.sph_solver import *
from eng.gguishow import *

ti.init(arch=ti.cpu)

class ChkKernel(SPHSolver):
    def __init__(self, particle_system, kernel):
        super().__init__(particle_system, kernel)
        print("Class check kernel starts to serve!")
        self.fv = ti.field(dtype=float)
        self.d_fv = ti.Vector.field(self.ps.dim, dtype=float)
        self.g_fv = ti.Matrix.field(self.ps.dim, self.ps.dim, dtype=float)
        self.f_CSPM = ti.field(dtype=float)
        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.fv, self.d_fv, self.g_fv, self.f_CSPM)

    @ti.kernel
    def init_value(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] < 10:
                # self.ps.val[p_i] = 1.0
                # self.ps.val[p_i] = self.ps.x[p_i].sum()
                self.ps.val[p_i] = self.fv[p_i]
                # self.ps.val[p_i] = self.f_CSPM[p_i]
                # self.ps.val[p_i] = self.d_fv[p_i][0]
                # self.ps.val[p_i] = self.d_fv[p_i].norm()
                # self.ps.val[p_i] = self.g_fv[p_i][0,0]

    @ti.func
    def ff(self, x):
        # res = (x**2).sum()
        res = x.sum()
        return res

    @ti.kernel
    def cal_f_CSPM(self):
        for p_i in range(self.ps.particle_num[None]):
            x_i = self.ps.x[p_i]
            self.f_CSPM[p_i] = 0.0
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                self.f_CSPM[p_i] += self.ps.m_V * self.kernel(x_i - x_j)

    @ti.kernel
    def cal_f(self):
        for p_i in range(self.ps.particle_num[None]):
            x_i = self.ps.x[p_i]
            self.fv[p_i] = 0.0
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                self.fv[p_i] += self.ps.m_V * self.kernel(x_i - x_j) * self.ff(x_j)
                # self.fv[p_i] += self.kernel(x_i - x_j)
            self.fv[p_i] /= self.f_CSPM[p_i]


    @ti.kernel
    def cal_grad_f(self):
        for p_i in range(self.ps.particle_num[None]):
            x_i = self.ps.x[p_i]
            self.d_fv[p_i] = ti.Vector([0.0 for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                self.d_fv[p_i] += self.ps.m_V * (self.ff(x_j)-self.ff(x_i)) * (self.kernel_derivative(x_i - x_j))
                # self.d_fv[p_i] += self.ps.m_V * (self.ff(x_j)-self.ff(x_i)) * (self.ps.L[p_i] @ self.kernel_derivative(x_i - x_j))

    @ti.kernel
    def cal_f2(self):
        for p_i in range(self.ps.particle_num[None]):
            x_i = self.ps.x[p_i]
            self.g_fv[p_i] = ti.Matrix([[0.0 for _ in range(self.ps.dim)] for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                # tmp = self.kernel_derivative(x_i - x_j)
                tmp = self.ps.L[p_i] @ self.kernel_derivative(x_i - x_j)
                self.g_fv[p_i] += self.ps.m_V * (self.ff(x_j) - self.ff(x_i)) @ tmp.transpose()

    def step(self):
        self.ps.initialize_particle_system()
        self.cal_L()
        self.cal_f_CSPM()
        self.cal_f()
        self.cal_grad_f()
        # self.cal_f2()

if __name__ == "__main__":
    print("hallo test kernel function accuracy!")

    screen_to_world_ratio = 4   # exp: world = (150, 100), ratio = 4, screen res = (600, 400)
    rec_world = [120, 100]   # a rectangle world start from (0, 0) to this pos
    particle_radius = 2
    case1 = ParticleSystem(rec_world, particle_radius)
    case1.gen_boundary_dummy()
    case1.add_cube(lower_corner=[0, 0], cube_size=[80, 80], material=1)

    solver = ChkKernel(case1, 2)
    gguishow(case1, solver, rec_world, screen_to_world_ratio, step_ggui=20, pause_init=1, iparticle=-1, kradius=1.05, color_title="f=x+y, f, WLC2")
    # f=x+y, f, |f'|, f'[0,0], CS, WLC2, Gaus

