import taichi as ti
import numpy as np
from .sph_solver import SPHSolver

class MCmuISPHSolver(SPHSolver):
    def __init__(self, particle_system, TDmethod, density, cohesion, friction, eta_0):
        super().__init__(particle_system, TDmethod)
        print("Class M-C μ(I) Soil SPH Solver starts to serve!")

        # basic paras
        self.density_0 = density
        self.coh = cohesion
        self.fric_deg = friction
        self.eta_0 = eta_0
        self.fric = self.fric_deg / 180 * np.pi
        self.mass = self.ps.m_V * self.density_0

        # allocate memories
        self.u_grad = ti.Matrix.field((self.ps.dim, self.ps.dim), dtype=float)
        self.strain_dbdot = ti.field(dtype=float)
        self.tau = ti.Matrix.field((self.ps.dim, self.ps.dim), dtype=float)
        self.d_density = ti.field(dtype=float)
        self.d_u = ti.Vector.field(self.ps.dim, dtype=float)
        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.u_grad, self.strain_dbdot, self.tau, self.d_density, self.d_u)

    @ti.kernel
    def cal(self):
        for p_i in range(self.ps.particle_num[None]):
            pass

    @ti.kernel
    def cal_d_density(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            x_i = self.ps.x[p_i]
            self.d_density[p_i] = 0.0
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                self.d_density[p_i] += 1


