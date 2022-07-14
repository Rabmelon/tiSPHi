import taichi as ti
import numpy as np
from .sph_solver import SPHSolver

class DPSESPHSolver(SPHSolver):
    def __init__(self, particle_system, TDmethod, kernel, density, cohesion, friction, poison, EYoungMod):
        super().__init__(particle_system, TDmethod, kernel)
        print("Class Drucker-Prager Soil SPH Solver starts to serve!")

        # basic paras
        self.density_0 = density
        self.coh = cohesion
        self.fric_deg = friction
        self.poi = poison
        self.EYoungMod = EYoungMod
        self.fric = self.fric_deg / 180 * np.pi
        self.mass = self.ps.m_V * self.density_0

        # allocate memories
        self.u_grad = ti.Matrix.field(self.ps.dim, self.ps.dim, dtype=float)
        self.strain_dbdot = ti.field(dtype=float)
        self.d_density = ti.field(dtype=float)
        self.d_u = ti.Vector.field(self.ps.dim, dtype=float)
        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.u_grad, self.strain_dbdot, self.d_density, self.d_u)

    @ti.kernel
    def init_value(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] < 10:
                self.ps.val[p_i] = self.ps.u[p_i].norm()
                # self.ps.val[p_i] = self.ps.density[p_i]
                # self.ps.val[p_i] = self.d_density[p_i]
                # self.ps.val[p_i] = self.pressure[p_i]
                # self.ps.val[p_i] = self.ps.u[p_i][0]

    @ti.func
    def update_boundary_particles(self, p_i, p_j):
        self.ps.density[p_j] = self.density_0
        self.ps.u[p_j] = (1.0 - min(1.5, 1.0 + self.cal_d_BA(p_i, p_j))) * self.ps.u[p_i]
        self.pressure[p_j] = self.pressure[p_i]

    @ti.kernel
    def cal_d_density(self):
        for p_i in range(self.ps.particle_num[None]):
            dd = 0.0
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                if self.ps.material[p_j] == self.ps.material_dummy:
                    self.update_boundary_particles(p_i, p_j)
                tmp = (self.ps.u[p_i] - self.ps.u[p_j]).transpose() @ self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
                # tmp = (self.ps.u[p_i] - self.ps.u[p_j]).transpose() @ (self.ps.L[p_i] @ self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j]))
                dd += self.ps.density[p_j] * tmp[0]
            self.d_density[p_i] =  dd * self.mass / self.ps.density[p_i]

    @ti.kernel
    def cal_pressure(self):
        for p_i in range(self.ps.particle_num[None]):
            self.pressure[p_i] = ti.max(self.usound2 * (self.ps.density[p_i] - self.density_0), 0.0)

    @ti.kernel
    def cal_u_grad(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            u_g = ti.Matrix([[0.0 for _ in range(self.ps.dim)] for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                if self.ps.material[p_j] == self.ps.material_dummy:
                    self.update_boundary_particles(p_i, p_j)
                tmp = self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
                # tmp = self.ps.L[p_i] @ self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
                u_g += self.ps.m_V * (self.ps.u[p_j] - self.ps.u[p_i]) @ tmp.transpose()
            self.u_grad[p_i] = u_g

    @ti.kernel
    def cal_strain(self):
        for p_i in range(self.ps.particle_num[None]):
            for i, j in ti.static(ti.ndrange(self.ps.dim, self.ps.dim)):
                self.ps.strain[p_i][i, j] = 0.5 * (self.u_grad[p_i][i, j] + self.u_grad[p_i][j, i])
            self.strain_dbdot[p_i] = ti.sqrt(0.5 * (self.ps.strain[p_i] * self.ps.strain[p_i]).sum()) + self.epsilon

    @ti.kernel
    def cal_tau(self):
        for p_i in range(self.ps.particle_num[None]):
            self.tau[p_i] = (self.eta_0 + (self.coh + self.pressure[p_i] * self.mu) / self.strain_dbdot[p_i]) * self.ps.strain[p_i]

    @ti.kernel
    def cal_stress(self):
        for p_i in range(self.ps.particle_num[None]):
            self.ps.stress[p_i] = self.tau[p_i] - self.pressure[p_i] * self.I

    @ti.kernel
    def cal_d_velocity(self):
        for p_i in range(self.ps.particle_num[None]):
            du = ti.Vector([0.0 for _ in range(self.ps.dim)])
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                if self.ps.material[p_j] == self.ps.material_dummy:
                    self.update_boundary_particles(p_i, p_j)
                du += self.ps.density[p_j] * self.ps.m_V * (self.ps.stress[p_j] / self.ps.density[p_j]**2 + self.ps.stress[p_i] / self.ps.density[p_i]**2) @ self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
                # du += self.ps.density[p_j] * self.ps.m_V * (self.ps.stress[p_j] / self.ps.density[p_j]**2 + self.ps.stress[p_i] / self.ps.density[p_i]**2) @ (self.ps.L[p_i] @ self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j]))
            if self.ps.dim == 2:
                du += ti.Vector([0, self.g])
            else:
                print("!!!!!My Error: cannot used in 3D now!")
            self.d_u[p_i] = du

    @ti.kernel
    def cal_density(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_soil:
                self.ps.density[p_i] += self.d_density[p_i] * self.dt[None]

    @ti.kernel
    def chk_density(self):
        for p_i in range(self.ps.particle_num[None]):
            self.ps.density[p_i] = ti.max(self.density_0, self.ps.density[p_i])
            if self.ps.density[p_i] > self.density_0 * 1.25:
                print("stop because particle", p_i, "has a density", self.ps.density[p_i], "and pressure", self.pressure[p_i], "with neighbour num", self.ps.particle_neighbors_num[p_i])
            assert self.ps.density[p_i] < self.density_0 * 1.25

    @ti.kernel
    def advect_SE(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_soil:
                self.ps.u[p_i] += self.d_u[p_i] * self.dt[None]
                self.ps.x[p_i] += self.ps.u[p_i] * self.dt[None]

    def substep_SympEuler(self):
        self.cal_d_density()
        self.cal_pressure()
        self.cal_u_grad()
        self.cal_strain()
        self.cal_tau()
        self.cal_stress()
        self.cal_d_velocity()
        self.cal_density()
        self.chk_density()
        self.advect_SE()

