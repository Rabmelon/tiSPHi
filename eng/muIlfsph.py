import taichi as ti
import numpy as np
from .sph_solver import SPHSolver

# TODO: need to add damping?
# TODO: need to init the stress state when starting the simulation?

class MCmuILFSPHSolver(SPHSolver):
    def __init__(self, particle_system, TDmethod, kernel, density, cohesion, friction, eta_0):
        super().__init__(particle_system, TDmethod, kernel)
        print("Class M-C μ(I) Soil SPH Solver starts to serve!")

        # basic paras
        self.density_0 = density
        self.coh = cohesion
        self.fric_deg = friction
        self.eta_0 = eta_0
        self.fric = self.fric_deg / 180 * np.pi
        self.mass = self.ps.m_V * self.density_0
        self.mu = ti.tan(self.fric)

        # artificial viscosity and density diffusion terms
        self.delta = 0.1
        self.alphapi = 0.04
        self.Psi = ti.Vector.field(self.ps.dim, dtype=float)

        # allocate memories
        self.density2 = ti.field(dtype=float)
        self.u2 = ti.Vector.field(self.ps.dim, dtype=float)
        self.u_grad = ti.Matrix.field(self.ps.dim, self.ps.dim, dtype=float)
        self.strain_dbdot = ti.field(dtype=float)
        self.tau = ti.Matrix.field(self.ps.dim, self.ps.dim, dtype=float)
        self.d_density = ti.field(dtype=float)
        self.pressure = ti.field(dtype=float)
        self.d_u = ti.Vector.field(self.ps.dim, dtype=float)
        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.density2, self.u2, self.u_grad, self.strain_dbdot, self.tau, self.d_density, self.pressure, self.d_u, self.Psi)

    @ti.kernel
    def init_value(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] < 10:
                self.ps.val[p_i] = self.ps.u[p_i].norm()
                # self.ps.val[p_i] = self.ps.density[p_i]
                # self.ps.val[p_i] = self.d_density[p_i]
                # self.ps.val[p_i] = self.pressure[p_i]
                # self.ps.val[p_i] = self.ps.u[p_i][0]

    @ti.kernel
    def init_LF_f(self):
        for p_i in range(self.ps.particle_num[None]):
            self.density2[p_i] = self.ps.density[p_i]
            self.u2[p_i] = self.ps.u[p_i]

    @ti.func
    def update_boundary_particles(self, p_i, p_j):
        self.density2[p_j] = self.density_0
        self.u2[p_j] = (1.0 - min(1.5, 1.0 + self.cal_d_BA(p_i, p_j))) * self.u2[p_i]
        self.pressure[p_j] = self.pressure[p_i]

    @ti.kernel
    def cal_Psi(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            Psi = ti.Vector([0.0 for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                if self.ps.material[p_j] == self.ps.material_dummy:
                    self.update_boundary_particles(p_i, p_j)



    @ti.kernel
    def cal_d_density(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            dd = 0.0
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                if self.ps.material[p_j] == self.ps.material_dummy:
                    self.update_boundary_particles(p_i, p_j)
                tmp = (self.u2[p_i] - self.u2[p_j]).transpose() @ self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
                # tmp = (self.u2[p_i] - self.u2[p_j]).transpose() @ (self.ps.L[p_i] @ self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j]))
                dd += self.mass / self.density2[p_j] * tmp[0]
            self.d_density[p_i] = self.density2[p_i] * dd

    @ti.kernel
    def cal_pressure(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            self.pressure[p_i] = ti.max(self.usound2 * (self.density2[p_i] - self.density_0), 0.0)

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
                u_g += self.ps.m_V * (self.u2[p_j] - self.u2[p_i]) @ tmp.transpose()
                self.u_grad[p_i] = u_g

    @ti.kernel
    def cal_strain(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            for i, j in ti.static(ti.ndrange(self.ps.dim, self.ps.dim)):
                self.ps.strain[p_i][i, j] = 0.5 * (self.u_grad[p_i][i, j] + self.u_grad[p_i][j, i])
            self.strain_dbdot[p_i] = ti.sqrt(0.5 * (self.ps.strain[p_i] * self.ps.strain[p_i]).sum()) + self.epsilon

    @ti.kernel
    def cal_tau(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            self.tau[p_i] = (self.eta_0 + (self.coh + self.pressure[p_i] * self.mu) / self.strain_dbdot[p_i]) * self.ps.strain[p_i]

    @ti.kernel
    def cal_stress(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
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
                du += self.density2[p_j] * self.ps.m_V * (self.ps.stress[p_j] / self.density2[p_j]**2 + self.ps.stress[p_i] / self.density2[p_i]**2) @ self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
            if self.ps.dim == 2:
                du += ti.Vector([0, self.g])
            else:
                print("!!!!!My Error: cannot used in 3D now!")
            self.d_u[p_i] = du

    @ti.kernel
    def chk_density(self):
        for p_i in range(self.ps.particle_num[None]):
            self.ps.density[p_i] = ti.max(self.density_0, self.ps.density[p_i])
            if self.ps.density[p_i] > self.density_0 * 1.25:
                print("stop because particle", p_i, "has a density", self.ps.density[p_i], "and pressure", self.pressure[p_i], "with neighbour num", self.ps.particle_neighbors_num[p_i])
            assert self.ps.density[p_i] < self.density_0 * 1.25


    @ti.kernel
    def advect_LF_half(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_soil:
                self.density2[p_i] += self.d_density[p_i] * self.dt[None] * 0.5
                self.u2[p_i] += self.d_u[p_i] * self.dt[None] * 0.5

    @ti.kernel
    def advect_LF(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_soil:
                self.ps.density[p_i] += self.d_density[p_i] * self.dt[None]
                self.ps.u[p_i] += self.d_u[p_i] * self.dt[None]
                self.ps.x[p_i] += self.ps.u[p_i] * self.dt[None]

    def LF_one_step(self):
        self.cal_d_density()
        self.cal_pressure()
        self.cal_u_grad()
        self.cal_strain()
        self.cal_tau()
        self.cal_stress()
        self.cal_d_velocity()

    def substep_LeapFrog(self):
        self.init_LF_f()
        self.LF_one_step()
        self.advect_LF_half()
        self.LF_one_step()
        self.advect_LF()
        self.chk_density()

