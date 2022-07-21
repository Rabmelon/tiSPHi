import taichi as ti
import numpy as np
from .sph_solver import SPHSolver

# ! 2D only

class DPSESPHSolver(SPHSolver):
    def __init__(self, particle_system, TDmethod, kernel, density, cohesion, friction, EYoungMod=5.0e6, poison=0.3, dilatancy=0.0):
        super().__init__(particle_system, TDmethod, kernel)
        print("Class Drucker-Prager Soil SPH Solver starts to serve!")

        # basic paras
        self.density_0 = density
        self.coh = cohesion
        self.fric = friction / 180 * np.pi
        self.EYoungMod = EYoungMod
        self.poi = poison
        self.dila = dilatancy / 180 * np.pi
        self.mass = self.ps.m_V * self.density_0

        # calculated paras
        self.alpha_fric = ti.tan(self.fric) / ti.sqrt(9 + 12 * ti.tan(self.fric)**2)
        self.k_c = 3 * self.coh / ti.sqrt(9 + 12 * ti.tan(self.fric)**2)
        self.GShearMod = self.EYoungMod / (2 * (1 + self.poi))
        self.KBulkMod = self.EYoungMod / (3 * (1 - 2 * self.poi))
        self.De = ti.Matrix([[1.0 - self.poi, self.poi, 0.0, self.poi],
                            [self.poi, 1.0 - self.poi, 0.0, self.poi],
                            [0.0, 0.0, (1.0 - 2.0 * self.poi) * 0.5, 0.0],
                            [self.poi, self.poi, 0.0, 1.0 - self.poi]
                            ]) * (self.EYoungMod / ((1.0 + self.poi) * (1.0 - 2.0 * self.poi)))

        # allocate memories
        self.v_grad = ti.Matrix.field(self.ps.dim, self.ps.dim, dtype=float)
        self.f_stress = ti.Vector.field(self.ps.dim_v, dtype=float)
        self.f_v = ti.Matrix.field(self.ps.dim_v, self.ps.dim, dtype=float)
        self.stress_s = ti.Matrix.field(self.ps.dim, self.ps.dim, dtype=float)
        self.I1 = ti.field(dtype=float)
        self.sJ2 = ti.field(dtype=float)
        self.fDP_old = ti.field(dtype=float)
        self.flag_adapt = ti.field(dtype=float)

        self.d_density = ti.field(dtype=float)
        self.d_v = ti.Vector.field(self.ps.dim, dtype=float)
        self.d_f_stress = ti.Matrix.field(self.ps.dim, self.ps.dim, dtype=float)

        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.v_grad, self.f_stress, self.f_v, self.stress_s, self.I1, self.sJ2, self.fDP_old, self.flag_adapt, self.d_density, self.d_v, self.d_f_stress)


    ###########################################################################
    # colored value
    ###########################################################################
    @ti.kernel
    def init_value(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] < 10:
                self.ps.val[p_i] = self.ps.u[p_i].norm()
                # self.ps.val[p_i] = self.ps.density[p_i]
                # self.ps.val[p_i] = self.d_density[p_i]
                # self.ps.val[p_i] = self.pressure[p_i]
                # self.ps.val[p_i] = self.ps.u[p_i][0]

    ###########################################################################
    # assisting funcs
    ###########################################################################
    @ti.func
    def update_boundary_particles(self, p_i, p_j):
        self.ps.density[p_j] = self.density_0
        self.ps.u[p_j] = (1.0 - min(1.5, 1.0 + self.cal_d_BA(p_i, p_j))) * self.ps.u[p_i]

    @ti.func
    def cal_stress_s(self, stress):
        res = stress - stress.trace() / 3.0 * self.I
        return res

    @ti.func
    def cal_I1(self, stress):
        res = stress.trace()
        return res

    @ti.func
    def cal_sJ2(self, s):
        res = ti.sqrt(0.5 * (s*s).sum())
        return res

    @ti.func
    def cal_fDP(self, I1, sJ2, a_f, k_c):
        res = sJ2 + a_f * I1 - k_c
        return res

    @ti.func
    def chk_flag_DP(self, fDP_new):
        flag = 0

    ###########################################################################
    # assisting kernels
    ###########################################################################
    @ti.kernel
    def init_basic_terms(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            self.f_stress[p_i] = ti.Vector([self.ps.stress[0,0], self.ps.stress[1,1], self.ps.stress[0,1], 0.0])
            self.f_v[p_i] = ti.Matrix(
                [[self.De[0, 0] * self.ps.v[0], self.De[0, 1] * self.ps.v[1]],
                 [self.De[1, 0] * self.ps.v[0], self.De[1, 1] * self.ps.v[1]],
                 [self.De[2, 2] * self.ps.v[1], self.De[2, 2] * self.ps.v[0]],
                 [self.De[3, 0] * self.ps.v[0], self.De[3, 1] * self.ps.v[1]]])
            self.stress_s[p_i] = self.cal_stress_s(self.ps.stress[p_i])
            self.I1[p_i] = self.cal_I1(self.ps.stress[p_i])
            self.sJ2[p_i] = self.cal_sJ2(self.stress_s[p_i])
            self.fDP_old[p_i] = self.cal_fDP(self.I1[p_i], self.sJ2[p_i], self.alpha_fric, self.k_c)

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
                u_g += (self.ps.u[p_j] - self.ps.u[p_i]) @ tmp.transpose() / self.ps.density[p_j]
            self.u_grad[p_i] = u_g * self.mass

    ###########################################################################
    # stress adaptation
    ###########################################################################
    @ti.kernel
    def adapt_stress(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue


    ###########################################################################
    # approximation
    ###########################################################################
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
                tmp = (self.ps.u[p_i] - self.ps.u[p_j]).transpose() @ self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
                # tmp = (self.ps.u[p_i] - self.ps.u[p_j]).transpose() @ (self.ps.L[p_i] @ self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j]))
                dd += tmp[0] / self.ps.density[p_j]
            self.d_density[p_i] =  dd * self.mass * self.ps.density[p_i]

    @ti.kernel
    def cal_d_velocity(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            du = ti.Vector([0.0 for _ in range(self.ps.dim)])
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
    def cal_d_f_stress(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            omega_r_xy = (self.v_grad[p_i][0, 1] - self.v_grad[p_i][1, 0]) * 0.5
            tmp_J = ti.Vector([2.0 * self.ps.stress[p_i][0, 1] * omega_r_xy, -2.0 * self.ps.stress[p_i][0, 1] * omega_r_xy,
                               -self.ps.stress[p_i][0, 0] * omega_r_xy + self.ps.stress[p_i][1, 1] * omega_r_xy, 0.0])
            strain_r = 0.5 * ti.Matrix([[self.v_grad[p_i][i, j] + self.v_grad[p_i][j, i] for i in range(self.ps.dim)] for j in range(self.ps.dim)])
            lambda_r = (3.0 * self.alpha_fric * strain_r.trace() + (self.GShearMod / self.sJ2[p_i]) * (self.stress_s[p_i]*strain_r).sum()) / (27.0 * self.alpha_fric * self.KBulkMod * ti.sin(self.dila) + self.GShearMod)
            tmp_g_dim = lambda_r * (9 * self.KBulkMod * ti.sin(self.dila) * self.I + self.GShearMod / self.sJ2[p_i] / self.stress_s[p_i])
            tmp_g = ti.Vector([tmp_g_dim[0,0], tmp_g_dim[1,1], tmp_g_dim[0,1], 0.0])
            tmp_v = ti.Vector([0.0 for _ in range(self.ps.dim_v)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                if self.ps.material[p_j] == self.ps.material_dummy:
                    self.update_boundary_particles(p_i, p_j)
                tmp_v += (self.f_v[p_j] - self.f_v[p_i]) @ self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j]) / self.ps.density[p_j]
            self.d_f_stress[p_i] += tmp_J + tmp_g + tmp_v * self.mass

    ###########################################################################
    # advection
    ###########################################################################
    @ti.kernel
    def cal_density(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_soil:
                self.ps.density[p_i] += self.d_density[p_i] * self.dt[None]

    @ti.kernel
    def chk_density(self):
        for p_i in range(self.ps.particle_num[None]):
            self.ps.density[p_i] = ti.max(self.density_0, self.ps.density[p_i])
            if self.ps.density[p_i] > self.density_0 * self.alertratio:
                print("stop because particle", p_i, "has a large density", self.ps.density[p_i], "and pressure", self.pressure[p_i], "with neighbour num", self.ps.particle_neighbors_num[p_i])
            assert self.ps.density[p_i] < self.density_0 * self.alertratio

    @ti.kernel
    def advect_SE(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_soil:
                self.ps.u[p_i] += self.d_u[p_i] * self.dt[None]
                self.ps.x[p_i] += self.ps.u[p_i] * self.dt[None]

    def substep_SympEuler(self):
        self.init_basic_terms()
        self.cal_u_grad()
        self.cal_d_density()
        self.cal_d_f_stress()
        self.adapt_stress()
        self.cal_d_velocity()
        self.cal_density()
        self.chk_density()
        self.advect_SE()
