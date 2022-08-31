import taichi as ti
import numpy as np
from eng.sph_solver import SPHSolver

# ! 2D only

class DPLFSPHSolver(SPHSolver):
    def __init__(self, particle_system, kernel, density, cohesion, friction, EYoungMod=5.0e6, poison=0.3, dilatancy=0.0,
                 alpha_Pi=0.0, beta_Pi=0.0):
        super().__init__(particle_system, kernel)
        print("Class Drucker-Prager Soil SPH Solver starts to serve!")

        # basic paras
        self.density_0 = density
        self.coh = cohesion
        self.fric = friction / 180 * np.pi
        self.EYoungMod = EYoungMod
        self.poi = poison
        self.dila = dilatancy / 180 * np.pi
        self.alpha_Pi = alpha_Pi
        self.beta_Pi = beta_Pi

        self.mass = self.ps.m_V * self.density_0
        self.dim = 3
        self.dim_v = 4
        self.I3 = ti.Matrix(np.eye(self.dim))
        self.max_x1 = ti.field(float, shape=())
        self.vsound = ti.sqrt(self.EYoungMod / self.density_0)        # speed of sound, m/s
        dt_tmp = 0.2 * self.ps.smoothing_len / self.vsound
        self.dt[None] = ti.max(self.dt_min, dt_tmp - dt_tmp % self.dt_min)  # CFL

        # calculated paras
        self.alpha_fric = ti.tan(self.fric) / ti.sqrt(9 + 12 * ti.tan(self.fric)**2)
        self.k_c = 3 * self.coh / ti.sqrt(9 + 12 * ti.tan(self.fric)**2)
        self.GShearMod = self.EYoungMod / (2 * (1 + self.poi))
        self.KBulkMod = self.EYoungMod / (3 * (1 - 2 * self.poi))
        self.De = ti.Matrix([[1.0 - self.poi, self.poi, 0.0, self.poi],
                            [self.poi, 1.0 - self.poi, 0.0, self.poi],
                            [0.0, 0.0, (1.0 - 2.0 * self.poi) * 0.5, 0.0],
                            [self.poi, self.poi, 0.0, 1.0 - self.poi]]) * (self.EYoungMod / ((1.0 + self.poi) * (1.0 - 2.0 * self.poi)))

        # allocate memories
        self.density2 = ti.field(dtype=float)
        self.v2 = ti.Vector.field(self.ps.dim, dtype=float)

        self.v_grad = ti.Matrix.field(self.ps.dim, self.ps.dim, dtype=float)
        self.f_stress = ti.Vector.field(self.dim_v, dtype=float)
        self.f_v = ti.Matrix.field(self.dim_v, 2, dtype=float)
        self.stress = ti.Matrix.field(self.dim, self.dim, dtype=float)
        self.stress_s = ti.Matrix.field(self.dim, self.dim, dtype=float)
        self.I1 = ti.field(dtype=float)
        self.sJ2 = ti.field(dtype=float)
        self.fDP_old = ti.field(dtype=float)
        self.flag_adapt = ti.field(dtype=float)
        self.strain_p_equ = ti.field(dtype=float)

        self.d_density = ti.field(dtype=float)
        self.d_v = ti.Vector.field(self.ps.dim, dtype=float)
        self.d_f_stress = ti.Vector.field(self.dim_v, dtype=float)
        self.d_strain_p_equ = ti.field(dtype=float)

        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.density2, self.v2, self.v_grad, self.f_stress, self.f_v, self.stress, self.stress_s, self.I1, self.sJ2, self.fDP_old, self.flag_adapt, self.d_density, self.d_v, self.d_f_stress, self.strain_p_equ, self.d_strain_p_equ)

        self.assign_x0()
        self.cal_max_hight()
        self.init_stress()


    ###########################################################################
    # colored value
    ###########################################################################
    @ti.kernel
    def init_value(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] < 10:
                # self.ps.val[p_i] = p_i
                # self.ps.val[p_i] = self.ps.v[p_i].norm()
                self.ps.val[p_i] = self.ps.density[p_i]
                # self.ps.val[p_i] = self.d_density[p_i]
                # self.ps.val[p_i] = self.pressure[p_i]
                # self.ps.val[p_i] = self.ps.v[p_i][0]
                # self.ps.val[p_i] = -self.stress[p_i][1,1]
                # self.ps.val[p_i] = -(self.stress[p_i][0,0] + self.stress[p_i][1,1] + self.stress[p_i][2,2]) / 3
                # self.ps.val[p_i] = self.strain_p_equ[p_i]
                # self.ps.val[p_i] = ti.sqrt(((self.ps.x[p_i] - self.ps.x0[p_i])**2).sum())

    ###########################################################################
    # assisting funcs
    ###########################################################################
    @ti.func
    def update_boundary_particles(self, p_i, p_j):
        self.density2[p_j] = self.density_0
        self.v2[p_j] = (1.0 - min(1.5, 1.0 + self.calc_d_BA_rec(p_i, p_j))) * self.v2[p_i]

    @ti.func
    def cal_f_v(self, v):
        res = ti.Matrix([[self.De[0, 0] * v[0], self.De[0, 1] * v[1]],
                         [self.De[1, 0] * v[0], self.De[1, 1] * v[1]],
                         [self.De[2, 2] * v[1], self.De[2, 2] * v[0]],
                         [self.De[3, 0] * v[0], self.De[3, 1] * v[1]]])
        return res

    @ti.func
    def stress2_fs(self, stress):
        res = ti.Vector([stress[0,0], stress[1,1], stress[0,1], 0.0])
        return res

    @ti.func
    def stress3_fs(self, stress):
        res = ti.Vector([stress[0,0], stress[1,1], stress[0,1], stress[2,2]])
        return res

    @ti.func
    def stress_stress2(self, stress):
        res = ti.Matrix([[stress[0,0], stress[0,1]], [stress[1,0], stress[1,1]]])
        return res

    @ti.func
    def fs_stress3(self, f_stress):
        res = ti.Matrix([[f_stress[0], f_stress[2], 0.0],
                         [f_stress[2], f_stress[1], 0.0],
                         [0.0, 0.0, f_stress[3]]])
        return res

    @ti.func
    def cal_stress_s(self, stress):
        res = stress - stress.trace() / 3.0 * self.I3
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
    def cal_fDP(self, I1, sJ2):
        res = sJ2 + self.alpha_fric * I1 - self.k_c
        return res

    @ti.func
    def cal_from_stress(self, stress):
        stress_s = self.cal_stress_s(stress)
        vI1 = self.cal_I1(stress)
        sJ2 = self.cal_sJ2(stress_s)
        fDP = self.cal_fDP(vI1, sJ2)
        return stress_s, vI1, sJ2, fDP

    ###########################################################################
    # assisting kernels
    ###########################################################################
    @ti.kernel
    def init_LF_f(self):
        for p_i in range(self.ps.particle_num[None]):
            self.density2[p_i] = self.ps.density[p_i]
            self.v2[p_i] = self.ps.v[p_i]

    @ti.kernel
    def cal_max_hight(self):
        vmax = -float('Inf')
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] < 10:
                ti.atomic_max(vmax, self.ps.x[p_i][1])
        self.max_x1[None] = vmax

    @ti.kernel
    def init_stress(self):
        K0 = 1.0 - ti.sin(self.fric)
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            ver_stress = self.density_0*self.g*(self.max_x1[None] - self.ps.x[p_i][1])
            self.stress[p_i] = self.fs_stress3(ti.Vector([K0*ver_stress, ver_stress, 0.0, K0*ver_stress]))

    @ti.kernel
    def init_basic_terms(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            self.f_stress[p_i] = self.stress3_fs(self.stress[p_i])
            self.f_v[p_i] = self.cal_f_v(self.ps.v[p_i])
            self.stress_s[p_i] = self.cal_stress_s(self.stress[p_i])
            self.I1[p_i] = self.cal_I1(self.stress[p_i])
            self.sJ2[p_i] = self.cal_sJ2(self.stress_s[p_i])
            self.fDP_old[p_i] = self.cal_fDP(self.I1[p_i], self.sJ2[p_i])

    @ti.kernel
    def cal_v_grad(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            v_g = ti.Matrix([[0.0 for _ in range(self.ps.dim)] for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                if self.ps.material[p_j] == self.ps.material_dummy:
                    self.update_boundary_particles(p_i, p_j)
                if self.ps.material[p_j] > 10:
                    continue
                tmp = self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
                # tmp = self.CSPM_L[p_i] @ self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
                v_g += (self.v2[p_j] - self.v2[p_i]) @ tmp.transpose() / self.density2[p_j]
            self.v_grad[p_i] = v_g * self.mass

    ###########################################################################
    # Artificial terms
    ###########################################################################
    @ti.func
    def cal_artificial_viscosity(self, alpha_Pi, beta_Pi, p_i, p_j):
        res = 0.0
        vare = 0.01
        xij = self.ps.x[p_i] - self.ps.x[p_j]
        vij = self.v2[p_i] - self.v2[p_j]
        vijxij = (vij * xij).sum()
        if vijxij < 0.0:
            rhoij = 0.5 * (self.density2[p_i] + self.density2[p_j])
            hij = self.ps.smoothing_len
            cij = self.vsound
            phiij = hij * vijxij / ((xij.norm())**2 + vare * hij**2)
            res = (-alpha_Pi * cij * phiij + beta_Pi * phiij**2) / rhoij
        return res

    # these two regularisation ways does not make effort!
    @ti.kernel
    def regu_stress(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            tmp = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                stress_j = self.stress[p_j]
                xij = self.ps.x[p_i] - self.ps.x[p_j]
                if self.ps.material[p_j] == self.ps.material_dummy:
                    self.update_boundary_particles(p_i, p_j)
                    stress_j = self.stress[p_i]
                if self.ps.material[p_j] > 10:
                    continue
                Wij_MLS = self.kernel(xij) * (self.MLS_beta[p_i][0] + self.MLS_beta[p_i][1] * xij[0] + self.MLS_beta[p_i][2] * xij[1])
                tmp += self.mass / self.density2[p_j] * stress_j * Wij_MLS
            self.stress[p_i] = tmp

    @ti.func
    def regu_stress_i(self, p_i):
        tmp = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        for j in range(self.ps.particle_neighbors_num[p_i]):
            p_j = self.ps.particle_neighbors[p_i, j]
            stress_j = self.stress[p_j]
            xij = self.ps.x[p_i] - self.ps.x[p_j]
            if self.ps.material[p_j] == self.ps.material_dummy:
                self.update_boundary_particles(p_i, p_j)
                stress_j = self.stress[p_i]
                # self.calc_stress_roller(p_j)
            if self.ps.material[p_j] > 10:
                continue
            # if p_i == 2316:
                # print("stress", p_j, stress_j)
            tmp += self.mass / self.density2[p_j] * stress_j * self.kernel(xij)
            # tmp *= self.MLS_beta[p_i][0] + self.MLS_beta[p_i][1] * xij[0] + self.MLS_beta[p_i][2] * xij[1]
        tmp *= self.CSPM_f[p_i]
        self.stress[p_i] = tmp

    @ti.func
    def calc_stress_roller(self, p_j):
        tmp = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        for k in range(self.ps.particle_neighbors_num[p_j]):
            p_k = self.ps.particle_neighbors[p_j, k]
            if self.ps.material[p_k] != self.ps.material_soil:
                continue
            stress_k = self.stress[p_k]
            xjk = self.ps.x[p_j] - self.ps.x[p_k]
            tmp += self.mass / self.density2[p_k] * stress_k * self.kernel(xjk)
            # tmp *= self.MLS_beta[p_i][0] + self.MLS_beta[p_i][1] * xij[0] + self.MLS_beta[p_i][2] * xij[1]
        tmp *= self.CSPM_f[p_j]
        self.stress[p_j] = tmp * ti.Matrix([[1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])




    ###########################################################################
    # stress adaptation
    ###########################################################################
    @ti.func
    def adapt_stress(self, stress):
        # TODO: add a return of the new DP flag and adaptation flag
        # TODO: what is the usage of dfDP?
        res = stress
        stress_s, vI1, sJ2, fDP_new = self.cal_from_stress(res)
        if fDP_new > 1e-4:
            if fDP_new > sJ2:
                res = self.adapt_1(res, vI1)
            stress_s, vI1, sJ2, fDP_new = self.cal_from_stress(res)
            res = self.adapt_2(stress_s, vI1, sJ2)
        return res

    @ti.func
    def chk_flag_DP(self, fDP_new, sJ2):
        flag = 0
        if fDP_new > self.epsilon:
            if fDP_new >= sJ2:
                flag = 1
            else:
                flag = 2
        return flag

    @ti.func
    def adapt_1(self, stress, I1):
        tmp = (I1-self.k_c/self.alpha_fric) / 3.0
        res = stress - tmp * self.I3
        return res

    @ti.func
    def adapt_2(self, s, I1, sJ2):
        r = (-I1 * self.alpha_fric + self.k_c) / sJ2
        res = r * s + self.I3 * I1 / 3.0
        return res


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
                if self.ps.material[p_j] > 10:
                    continue
                # tmp = (self.v2[p_i] - self.v2[p_j]).transpose() @ self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
                tmp = (self.v2[p_i] - self.v2[p_j]).transpose() @ (self.CSPM_L[p_i] @ self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j]))
                dd += tmp[0] / self.density2[p_j]
            self.d_density[p_i] =  dd * self.mass * self.density2[p_i]

    @ti.kernel
    def cal_d_velocity(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            dv = ti.Vector([0.0 for _ in range(self.ps.dim)])
            rep = ti.Vector([0.0 for _ in range(self.ps.dim)])
            stress_i_2d = self.stress_stress2(self.stress[p_i])

            # viscous damping
            # xi = 5.0e-5
            # cd = xi * ti.sqrt(self.EYoungMod / (self.density_0 * self.ps.smoothing_len**2))
            # Fd = -cd * self.ps.v[p_i]
            Fd = 0.0

            # artificial viscosity
            tmp_av = 0.0
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                stress_j_2d = self.stress_stress2(self.stress[p_j])
                if self.ps.material[p_j] == self.ps.material_dummy:
                    self.update_boundary_particles(p_i, p_j)
                    stress_j_2d = self.stress_stress2(self.stress[p_i])
                if self.ps.material[p_j] == self.ps.material_repulsive:
                    rep += self.calc_repulsive_force(self.ps.x[p_i] - self.ps.x[p_j], self.vsound)
                    continue
                if self.ps.material[p_j] == self.ps.material_soil and self.alpha_Pi > 0.0:
                    tmp_av = self.cal_artificial_viscosity(self.alpha_Pi, self.beta_Pi, p_i, p_j)
                dv += self.density2[p_j] * self.ps.m_V * (stress_j_2d / self.density2[p_j]**2 + stress_i_2d / self.density2[p_i]**2 - tmp_av * self.I) @ self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
            if self.ps.dim == 2:
                dv += ti.Vector([0.0, self.g])
            else:
                print("!!!!!My Error: cannot used in 3D now!")
            self.d_v[p_i] = dv + rep + Fd

    @ti.kernel
    def cal_d_f_stress_Bui2008(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            strain_r = 0.5 * ti.Matrix([[self.v_grad[p_i][i, j] + self.v_grad[p_i][j, i] for j in range(self.ps.dim)] for i in range(self.ps.dim)])
            spin_r = 0.5 * ti.Matrix([[self.v_grad[p_i][i, j] - self.v_grad[p_i][j, i] for j in range(self.ps.dim)] for i in range(self.ps.dim)])
            tmp_J = ti.Matrix([[
                self.stress[p_i][i, 0] * spin_r[j, 0] + self.stress[p_i][i, 1] * spin_r[j, 1] +
                self.stress[p_i][0, j] * spin_r[i, 0] + self.stress[p_i][1, j] * spin_r[i, 1] for j in range(self.ps.dim)] for i in range(self.ps.dim)])
            lambda_r = 0.0
            tmp_g = ti.Matrix([[0.0 for _ in range(self.ps.dim)] for _ in range(self.ps.dim)])
            if self.fDP_old[p_i] >= -self.epsilon and self.sJ2[p_i] > self.epsilon:
                lambda_r = (
                    3.0 * self.alpha_fric * self.KBulkMod * strain_r.trace() +
                    (self.GShearMod / self.sJ2[p_i]) * (self.stress_stress2(self.stress_s[p_i]) * strain_r).sum()
                ) / (27.0 * self.alpha_fric * self.KBulkMod * ti.sin(self.dila) + self.GShearMod)
                tmp_g = lambda_r * (9.0 * self.KBulkMod * ti.sin(self.dila) * self.I + self.GShearMod / self.sJ2[p_i] * self.stress_stress2(self.stress_s[p_i]))

            strain_r_equ = strain_r - strain_r.trace() / 3.0 * self.I
            tmp_v = 2.0 * self.GShearMod * strain_r_equ + self.KBulkMod * strain_r.trace() * self.I
            self.d_f_stress[p_i] = self.stress2_fs(tmp_J + tmp_g + tmp_v)

            # calculate the equivalent plastic strain
            self.d_strain_p_equ[p_i] = ti.sqrt((strain_r_equ*strain_r_equ).sum() * 2 / 3)

    @ti.kernel
    def cal_d_f_stress_Chalk2020(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            omega_r_xy = (self.v_grad[p_i][0,1] - self.v_grad[p_i][1,0]) * 0.5
            tmp_J = ti.Vector([2.0 * self.stress[p_i][0, 1] * omega_r_xy, -2.0 * self.stress[p_i][0, 1] * omega_r_xy,
                               -self.stress[p_i][0, 0] * omega_r_xy + self.stress[p_i][1, 1] * omega_r_xy, 0.0])
            strain_r = 0.5 * ti.Matrix([[self.v_grad[p_i][i, j] + self.v_grad[p_i][j, i] for j in range(self.ps.dim)] for i in range(self.ps.dim)])
            tmp_g = ti.Vector([0.0 for _ in range(self.dim_v)])
            if self.fDP_old[p_i] >= -self.epsilon and self.sJ2[p_i] > self.epsilon:
                lambda_r = (3.0 * self.alpha_fric * strain_r.trace() + (self.GShearMod / self.sJ2[p_i]) * (self.stress_stress2(self.stress_s[p_i]) * strain_r).sum()) / (27.0 * self.alpha_fric * self.KBulkMod * ti.sin(self.dila) + self.GShearMod)
                tmp_g_dim = lambda_r * (9 * self.KBulkMod * ti.sin(self.dila) * self.I3 + self.GShearMod / self.sJ2[p_i] * self.stress_s[p_i])
                tmp_g = ti.Vector([tmp_g_dim[0,0], tmp_g_dim[1,1], tmp_g_dim[0,1], tmp_g_dim[2,2]])
            tmp_v = ti.Vector([0.0 for _ in range(self.dim_v)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                if self.ps.material[p_j] == self.ps.material_dummy:
                    self.update_boundary_particles(p_i, p_j)
                    self.f_v[p_j] = self.cal_f_v(self.v2[p_j])
                tmp_v += (self.f_v[p_j] - self.f_v[p_i]) @ self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j]) / self.density2[p_j]
            self.d_f_stress[p_i] += tmp_J + tmp_g + tmp_v * self.mass


    ###########################################################################
    # advection
    ###########################################################################
    @ti.kernel
    def chk_density(self):
        for p_i in range(self.ps.particle_num[None]):
            self.ps.density[p_i] = ti.max(self.density_0, self.ps.density[p_i])
            if self.ps.density[p_i] > self.density_0 * self.alertratio:
                print("stop because particle", p_i, "has a large density", self.ps.density[p_i], "with neighbour num", self.ps.particle_neighbors_num[p_i])
            assert self.ps.density[p_i] < self.density_0 * self.alertratio

    @ti.kernel
    def advect_LF_half(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_soil:
                # self.density2[p_i] = self.density_0
                self.density2[p_i] += self.d_density[p_i] * self.dt[None] * 0.5
                self.v2[p_i] += self.d_v[p_i] * self.dt[None] * 0.5
                self.strain_p_equ[p_i] += self.d_strain_p_equ[p_i] * self.dt[None] * 0.5
                self.f_stress[p_i] += self.d_f_stress[p_i] * self.dt[None] * 0.5
                self.stress[p_i] = self.fs_stress3(self.f_stress[p_i])
                # self.regu_stress_i(p_i)
                self.stress[p_i] = self.adapt_stress(self.stress[p_i])

    @ti.kernel
    def advect_LF(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_soil:
                # self.ps.density[p_i] = self.density_0
                self.ps.density[p_i] += self.d_density[p_i] * self.dt[None]
                self.ps.v[p_i] += self.d_v[p_i] * self.dt[None]
                self.ps.x[p_i] += self.ps.v[p_i] * self.dt[None]
                self.strain_p_equ[p_i] += self.d_strain_p_equ[p_i] * self.dt[None]
                self.f_stress[p_i] += self.d_f_stress[p_i] * self.dt[None]
                self.stress[p_i] = self.fs_stress3(self.f_stress[p_i])
                # self.regu_stress_i(p_i)
                self.stress[p_i] = self.adapt_stress(self.stress[p_i])

    def LF_one_step(self):
        self.init_basic_terms()
        self.cal_v_grad()
        self.cal_d_density()
        self.cal_d_f_stress_Bui2008()
        self.cal_d_velocity()

    def substep(self):
        self.init_LF_f()
        self.LF_one_step()
        self.advect_LF_half()
        self.LF_one_step()
        self.advect_LF()
        self.chk_density()
