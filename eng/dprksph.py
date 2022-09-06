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
        self.Frho = ti.Vector.field(self.ps.dim, dtype=float)
        self.Fv = ti.Vector.field(self.ps.dim, dtype=float)
        self.Fstress = ti.Vector.field(self.ps.dim, dtype=float)

        self.density = ti.field(dtype=float)
        self.v = ti.Vector.field(self.ps.dim, dtype=float)

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
        particle_node.place(self.density, self.v, self.v_grad, self.f_stress, self.f_v, self.stress, self.stress_s, self.I1, self.sJ2, self.fDP_old, self.flag_adapt, self.d_density, self.d_v, self.d_f_stress, self.strain_p_equ, self.d_strain_p_equ)
        particle_node.dense(ti.j, 4).place(self.Frho, self.Fv, self.Fstress)

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
                # self.ps.val[p_i] = self.ps.density[p_i]
                # self.ps.val[p_i] = self.d_density[p_i]
                # self.ps.val[p_i] = self.pressure[p_i]
                # self.ps.val[p_i] = self.ps.v[p_i][0]
                self.ps.val[p_i] = -self.stress[p_i][1,1]
                # self.ps.val[p_i] = -(self.stress[p_i][0,0] + self.stress[p_i][1,1] + self.stress[p_i][2,2]) / 3
                # self.ps.val[p_i] = self.strain_p_equ[p_i]
                # self.ps.val[p_i] = ti.sqrt(((self.ps.x[p_i] - self.ps.x0[p_i])**2).sum())

    ###########################################################################
    # assisting funcs
    ###########################################################################
    @ti.func
    def update_boundary_particles(self, p_i, p_j):
        self.density[p_j] = self.density_0
        self.v[p_j] = (1.0 - min(1.5, 1.0 + self.calc_d_BA_rec(p_i, p_j))) * self.v[p_i]

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
    def init_F(self):
        for p_i in range(self.ps.particle_num[None]):
            for m in range(4):
                self.Fv[p_i, m] = ti.Vector([0.0 for _ in range(self.ps.dim)])
                self.Frho[p_i, m] = ti.Vector([0.0 for _ in range(self.ps.dim)])
                self.Fstress[p_i, m] = ti.Vector([0.0 for _ in range(self.ps.dim)])

    @ti.kernel
    def upd_val_1(self, m: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            if m > 0:
                print('m1 =', m, end='; ')
                print('p_i =', p_i)
                continue
            self.v[p_i] = self.ps.v[p_i]
            self.density[p_i] = self.ps.density[p_i]

    @ti.kernel
    def upd_val_234(self, m: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            if m == 0 or m > 3:
                print('m2 =', m, end='; ')
                print('p_i =', p_i)
                continue
            self.v[p_i] = self.ps.v[p_i] + 0.5 * self.dt[None] * self.F[p_i, m-1]
            self.density[p_i] = self.ps.density[p_i] + 0.5 * self.dt[None] * self.F[p_i, m-1]

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
