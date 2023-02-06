import taichi as ti
from eng.solver_sph_base import SPHBase
from eng.type_define import *


class DPSPHSolver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        print("Drucker-Prager SPH starts to serve!")

        # ! now only for one kind of soil
        self.density0 = self.ps.mat_soil[0]["density0"]
        self.coh = self.ps.mat_soil[0]["cohesion"]
        self.fric = self.ps.mat_soil[0]["friction"] / 180 * ti.math.pi
        self.E = self.ps.mat_soil[0]["EYoungMod"]
        self.poi = self.ps.mat_soil[0]["poison"]
        self.dila = self.ps.mat_soil[0]["dilatancy"] / 180 * ti.math.pi

        # self.vsound2 = self.E * (1 - self.poi) / ((1 + self.poi) * (1 - 2 * self.poi) * self.density0)  # @yang2020
        self.vsound2 = self.E / self.density0   # @bui2013
        self.vsound = ti.sqrt(self.vsound2)
        self.dt[None] = self.calc_dt_CFL(CFL_component=0.2, vsound=self.vsound, dt_min=self.dt_min)
        self.eps_f = 1e-4

        # calculated paras
        self.alpha_fric = ti.tan(self.fric) / ti.sqrt(9 + 12 * ti.tan(self.fric)**2)
        self.k_c = 3 * self.coh / ti.sqrt(9 + 12 * ti.tan(self.fric)**2)
        self.G = self.E / (2 * (1 + self.poi))
        self.K = self.E / (3 * (1 - 2 * self.poi))
        self.De = ti.Matrix([[1.0 - self.poi, self.poi, 0.0, self.poi],
                            [self.poi, 1.0 - self.poi, 0.0, self.poi],
                            [0.0, 0.0, (1.0 - 2.0 * self.poi) * 0.5, 0.0],
                            [self.poi, self.poi, 0.0, 1.0 - self.poi]]) * (self.E / ((1.0 + self.poi) * (1.0 - 2.0 * self.poi)))

        self.init_stress(self.density0, self.fric)


    ##############################################
    # Stress related functions
    ##############################################
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
        res = ti.sqrt(0.5 * (s * s).sum())
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


    ##############################################
    # Stress adaptation
    ##############################################
    @ti.func
    def adapt_stress(self, stress):
        # TODO: add a return of the new DP flag and adaptation flag
        # TODO: what is the usage of dfDP?
        res = stress
        stress_s, vI1, sJ2, fDP_new = self.cal_from_stress(res)
        if fDP_new > self.eps_f:
            if fDP_new > sJ2:
                res = self.adapt_1(res, vI1)
            stress_s, vI1, sJ2, fDP_new = self.cal_from_stress(res)
            res = self.adapt_2(stress_s, vI1, sJ2)
        return res

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

    @ti.func
    def upd_flag_DP(self, stress):
        stress_s, vI1, sJ2, fDP_new = self.cal_from_stress(stress)
        flag = 1	#  Perfectly plastic
        if fDP_new < -self.eps_f:
            flag = 0   # Elastic
        elif fDP_new > self.eps_f:
            if fDP_new >= sJ2:
                flag = 3	# Tension cracking
            else:
                flag = 2	# Imperfectly plastic responses
        return flag

    @ti.func
    def calc_strain_r_p(self, lambda_r, g_d, stress_d):
        res = type_mat3f(0)
        for i in ti.static(range(self.ps.dim3)):
            for j in ti.static(range(self.ps.dim3)):
                if ti.abs(stress_d[i, j]) > self.epsilon:
                    res[i, j] = g_d / stress_d[i, j]
        return res * lambda_r


    ##############################################
    # Tasks
    ##############################################
    @ti.func
    def calc_rep_force_task(self, i, j, ret: ti.template()):
        if self.ps.is_rep_particle(j):
            ret += self.calc_repulsive_force(self.ps.pt[i].x - self.ps.pt[j].x, self.vsound)

    @ti.func
    def regu_density_task(self, i, j, ret: ti.template()):
        if self.ps.is_same_type(i, j):
            ret += self.ps.pt[j].m_V * self.ps.pt[j].density * self.kernel(self.ps.pt[i].x - self.ps.pt[j].x)

    @ti.func
    def regu_stress_task(self, i, j, ret: ti.template()):
        if self.ps.is_same_type(i, j):
            ret += self.ps.pt[j].m_V * self.ps.pt[j].stress * self.kernel(self.ps.pt[i].x - self.ps.pt[j].x)

    @ti.func
    def regu_strain_equ_task(self, i, j, ret: ti.template()):
        if self.ps.is_same_type(i, j):
            ret += self.ps.pt[j].m_V * self.ps.pt[j].strain_equ * self.kernel(self.ps.pt[i].x - self.ps.pt[j].x)

    @ti.func
    def stress_diffusive_term_task(self, i, j, ret: ti.template()):
        if self.ps.is_same_type(i, j):
            K0 = 1 - ti.sin(self.fric)
            tmp = self.kernel_deriv_corr(i, j)
            xij = self.ps.pt[i].x - self.ps.pt[j].x
            ret += self.stress_diffusive_term_Psi(i, j, K0) * ((xij @ tmp.transpose()) / (xij.norm()**2 + 0.01 * self.ps.smoothing_len**2)) * self.ps.pt[j].m_V

    @ti.func
    def stress_diffusive_term_Psi(self, i, j, K0):
        return (self.ps.pt[i].stress_tmp - self.ps.pt[j].stress_tmp) - self.density0 * self.g.y * (self.ps.pt[i].x.y - self.ps.pt[j].x.y) * trans_vec3_diag(type_vec3f(K0, K0, 1))

    @ti.func
    def calc_d_vel_from_stress_task(self, i, j, ret: ti.template()):

        arti_visco = 0.0
        # arti_visco = self.calc_arti_viscosity(1.0, 0.0, i, j, self.vsound) if self.ps.is_soil_particle(j) else 0.0
        tmp = self.ps.pt[j].m_V * self.ps.pt[j].density_tmp * (self.ps.pt[j].stress_tmp / self.ps.pt[j].density_tmp**2 + self.ps.pt[i].stress_tmp / self.ps.pt[i].density_tmp**2 + arti_visco * self.I3) @ self.kernel_deriv_corr(i, j)
        ret += tmp

        if self.ps.is_rigid_dynamic(j):
            self.ps.pt[j].d_vel -= tmp


    ##############################################
    # One step
    ##############################################
    @ti.func
    def calc_d_stress_Bui2008(self, pti):
        g_p = 0.0
        d_stress = type_mat3f(0)
        d_strain_equ = 0.0
        d_strain_equ_p = 0.0
        stress_s, vI1, sJ2, fDP_old = self.cal_from_stress(pti.stress_tmp)
        strain_r = 0.5 * (pti.v_grad + pti.v_grad.transpose())
        spin_r = 0.5 * (pti.v_grad - pti.v_grad.transpose())

        tmp_J = ti.Matrix([[
            pti.stress_tmp[i, 0] * spin_r[j, 0] +
            pti.stress_tmp[i, 1] * spin_r[j, 1] +
            pti.stress_tmp[i, 2] * spin_r[j, 2] +
            pti.stress_tmp[0, j] * spin_r[i, 0] +
            pti.stress_tmp[1, j] * spin_r[i, 1] +
            pti.stress_tmp[2, j] * spin_r[i, 2] for j in range(self.ps.dim3)] for i in range(self.ps.dim3)])

        strain_r_equ = strain_r - strain_r.trace() / 3.0 * self.I3
        tmp_e = 2.0 * self.G * strain_r_equ + self.K * strain_r.trace() * self.I3

        lambda_r = 0.0
        tmp_g = type_mat3f(0)
        if fDP_old >= -self.eps_f and sJ2 > self.epsilon:
            lambda_r = (3.0 * self.alpha_fric * self.K * strain_r.trace() + (self.G / sJ2) * (stress_s * strain_r).sum()) / (
                        27.0 * self.alpha_fric * self.K * ti.sin(self.dila) + self.G)
            tmp_g = lambda_r * (9.0 * self.K * ti.sin(self.dila) * self.I3 + self.G / sJ2 * stress_s)

        d_stress = tmp_J + tmp_e - tmp_g
        d_strain_equ = calc_dev_component(strain_r_equ)

        if fDP_old >= -self.eps_f and sJ2 > self.epsilon:
            g_p = sJ2 + 3 * vI1 * ti.sin(self.dila)
            strain_r_p = self.calc_strain_r_p(lambda_r, g_p - pti.g_p, d_stress)
            strain_r_p_equ = strain_r_p - strain_r_p.trace() / 3.0 * self.I3
            d_strain_equ_p = calc_dev_component(strain_r_p_equ)

        return d_stress, d_strain_equ, d_strain_equ_p

    @ti.kernel
    def one_step(self):
        # ti.loop_config(serialize=True)

        # Adapt tmp stress
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_soil_particle(i):
                self.ps.pt[i].stress_tmp = self.adapt_stress(self.ps.pt[i].stress_tmp)

        # Bdy condition
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_bdy_particle(i) or self.ps.is_rigid(i):
                # # @adami2012
                bdy_v_tmp = type_vec3f(0)
                self.ps.for_all_neighbors(i, self.calc_bdy_vel_task, bdy_v_tmp)
                self.ps.pt[i].v_tmp = 2 * self.ps.pt[i].v - bdy_v_tmp * self.ps.pt[i].CSPM_f

                self.ps.pt[i].density_tmp = self.density0

                bdy_stress_tmp = type_mat3f(0)
                self.ps.for_all_neighbors(i, self.calc_bdy_stress_task, bdy_stress_tmp)
                self.ps.pt[i].stress_tmp = bdy_stress_tmp * self.ps.pt[i].CSPM_f

            if self.ps.is_rigid_dynamic(i):
                self.ps.pt[i].d_vel = self.g

        # Upd acceration from stress in last step
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_soil_particle(i):
                # vel gradient
                v_grad = type_mat3f(0)
                self.ps.for_all_neighbors(i, self.calc_v_grad_task, v_grad)
                self.ps.pt[i].v_grad = v_grad

                # d density
                dd = 0.0
                self.ps.for_all_neighbors(i, self.calc_d_density_task, dd)
                self.ps.pt[i].d_density = dd * self.ps.pt[i].density_tmp

                # d stress
                self.ps.pt[i].d_stress, self.ps.pt[i].d_strain_equ, self.ps.pt[i].d_strain_equ_p = self.calc_d_stress_Bui2008(self.ps.pt[i])

                # stress diffusive term
                # c0 = self.vsound
                # sts_diff = type_mat3f(0)
                # self.ps.for_all_neighbors(i, self.stress_diffusive_term_task, sts_diff)
                # sts_diff *= 2 * 0.1 * self.ps.smoothing_len * c0
                # self.ps.pt[i].d_stress += sts_diff

                # d vel
                d_v = type_vec3f(0)
                self.ps.for_all_neighbors(i, self.calc_d_vel_from_stress_task, d_v)

                # repulsive force
                # if self.ps.flag_boundary == self.ps.bdy_dummy_rep or self.ps.flag_boundary == self.ps.bdy_rep:
                #     self.ps.for_all_neighbors(i, self.calc_rep_force_task, d_v)

                # Fd = 0.0
                Fd = self.calc_viscous_damping(i, self.E)

                self.ps.pt[i].d_vel = d_v + self.g + Fd

        for i in range(self.ps.particle_num[None]):
            if self.ps.is_rigid_static(i):
                self.ps.pt[i].d_vel = type_vec3f(0)

    @ti.func
    def advect_something_func(self, i):
        if self.ps.is_soil_particle(i):
            # regulisation density
            # regu_density = 0.0
            # self.ps.for_all_neighbors(i, self.regu_density_task, regu_density)
            # self.ps.pt[i].density = regu_density * self.ps.pt[i].CSPM_f
            self.chk_density(i, self.density0)

            # regulisation stress
            # regu_stress = type_mat3f(0)
            # self.ps.for_all_neighbors(i, self.regu_stress_task, regu_stress)
            # self.ps.pt[i].stress = regu_stress * self.ps.pt[i].CSPM_f

            # adapt stress
            self.ps.pt[i].flag_retmap = self.upd_flag_DP(self.ps.pt[i].stress)
            self.ps.pt[i].stress = self.adapt_stress(self.ps.pt[i].stress)

            # advect strain equ
            self.ps.pt[i].strain_equ += self.dt[None] * self.ps.pt[i].d_strain_equ
            self.ps.pt[i].strain_equ_p += self.dt[None] * self.ps.pt[i].d_strain_equ_p
