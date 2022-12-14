import taichi as ti
import numpy as np
from eng.particle_system import ParticleSystem
from eng.type_define import *


# TODO: CSPM not working, for zero denominator
# TODO: MLS not working now

@ti.data_oriented
class SPHBase:
    def __init__(self, particle_system: ParticleSystem):
        self.ps = particle_system
        self.g = type_vec3f(np.array(self.ps.cfg.get_cfg("gravitation")))
        self.flagKernel = self.ps.cfg.get_cfg("kernel")     # 0: Cubic spline, 1: Wendland C2
        self.flagKernelCorr = self.ps.cfg.get_cfg("kernelCorrection")    # 0: None, 1: CSPM, 2: MLS
        self.flagTI = self.ps.cfg.get_cfg("timeIntegration")     # 1: SE, 2: LF, 4: RK
        self.flagXSPH = self.ps.cfg.get_cfg("xsph")
        self.dt_min = self.ps.cfg.get_cfg("timeStepSizeMin")
        self.dt = ti.field(float, shape=())
        self.dt[None] = self.dt_min
        self.I = ti.Matrix(np.eye(self.ps.dim))
        self.I3 = ti.Matrix(np.eye(3))
        self.epsilon = 1e-8
        self.alert_ratio = 0.01

        self.init_rigid_body()

    ##############################################
    # Task template
    ##############################################
    @ti.func
    def tmp_task(self, i, j, ret: ti.template()):
        if self.ps.is_real_particle(j):
            ret += 0.0

    ##############################################
    # Time integration
    ##############################################
    def step(self):
        self.ps.initialize_particle_system()
        self.calc_kernel_corr()
        if self.ps.flag_boundary == self.ps.bdy_dummy:
            self.calc_boundary_dist()
        self.init_real_particles_tmp()
        self.substep()
        self.advect_pos()
        self.advect_something()
        self.solve_rigid_body()
        self.enforce_boundary()

    def substep(self):
        if self.flagTI == 1:
            self.substep_SE()
        elif self.flagTI == 2:
            self.substep_LF()
        elif self.flagTI == 4:
            self.substep_RK()

    @ti.kernel
    def one_step(self):
        pass

    @ti.kernel
    def init_real_particles_tmp(self):
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_real_particle(i):
                self.ps.pt[i].density_tmp = self.ps.pt[i].density
                self.ps.pt[i].v_tmp = self.ps.pt[i].v
                self.ps.pt[i].stress_tmp = self.ps.pt[i].stress

    @ti.func
    def upd_from_tmp(self, i):
        if self.ps.is_real_particle(i):
            self.ps.pt[i].density = self.ps.pt[i].density_tmp
            self.ps.pt[i].v = self.ps.pt[i].v_tmp
            self.ps.pt[i].stress = self.ps.pt[i].stress_tmp

    ##############################################
    # SE
    @ti.kernel
    def advect_SE(self):
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_real_particle(i):
                self.ps.pt[i].density += self.dt[None] * self.ps.pt[i].d_density
                self.ps.pt[i].set_volume()
                self.ps.pt[i].v += self.dt[None] * self.ps.pt[i].d_vel
                self.ps.pt[i].stress += self.dt[None] * self.ps.pt[i].d_stress

    def substep_SE(self):
        self.one_step()
        self.advect_SE()


    ##############################################
    # LF
    @ti.kernel
    def advect_LF_half(self):
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_real_particle(i):
                self.ps.pt[i].density_tmp += 0.5 * self.dt[None] * self.ps.pt[i].d_density
                self.ps.pt[i].m_V = self.ps.pt[i].mass / self.ps.pt[i].density_tmp
                self.ps.pt[i].v_tmp += 0.5 * self.dt[None] * self.ps.pt[i].d_vel
                self.ps.pt[i].stress_tmp += 0.5 * self.dt[None] * self.ps.pt[i].d_stress

    @ti.kernel
    def advect_LF(self):
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_real_particle(i):
                self.ps.pt[i].density += self.dt[None] * self.ps.pt[i].d_density
                self.ps.pt[i].set_volume()
                self.ps.pt[i].v += self.dt[None] * self.ps.pt[i].d_vel
                self.ps.pt[i].stress += self.dt[None] * self.ps.pt[i].d_stress

    def substep_LF(self):
        self.one_step()
        self.advect_LF_half()
        self.one_step()
        self.advect_LF()


    ##############################################
    # RK # ! too large increase!!!
    @ti.kernel
    def advect_RK_4(self):
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_real_particle(i):
                self.ps.pt[i].density_tmp = 0.5 * self.dt[None] * self.ps.pt[i].d_density + self.ps.pt[i].density
                self.ps.pt[i].m_V = self.ps.pt[i].mass / self.ps.pt[i].density_tmp
                self.ps.pt[i].v_tmp = 0.5 * self.dt[None] * self.ps.pt[i].d_vel + self.ps.pt[i].v
                self.ps.pt[i].stress_tmp = 0.5 * self.dt[None] * self.ps.pt[i].d_stress + self.ps.pt[i].stress

    @ti.kernel
    def init_RK(self):
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_real_particle(i):
                self.ps.pt[i].d_density_RK = 0.0
                self.ps.pt[i].d_vel_RK = type_vec3f(0)
                self.ps.pt[i].d_stress_RK = type_mat3f(0)

    @ti.kernel
    def update_RK(self, m: int):
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_real_particle(i):
                self.ps.pt[i].d_density_RK += self.ps.pt[i].d_density * m
                self.ps.pt[i].d_vel_RK += self.ps.pt[i].d_vel * m
                self.ps.pt[i].d_stress_RK += self.ps.pt[i].d_stress * m

    @ti.kernel
    def advect_RK(self):
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_real_particle(i):
                self.ps.pt[i].density += self.dt[None] / 6.0 * self.ps.pt[i].d_density_RK
                self.ps.pt[i].set_volume()
                self.ps.pt[i].v += self.dt[None] / 6.0 * self.ps.pt[i].d_vel_RK
                self.ps.pt[i].stress += self.dt[None] / 6.0 * self.ps.pt[i].d_stress_RK

    def substep_RK(self):
        self.init_RK()
        m_RK = [1, 2, 2, 1]
        for i in range(4):
            self.one_step()
            self.update_RK(m_RK[i])
            if i < 3:
                self.advect_RK_4()
        self.advect_RK()


    ##############################################
    # Public tasks
    ##############################################
    @ti.func
    def calc_v_grad_task(self, i, j, ret: ti.template()):
        tmp = self.kernel_deriv_corr(i, j)

        # if self.ps.is_bdy_particle(j) or self.ps.is_rigid(j):
        #     self.calc_dummy_v_tmp(i, j)

        ret += self.ps.pt[j].m_V * (self.ps.pt[j].v_tmp - self.ps.pt[i].v_tmp) @ tmp.transpose()
        # if self.ps.pt[j].mat_type == self.ps.pt[i].mat_type:
        #     tmp = self.kernel_deriv_corr(i, j)
        #     ret += self.ps.pt[j].m_V * (self.ps.pt[j].v_tmp - self.ps.pt[i].v_tmp) @ tmp.transpose()

    @ti.func
    def calc_d_density_task(self, i, j, ret: ti.template()):
        # * need to multiply pti.density_tmp after summation

        # if self.ps.is_bdy_particle(j) or self.ps.is_rigid(j):
        #     self.calc_dummy_v_tmp(i, j)

        tmp = self.ps.pt[j].m_V * (self.ps.pt[i].v_tmp - self.ps.pt[j].v_tmp).transpose() @ self.kernel_deriv_corr(i, j)
        ret += tmp[0]

    @ti.func
    def calc_d_vel_from_stress_task(self, i, j, ret: ti.template()):
        arti_visco = self.calc_arti_viscosity_task(0.0, 0.0, i, j, self.vsound) if self.ps.is_soil_particle(j) else 0.0
        tmp = self.ps.pt[j].m_V * self.ps.pt[j].density_tmp * (self.ps.pt[j].stress_tmp / self.ps.pt[j].density_tmp**2 + self.ps.pt[i].stress_tmp / self.ps.pt[i].density_tmp**2 + arti_visco * self.I3) @ self.kernel_deriv_corr(i, j)
        ret += tmp
        if self.ps.is_rigid_dynamic(j):
            self.ps.pt[j].d_vel -= tmp



    ##############################################
    # Assist
    ##############################################
    def calc_dt_CFL(self, CFL_component, vsound, dt_min):
        dt = CFL_component * self.ps.smoothing_len / vsound
        return ti.max(dt_min, dt - dt % dt_min)

    @ti.func
    def chk_density(self, i, density0):
        density_min = density0
        # density_min = density0 * (1 - self.alert_ratio)
        self.ps.pt[i].density = ti.max(density_min, self.ps.pt[i].density)
        # density_max = density0 * (1 + self.alert_ratio)
        # self.ps.pt[i].density = ti.min(density_max, self.ps.pt[i].density)

    @ti.func
    def xsph_task(self, i, j, ret: ti.template()):
        if self.ps.pt[j].mat_type < 10:
            ret += self.ps.pt[j].m_V * (self.ps.pt[j].v - self.ps.pt[i].v) * self.kernel(self.ps.pt[i].x - self.ps.pt[j].x)

    @ti.kernel
    def advect_pos(self):
        xsph_component = 0.5    # * 0 ~ 1.0, 0.5 from @huModelingGranularMaterial2021
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_real_particle(i):
                xsph_sum = type_vec3f(0)
                if self.flagXSPH:
                    self.ps.for_all_neighbors(i, self.xsph_task, xsph_sum)
                self.ps.pt[i].x += self.dt[None] * (self.ps.pt[i].v + xsph_component * xsph_sum)

    @ti.func
    def advect_something_func(self, i):
        pass

    @ti.kernel
    def advect_something(self):
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_real_particle(i):
                self.advect_something_func(i)

    @ti.kernel
    def init_stress(self, density0: float, fric: float):
        ymax = -float('Inf')
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_soil_particle(i):
                ti.atomic_max(ymax, self.ps.pt[i].x.y)

        K0 = 1.0 - ti.sin(fric)
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_soil_particle(i):
                ver_stress = density0 * self.g.y * (ymax - self.ps.pt[i].x.y)
                self.ps.pt[i].set_stress_diag(K0 * ver_stress, ver_stress, K0 * ver_stress)

    @ti.kernel
    def init_pressure(self, density0: float):
        ymax = -float('Inf')
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_fluid_particle(i):
                ti.atomic_max(ymax, self.ps.pt[i].x.y)

        for i in range(self.ps.particle_num[None]):
            if self.ps.is_fluid_particle(i):
                self.ps.pt[i].pressure = -density0 * self.g.y * (ymax - self.ps.pt[i].x.y)



    ##############################################
    # Kernel functions
    ##############################################
    @ti.func
    def kernel(self, r):
        res = 0.0
        if self.flagKernel == 0:
            res = self.cubic_kernel(r)
        elif self.flagKernel == 1:
            res = self.WendlandC2_kernel(r)
        return res

    @ti.func
    def kernel_derivative(self, r):
        res = ti.Vector([0.0 for _ in range(self.ps.dim3)])
        if self.flagKernel == 0:
            res = self.cubic_kernel_derivative(r)
        elif self.flagKernel == 1:
            res = self.WendlandC2_kernel_derivative(r)
        return res

    ##############################################
    # Cubic spline kernel
    @ti.func
    def cubic_kernel(self, r):
        res = 0.0
        h1 = 1.0 / self.ps.smoothing_len
        k = 1.0 if self.ps.dim == 1 else 15.0 / 7.0 / ti.math.pi if self.ps.dim == 2 else 3.0 / 2.0 / ti.math.pi
        k *= h1**self.ps.dim
        r_norm = r.norm()
        q = r_norm * h1
        if r_norm > self.epsilon and q <= 2.0:
            if q <= 1.0:
                q2 = q * q
                q3 = q2 * q
                res = k * (0.5 * q3 - q2 + 2.0 / 3.0)
            else:
                res = k / 6.0 * ti.pow(2.0 - q, 3.0)
        return res

    @ti.func
    def cubic_kernel_derivative(self, r):
        res = ti.Vector([0.0 for _ in range(self.ps.dim3)])
        h1 = 1.0 / self.ps.smoothing_len
        k = 1.0 if self.ps.dim == 1 else 15.0 / 7.0 / ti.math.pi if self.ps.dim == 2 else 3.0 / 2.0 / ti.math.pi
        k *= h1**self.ps.dim
        r_norm = r.norm()
        q = r_norm * h1
        if r_norm > self.epsilon and q <= 2.0:
            grad_q = r / r_norm * h1
            if q <= 1.0:
                res = k * q * (3.0 / 2.0 * q - 2.0) * grad_q
            else:
                factor = 2.0 - q
                res = k * (-0.5 * factor * factor) * grad_q
        return res

    ##############################################
    # Wendland C2 kernel
    @ti.func
    def WendlandC2_kernel(self, r):
        res = 0.0
        h1 = 1.0 / self.ps.smoothing_len
        k = 7.0 / (4.0 * ti.math.pi) if self.ps.dim == 2 else 21.0 / (2.0 * ti.math.pi) if self.ps.dim == 3 else 0.0
        k *= h1**self.ps.dim
        r_norm = r.norm()
        q = r_norm * h1
        if r_norm > self.epsilon and q <= 2.0:
            q1 = 1.0 - 0.5 * q
            res = k * ti.pow(q1, 4.0) * (1.0 + 2.0 * q)
        return res

    @ti.func
    def WendlandC2_kernel_derivative(self, r):
        res = ti.Vector([0.0 for _ in range(self.ps.dim3)])
        h1 = 1.0 / self.ps.smoothing_len
        k = 7.0 / (4.0 * ti.math.pi) if self.ps.dim == 2 else 21.0 / (2.0 * ti.math.pi) if self.ps.dim == 3 else 0.0
        k *= h1**self.ps.dim
        r_norm = r.norm()
        q = r_norm * h1
        if r_norm > self.epsilon and q <= 2.0:
            q1 = 1.0 - 0.5 * q
            res = k * ti.pow(q1, 3.0) * (-5.0 * q) * h1 * r / r_norm
        return res

    ##############################################
    # Kernel correction
    ##############################################
    def calc_kernel_corr(self):
        self.calc_CSPM_f()
        if self.flagKernelCorr == 1:
            self.calc_CSPM_L()
        elif self.flagKernelCorr == 2:
            pass

    @ti.func
    def kernel_corr(self, pti, ptj):
        pass

    @ti.func
    def kernel_deriv_corr(self, i, j):
        tmp = type_vec3f(0)
        if self.flagKernelCorr == 0:
            tmp = self.kernel_derivative(self.ps.pt[i].x - self.ps.pt[j].x)
        elif self.flagKernelCorr == 1:
            tmp = self.ps.pt[i].CSPM_L @ self.kernel_derivative(self.ps.pt[i].x - self.ps.pt[j].x)
        elif self.flagKernelCorr == 2:
            pass
        return tmp

    # CSPM
    @ti.func
    def calc_CSPM_f_task(self, i, j, ret: ti.template()):
        # if self.ps.pt[j].mat_type == self.ps.pt[i].mat_type:
        if self.ps.is_real_particle(j):
            ret += self.ps.pt[j].m_V * self.kernel(self.ps.pt[i].x - self.ps.pt[j].x)

    @ti.kernel
    def calc_CSPM_f(self):
        for i in range(self.ps.particle_num[None]):
            tmp_CSPM_f = 0.0
            self.ps.for_all_neighbors(i, self.calc_CSPM_f_task, tmp_CSPM_f)
            self.ps.pt[i].CSPM_f = 1.0 / tmp_CSPM_f if tmp_CSPM_f != 0.0 else 1.0

    @ti.func
    def calc_CSPM_L_task(self, i, j, ret: ti.template()):
        if self.ps.pt[j].mat_type == self.ps.pt[i].mat_type:
            tmp = self.kernel_derivative(self.ps.pt[i].x - self.ps.pt[j].x)
            ret += self.ps.pt[j].m_V * (self.ps.pt[j].x - self.ps.pt[i].x) @ tmp.transpose()

    @ti.kernel
    def calc_CSPM_L(self):
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_real_particle(i):
                tmp_CSPM_L = type_mat3f(0)
                tmp_CSPM_L_inv = ti.math.eye(3)
                self.ps.for_all_neighbors(i, self.calc_CSPM_L_task, tmp_CSPM_L)
                if self.ps.dim == 2:
                    tmp2 = trans_mat_3_2(tmp_CSPM_L)
                    if ti.math.determinant(tmp2) > self.epsilon:
                        tmp_CSPM_L_inv = trans_mat_2_3_fill0((tmp2).inverse())
                elif self.ps.dim == 3:
                    if ti.math.determinant(tmp_CSPM_L) > self.epsilon:
                        tmp_CSPM_L_inv = tmp_CSPM_L.inverse()
                self.ps.pt[i].CSPM_L = tmp_CSPM_L_inv

    ##############################################
    # MLS
    @ti.func
    def calc_MLS_beta_task(self, i, j, ret: ti.template()):
        if self.ps.is_real_particle(i):
            xij = self.ps.pt[i].x - self.ps.pt[j].x
            p = type_vec4f(1, xij)
            Ap = p @ p.transpose()
            ret += self.ps.pt[j].m_V * Ap * self.kernel(xij)

    @ti.kernel
    def calc_MLS_beta(self):
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_real_particle(i):
                multi = type_vec4f(1,0,0,0)
                tmp_A0 = type_mat4f(0)
                self.ps.for_all_neighbors(i, self.calc_MLS_beta_task, tmp_A0)
                self.ps.pt[i].MLS_beta = tmp_A0.inverse() @ multi       # ! maybe error in 1 / 0

    @ti.func
    def calc_MLS_target_task(self, x):
        # * target function
        return x.sum()

    @ti.func
    def calc_MLS_task(self, i, j, ret: ti.template()):
        if self.ps.is_real_particle(i):
            xij = self.ps.pt[i].x - self.ps.pt[j].x
            Wij_MLS = self.kernel(xij) * (self.ps.pt[i].MLS_beta * type_vec4f(1, xij)).sum()
            ret += self.ps.pt[i].m_V * Wij_MLS * self.calc_MLS_target_task(self.ps.pt[j].x)

    # * target varable kernel function template
    @ti.kernel
    def calc_MLS(self):
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_real_particle(i):
                tmp_value = 0
                self.ps.for_all_neighbors(i, self.calc_MLS_task, tmp_value)

    ##############################################
    # Rigid body
    ##############################################
    def init_rigid_body(self):
        for r_obj_id in self.ps.object_id_rigid_body:
            self.calc_rigid_rest_m(r_obj_id)

    def solve_rigid_body(self):
        for r_obj_id in self.ps.object_id_rigid_body:
            R = self.solve_constraints(r_obj_id)


    @ti.kernel
    def solve_constraints(self, object_id: int) -> type_mat3f:
        # compute center of mass
        cm = self.calc_cm(object_id)

        A = type_mat3f(0)
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_rigid_dynamic(i) and self.ps.pt[i].obj_id == object_id:
                q = self.ps.pt[i].x0 - self.ps.rigid_rest_cm[object_id]
                p = self.ps.pt[i].x - cm
                A += self.ps.pt[i].m_V * self.ps.pt[i].density * p @ q.transpose()
        R, S = ti.polar_decompose(A)

        if all(abs(R) < self.epsilon):
            R = ti.Matrix.identity(type_f, self.ps.dim3)

        for i in range(self.ps.particle_num[None]):
            if self.ps.is_rigid_dynamic(i) and self.ps.pt[i].obj_id == object_id:
                goal = cm + R @ (self.ps.pt[i].x0 - self.ps.rigid_rest_cm[object_id])
                corr = (goal - self.ps.pt[i].x) * 1.0
                self.ps.pt[i].x += corr
        return R

    @ti.func
    def calc_cm(self, object_id):
        sum_m = 0.0
        cm = type_vec3f(0)
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_rigid_dynamic(i) and self.ps.pt[i].obj_id == object_id:
                cm += self.ps.pt[i].mass * self.ps.pt[i].x
                sum_m += self.ps.pt[i].mass
        cm /= sum_m
        return cm

    @ti.kernel
    def calc_cm_kernel(self, object_id: int) -> type_vec3f:
        return self.calc_cm(object_id)

    @ti.kernel
    def calc_rigid_rest_m(self, object_id: int):
        self.ps.rigid_rest_cm[object_id] = self.calc_cm(object_id)


    ##############################################
    # Boundary treatment
    ##############################################
    # Compulsory collision
    def enforce_boundary(self):
        if self.ps.flag_boundary == self.ps.bdy_collision:
            if self.ps.dim == 3:
                self.enforce_boundary_3D()
            elif self.ps.dim == 2:
                self.enforce_boundary_2D()

    @ti.kernel
    def enforce_boundary_3D(self):
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_real_particle(i) and self.ps.pt[i].is_dynamic:
                pos = self.ps.pt[i].x
                dist_pt_r = self.ps.particle_radius - self.epsilon
                collision_normal = type_vec3f(0)
                if pos[0] > self.ps.domain_end[0] - dist_pt_r:
                    collision_normal[0] += 1.0
                    self.ps.pt[i].x[0] = self.ps.domain_end[0] - dist_pt_r
                if pos[0] <= self.ps.domain_start[0] + dist_pt_r:
                    collision_normal[0] += -1.0
                    self.ps.pt[i].x[0] = self.ps.domain_start[0] + dist_pt_r

                # if pos[1] > self.ps.domain_end[1] - dist_pt_r:
                # collision_normal[1] += 1.0
                # self.ps.pt[i].x[1] = self.ps.domain_end[1] - dist_pt_r
                if pos[1] <= self.ps.domain_start[1] + dist_pt_r:
                    collision_normal[1] += -1.0
                    self.ps.pt[i].x[1] = self.ps.domain_start[1] + dist_pt_r

                if pos[2] > self.ps.domain_end[2] - dist_pt_r:
                    collision_normal[2] += 1.0
                    self.ps.pt[i].x[2] = self.ps.domain_end[2] - dist_pt_r
                if pos[2] <= self.ps.domain_start[2] + dist_pt_r:
                    collision_normal[2] += -1.0
                    self.ps.pt[i].x[2] = self.ps.domain_start[2] + dist_pt_r

                collision_normal_length = collision_normal.norm()
                if collision_normal_length > self.epsilon:
                    self.simulate_collisions(i, collision_normal / collision_normal_length)

    @ti.kernel
    def enforce_boundary_2D(self):
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_real_particle(i) and self.ps.pt[i].is_dynamic:
                pos = self.ps.pt[i].x
                dist_pt_r = self.ps.particle_radius - self.epsilon
                collision_normal = type_vec3f(0)
                if pos[0] > self.ps.domain_end[0] - dist_pt_r:
                    collision_normal[0] += 1.0
                    self.ps.pt[i].x[0] = self.ps.domain_end[0] - dist_pt_r
                if pos[0] <= self.ps.domain_start[0] + dist_pt_r:
                    collision_normal[0] += -1.0
                    self.ps.pt[i].x[0] = self.ps.domain_start[0] + dist_pt_r

                # if pos[1] > self.ps.domain_end[1] - dist_pt_r:
                #     collision_normal[1] += 1.0
                #     self.ps.pt[i].x[1] = self.ps.domain_end[1] - dist_pt_r
                if pos[1] <= self.ps.domain_start[1] + dist_pt_r:
                    collision_normal[1] += -1.0
                    self.ps.pt[i].x[1] = self.ps.domain_start[1] + dist_pt_r

                collision_normal_length = collision_normal.norm()
                if collision_normal_length > self.epsilon:
                    self.simulate_collisions(i, collision_normal / collision_normal_length)

    @ti.func
    def simulate_collisions(self, i, vec):
        # Collision factor, assume roughly (1-c_f)*velocity loss after collision
        c_f = 0.3
        self.ps.pt[i].v -= (1.0 + c_f) * self.ps.pt[i].v.dot(vec) * vec

    ##############################################
    # Dummy particles
    # enforcing method, non-slip, from @huModelingGranularMaterial2021
    @ti.func
    def calc_dummy_v_tmp(self, i, j):
        # i: real particle, j: dummy particle
        # from @huModelingGranularMaterial2021
        self.ps.pt[j].v_tmp = (self.ps.pt[j].dist_B / self.ps.pt[i].dist_B) * (self.ps.pt[j].v - self.ps.pt[i].v_tmp) + self.ps.pt[j].v
        # from @bui2008
        # beta = ti.min(1.5, 1.0 + self.ps.pt[j].dist_B / self.ps.pt[i].dist_B)
        # beta = 1.25
        # self.ps.pt[j].v_tmp = (1 - beta) * self.ps.pt[i].v_tmp + beta * self.ps.pt[j].v

    @ti.kernel
    def calc_boundary_dist(self):
        for i in range(self.ps.particle_num[None]):
            self.calc_boundary_dist_task(i)

    @ti.func
    def calc_boundary_dist_task(self, i):
        d = self.epsilon
        chi_numerator = 0.0
        chi_denominator = 0.0
        self.ps.for_all_neighbors(i, self.calc_chi_numerator_task, chi_numerator)
        self.ps.for_all_neighbors(i, self.calc_chi_denominator_task, chi_denominator)
        chi = chi_numerator / (chi_denominator) if chi_denominator > self.epsilon and chi_numerator > self.epsilon else 0.5
        chi = ti.min(ti.max(chi, 0.5), 1.0)
        d = self.ps.support_radius * (2 * chi - 1)
        if self.ps.is_fluid_particle(i) or self.ps.is_soil_particle(i):
            d = ti.max(d, ti.sqrt(3.0) * self.ps.smoothing_len / 4.0)
        self.ps.pt[i].dist_B = d

    @ti.func
    def calc_chi_numerator_task(self, i, j, ret: ti.template()):
        if self.ps.pt[j].mat_type == self.ps.pt[i].mat_type:
            ret += self.kernel(self.ps.pt[i].x - self.ps.pt[j].x)

    @ti.func
    def calc_chi_denominator_task(self, i, j, ret: ti.template()):
        ret += self.kernel(self.ps.pt[i].x - self.ps.pt[j].x)

    # smoothed velocity method, non-slip, from @adamiGeneralizedWallBoundary2012
    # i: bdy pt, j: real pt
    @ti.func
    def calc_bdy_density_task(self, i, j, ret: ti.template()):
        if self.ps.is_real_particle(j):
            ret += self.ps.pt[j].m_V * self.ps.pt[j].density_tmp * self.kernel(self.ps.pt[i].x - self.ps.pt[j].x)

    @ti.func
    def calc_bdy_vel_task(self, i, j, ret: ti.template()):
        if self.ps.is_real_particle(j):
            ret += self.ps.pt[j].m_V * self.ps.pt[j].v_tmp * self.kernel(self.ps.pt[i].x - self.ps.pt[j].x)

    @ti.func
    def calc_bdy_pressure_task(self, i, j, ret: ti.template()):
        if self.ps.is_real_particle(j):
            ret += self.ps.pt[j].m_V * self.ps.pt[j].pressure * self.kernel(self.ps.pt[i].x - self.ps.pt[j].x)

    @ti.func
    def calc_bdy_stress_task(self, i, j, ret: ti.template()):
        if self.ps.is_real_particle(j):
            ret += self.ps.pt[j].m_V * self.ps.pt[j].stress_tmp * self.kernel(self.ps.pt[i].x - self.ps.pt[j].x)


    ##############################################
    # Repulsive particles
    @ti.func
    def calc_repulsive_force(self, r, vsound):
        r_norm = r.norm()
        r_judge = self.ps.particle_diameter
        # r_judge = 1.5 * self.ps.particle_diameter
        chi = 1.0 - r_norm / r_judge if (r_norm > 0.0 and r_norm < r_judge) else 0.0
        gamma = r_norm / (0.75 * self.ps.smoothing_len)
        f = 0.0
        if gamma > 0 and gamma <= 2 / 3:
            f = 2 / 3
        elif gamma > 2 / 3 and gamma <= 1:
            f = 2 * gamma - 1.5 * gamma**2
        elif gamma > 1 and gamma < 2:
            f = 0.5 * (2 - gamma)**2
        res = 0.01 * vsound**2 * chi * f / (r_norm**2) * r
        return res


    ##############################################
    # Artificial terms
    ##############################################
    @ti.func
    def calc_arti_viscosity_task(self, alpha_Pi, beta_Pi, i, j, vsound):
        pti = self.ps.pt[i]
        ptj = self.ps.pt[j]
        res = 0.0
        vare = 0.01
        xij = pti.x - ptj.x
        vij = pti.v_tmp - ptj.v_tmp     # ! v or v_tmp???
        vijxij = (vij * xij).sum()
        if vijxij < 0.0:
            rhoij = 0.5 * (pti.density_tmp + ptj.density_tmp)
            hij = self.ps.smoothing_len
            cij = vsound
            phiij = hij * vijxij / ((xij.norm())**2 + vare * hij**2)
            res = (-alpha_Pi * cij * phiij + beta_Pi * phiij**2) / rhoij
        return res



    ##############################################
    # Visualization
    ##############################################
    @ti.kernel
    def assign_value_color(self):
        for i in range(self.ps.particle_num[None]):
            if self.ps.is_real_particle(i) or (self.ps.show_bdy and self.ps.is_dummy_particle(i)):
                if self.ps.color_title == 1:
                    self.ps.pt[i].val = self.ps.pt[i].id0
                elif self.ps.color_title == 2:
                    self.ps.pt[i].val = self.ps.pt[i].density
                elif self.ps.color_title == 21:
                    self.ps.pt[i].val = self.ps.pt[i].d_density
                elif self.ps.color_title == 3:
                    self.ps.pt[i].val = self.ps.pt[i].v.norm()
                elif self.ps.color_title == 31:
                    self.ps.pt[i].val = self.ps.pt[i].v.x
                elif self.ps.color_title == 32:
                    self.ps.pt[i].val = self.ps.pt[i].v.y
                elif self.ps.color_title == 33:
                    self.ps.pt[i].val = self.ps.pt[i].v.z
                elif self.ps.color_title == 34:
                    self.ps.pt[i].val = self.ps.pt[i].d_vel.norm()
                elif self.ps.color_title == 35:
                    self.ps.pt[i].val = self.ps.pt[i].d_vel.x
                elif self.ps.color_title == 36:
                    self.ps.pt[i].val = self.ps.pt[i].d_vel.y
                elif self.ps.color_title == 37:
                    self.ps.pt[i].val = self.ps.pt[i].d_vel.z
                elif self.ps.color_title == 4:
                    self.ps.pt[i].val = self.ps.pt[i].x.norm()
                elif self.ps.color_title == 41:
                    self.ps.pt[i].val = self.ps.pt[i].x.x
                elif self.ps.color_title == 42:
                    self.ps.pt[i].val = self.ps.pt[i].x.y
                elif self.ps.color_title == 43:
                    self.ps.pt[i].val = self.ps.pt[i].x.z
                elif self.ps.color_title == 44:
                    self.ps.pt[i].val = (self.ps.pt[i].x - self.ps.pt[i].x0).norm()
                elif self.ps.color_title == 51:
                    self.ps.pt[i].val = self.ps.pt[i].stress[0,0]
                elif self.ps.color_title == 52:
                    self.ps.pt[i].val = -self.ps.pt[i].stress[1,1]
                elif self.ps.color_title == 53:
                    self.ps.pt[i].val = self.ps.pt[i].stress[2,2]
                elif self.ps.color_title == 54:
                    self.ps.pt[i].val = self.ps.pt[i].stress[0,1]
                elif self.ps.color_title == 55:
                    self.ps.pt[i].val = self.ps.pt[i].stress[1,2]
                elif self.ps.color_title == 56:
                    self.ps.pt[i].val = self.ps.pt[i].stress[2,0]
                elif self.ps.color_title == 61:
                    self.ps.pt[i].val = self.ps.pt[i].strain_equ
                elif self.ps.color_title == 7:
                    self.ps.pt[i].val = self.ps.pt[i].pressure

                elif self.ps.color_title == 100:    # grid Id test
                    self.ps.pt[i].val = self.ps.pt[i].grid_ids
                elif self.ps.color_title == 101:    # random value drawing test
                    self.ps.pt[i].val = ti.random()
                elif self.ps.color_title == 102:
                    self.ps.pt[i].val = self.ps.pt[i].CSPM_L[0,0]
                    # if self.ps.pt[i].dist_B < 0.048 and self.ps.pt[i].dist_B > 0.0:
                    #     print("pt", i, self.ps.pt[i].id0, self.ps.pt[i].dist_B)
                elif self.ps.color_title == 103:
                    self.ps.pt[i].val = self.ps.pt[i].v_grad[0,0]

    ##############################################
    # Test
    ##############################################
