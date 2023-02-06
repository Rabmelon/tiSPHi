import taichi as ti
from eng.solver_sph_base import SPHBase
from eng.type_define import *


class WCSPHSolver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        print("WCSPH starts to serve!")

        # ! now only for one kind of fluid
        self.density0 = self.ps.mat_fluid[0]["density0"]
        self.viscosity = self.ps.mat_fluid[0]["viscosity"]
        self.stiffness = self.ps.mat_fluid[0]["stiffness"]
        self.exponent = self.ps.mat_fluid[0]["exponent"]
        # self.vsound = 40.0      # @Liu2012
        self.vsound = 60

        # self.vsound2 = 20 * ti.sqrt(9.81 * 0.6)
        # self.vsound = ti.sqrt(self.vsound2)
        self.dt[None] = self.calc_dt_CFL(CFL_component=0.2, vsound=self.vsound, dt_min=self.dt_min)

        # self.init_pressure(self.density0)


    ##############################################
    # Tasks
    ##############################################
    @ti.func
    def calc_density_task(self, i, j, ret: ti.template()):
        ret += self.ps.pt[j].mass * self.kernel(self.ps.pt[i].x - self.ps.pt[j].x)

    @ti.func
    def viscosity_force(self, i, j):
        pti = self.ps.pt[i]
        ptj = self.ps.pt[j]
        xij = pti.x - ptj.x
        res = type_vec3f(0)

        if self.ps.is_fluid_particle(j):
            res = 2 * (self.ps.dim + 2) * self.viscosity * ptj.m_V * ti.min((pti.v_tmp - ptj.v_tmp).dot(xij), 0.0) / (xij.norm()**2 + 0.01 * self.ps.smoothing_len**2) * self.kernel_derivative(xij)	# default
            # res = 2 * self.viscosity * (pti.m_V**2 + ptj.m_V**2) * (pti.v_tmp - ptj.v_tmp) / xij.norm() * self.kernel_derivative(xij) / pti.mass	# Adami2012
        elif self.ps.is_bdy_particle(j) or self.ps.is_rigid(j):
            res = 2 * (self.ps.dim + 2) * self.viscosity * ptj.m_V * self.density0 / pti.density_tmp * ti.min((pti.v_tmp - ptj.v_tmp).dot(xij), 0.0) / (xij.norm()**2 + 0.01 * self.ps.smoothing_len**2) * self.kernel_derivative(xij)

        return res

    @ti.func
    def pressure_force(self, i, j):
        pti = self.ps.pt[i]
        ptj = self.ps.pt[j]
        res = -self.density0 * ptj.m_V * (pti.pressure / pti.density_tmp**2 + ptj.pressure / ptj.density_tmp**2) * self.kernel_derivative(pti.x - ptj.x)	# Splishsplash
        # res = -ptj.density_tmp * ptj.m_V * (pti.pressure / pti.density_tmp**2 + ptj.pressure / ptj.density_tmp**2) * self.kernel_derivative(pti.x - ptj.x)	# default
        # res = -(pti.m_V**2 + ptj.m_V**2) * (ptj.density_tmp * pti.pressure + pti.density_tmp * ptj.pressure) / (pti.density_tmp + ptj.density_tmp) * self.kernel_deriv_corr(i, j) / pti.mass	# Adami2012
        return res

    @ti.func
    def calc_d_vel_task(self, i, j, ret: ti.template()):

        # if self.ps.is_bdy_particle(j) or self.ps.is_rigid(j):
        #     self.calc_dummy_v_tmp(i, j)
        #     self.ps.pt[j].pressure = self.ps.pt[i].pressure
        #     self.ps.pt[j].density_tmp = self.density0

        # arti_visco = self.calc_arti_viscosity(0.5, 0.0, i, j, self.vsound) if self.ps.is_fluid_particle(j) else 0.0
        # tmp = arti_visco * self.kernel_deriv_corr(i, j) + self.viscosity_force(i, j) + self.pressure_force(i, j)
        tmp = self.viscosity_force(i, j) + self.pressure_force(i, j)
        ret += tmp

        if self.ps.is_rigid_dynamic(j):
            self.ps.pt[j].d_vel -= tmp

    @ti.func
    def calc_rep_force_task(self, i, j, ret: ti.template()):
        if self.ps.is_rep_particle(j):
            ret += self.calc_repulsive_force(self.ps.pt[i].x - self.ps.pt[j].x, self.vsound)


    ##############################################
    # One step
    ##############################################
    @ti.kernel
    def one_step(self):
        # ti.loop_config(serialize=True)

        for i in range(self.ps.particle_num[None]):
            if self.ps.is_fluid_particle(i):
                self.ps.pt[i].pressure = ti.max(self.stiffness * (ti.pow(self.ps.pt[i].density_tmp / self.density0, self.exponent) - 1.0), 0.0)

            if self.ps.is_bdy_particle(i) or self.ps.is_rigid(i):
                # @adami2012
                bdy_v_tmp = type_vec3f(0)
                self.ps.for_all_neighbors(i, self.calc_bdy_vel_task, bdy_v_tmp)
                self.ps.pt[i].v_tmp = 2 * self.ps.pt[i].v - bdy_v_tmp * self.ps.pt[i].CSPM_f

                self.ps.pt[i].density_tmp = self.density0
                # bdy_density_tmp = 0.0
                # self.ps.for_all_neighbors(i, self.calc_bdy_density_task, bdy_density_tmp)
                # self.ps.pt[i].density_tmp = ti.max(bdy_density_tmp * self.ps.pt[i].CSPM_f, self.density0)

                bdy_pressure_tmp = 0.0
                self.ps.for_all_neighbors(i, self.calc_bdy_pressure_task, bdy_pressure_tmp)
                self.ps.pt[i].pressure = ti.max(bdy_pressure_tmp * self.ps.pt[i].CSPM_f, 0.0)

            if self.ps.is_rigid_dynamic(i):
                self.ps.pt[i].d_vel = self.g

        for i in range(self.ps.particle_num[None]):
            if self.ps.is_fluid_particle(i):
                # d density
                dd = 0.0
                self.ps.for_all_neighbors(i, self.calc_d_density_task, dd)
                self.ps.pt[i].d_density = dd * self.ps.pt[i].density_tmp

                # d vel
                tmp_d_vel = type_vec3f(0)
                self.ps.for_all_neighbors(i, self.calc_d_vel_task, tmp_d_vel)

                # repulsive force
                if self.ps.flag_boundary == self.ps.bdy_dummy_rep or self.ps.flag_boundary == self.ps.bdy_rep:
                    self.ps.for_all_neighbors(i, self.calc_rep_force_task, tmp_d_vel)

                self.ps.pt[i].d_vel = tmp_d_vel + self.g

            if self.ps.is_rigid_static(i):
                self.ps.pt[i].d_vel = type_vec3f(0)


    @ti.func
    def advect_something_func(self, i):
        if self.ps.is_fluid_particle(i):
            self.chk_density(i, self.density0)

