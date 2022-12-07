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
        self.vsound = 40.0      # @Liu2012

        self.init_pressure(self.density0)


    ##############################################
    # Tasks
    ##############################################
    @ti.func
    def calc_density_task(self, i, j, ret: ti.template()):
        ret += self.ps.pt[j].mass * self.kernel(self.ps.pt[i].x - self.ps.pt[j].x)

    @ti.func
    def viscosity_force(self, i, j):
        r = self.ps.pt[i].x - self.ps.pt[j].x
        res = 2 * (self.ps.dim + 2) * self.viscosity * self.ps.pt[j].m_V * (self.ps.pt[i].v_tmp - self.ps.pt[j].v_tmp).dot(r) / (r.norm()**2 + 0.01 * self.ps.smoothing_len**2) * self.kernel_deriv_corr(i, j)

        if self.ps.is_rigid_dynamic(j):
            self.ps.pt[j].d_vel += -res

        return res

    @ti.func
    def calc_non_pressure_force_task(self, i, j, ret: ti.template()):
        ret += self.viscosity_force(i, j)

    @ti.func
    def pressure_force(self, i, j):
        res = -self.ps.pt[j].density_tmp * self.ps.pt[j].m_V * (self.ps.pt[i].pressure / self.ps.pt[i].density_tmp**2 + self.ps.pt[j].pressure / self.ps.pt[j].density_tmp**2) * self.kernel_deriv_corr(i, j)

        if self.ps.is_rigid_dynamic(j):
            self.ps.pt[j].d_vel += -res

        return res

    @ti.func
    def calc_pressure_force_task(self, i, j, ret: ti.template()):
        ret += self.pressure_force(i, j)

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

        # for i in range(self.ps.particle_num[None]):
        #     if self.ps.is_fluid_particle(i):
        #         # density
        #         tmp_density = 0.0
        #         self.ps.for_all_neighbors(i, self.calc_density_task, tmp_density)
        #         # self.ps.pt[i].density = tmp_density
        #         self.ps.pt[i].density_tmp = tmp_density * self.ps.pt[i].CSPM_f

        for i in range(self.ps.particle_num[None]):
            if self.ps.is_bdy_particle(i) or self.ps.is_rigid_dynamic(i):
                # @adami2012
                bdy_v_tmp = type_vec3f(0)
                self.ps.for_all_neighbors(i, self.calc_bdy_vel_task, bdy_v_tmp)
                self.ps.pt[i].v_tmp = 2 * self.ps.pt[i].v - bdy_v_tmp * self.ps.pt[i].CSPM_f

                bdy_density_tmp = 0.0
                self.ps.for_all_neighbors(i, self.calc_bdy_density_task, bdy_density_tmp)
                self.ps.pt[i].density_tmp = ti.max(bdy_density_tmp * self.ps.pt[i].CSPM_f, self.density0)
                # self.ps.pt[i].density_tmp = self.density0

                bdy_pressure_tmp = 0.0
                self.ps.for_all_neighbors(i, self.calc_bdy_pressure_task, bdy_pressure_tmp)
                self.ps.pt[i].pressure = bdy_pressure_tmp * self.ps.pt[i].CSPM_f

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
                self.ps.pt[i].pressure = ti.max(self.stiffness * (ti.pow(self.ps.pt[i].density_tmp / self.density0, self.exponent) - 1.0), 0.0)

                # non-pressure force
                self.ps.for_all_neighbors(i, self.calc_non_pressure_force_task, tmp_d_vel)

                # pressure force
                self.ps.for_all_neighbors(i, self.calc_pressure_force_task, tmp_d_vel)

                # repulsive force
                if self.ps.flag_boundary == self.ps.bdy_dummy_rep or self.ps.flag_boundary == self.ps.bdy_rep:
                    self.ps.for_all_neighbors(i, self.calc_rep_force_task, tmp_d_vel)

                self.ps.pt[i].d_vel = tmp_d_vel + self.g

            if self.ps.is_rigid_static(i):
                self.ps.pt[i].d_vel = type_vec3f(0)

            # if self.ps.pt[i].id0 == 0:
            #     pti = self.ps.pt[i]
            #     print("==", pti.id0, i, pti.x[0:2], pti.density_tmp, pti.d_density, pti.v_tmp[0:2], pti.d_vel[0:2])
            # if self.ps.pt[i].id0 == 15:
            #     pti = self.ps.pt[i]
            #     print("==", pti.id0, i, pti.x[0:2], pti.density_tmp, pti.d_density, pti.v_tmp[0:2], pti.d_vel[0:2])



    @ti.func
    def advect_something_func(self, i):
        if self.ps.is_fluid_particle(i):
            self.chk_density(i, self.density0)

