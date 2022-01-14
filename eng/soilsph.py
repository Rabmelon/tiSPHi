from numpy.core.arrayprint import DatetimeFormat
from numpy.lib.function_base import piecewise
import taichi as ti
import numpy as np
from .sph_solver import SPHSolver

class SoilSPHSolver(SPHSolver):
    def __init__(self, particle_system, TDmethod, gamma, coh, fric):
        super().__init__(particle_system, TDmethod)
        print("Hallo, class SOILSPH Solver 2D starts to serve!")

        # Basic paras
        self.density_0 = gamma     # reference density of soil, kg/m3
        self.cohesion = coh       # the material cohesion, Pa
        self.friction_deg = fric      # the angle of internal friction, DEG
        self.poisson = 0.3          # Poisson’s ratio
        self.E = 8e7                # Young’s modulus, Pa

        # Paras based on basic paras
        self.mass = self.ps.m_V * self.density_0            # the self.mass of each particle, kg
        self.friction = self.friction_deg / 180 * np.pi     # the angle of internal friction, RAD
        self.Depq = self.E / (1 + self.poisson) / (1 - self.poisson) * ti.Matrix(
                [[1 - self.poisson, self.poisson, 0, self.poisson],
                 [self.poisson, 1 - self.poisson, 0, self.poisson],
                 [0, 0, (1 - 2 * self.poisson) / 2, 0],
                 [self.poisson, self.poisson, 0, 1 - self.poisson]])
        self.alpha_fric = ti.tan(self.friction) / ti.sqrt(9 + 12 * (ti.tan(self.friction))**2)
        self.kc = 3 * self.cohesion / ti.sqrt(9 + 12 * (ti.tan(self.friction))**2)
        self.Gshear = self.E / 2 / (1 + self.poisson)
        self.Kbulk = self.E / 3 / (1 - 2 * self.poisson)

        # Allocate memories
        self.f_stress = ti.Matrix.field(self.ps.dim, self.ps.dim, dtype=float)
        self.f_u = ti.Matrix.field(self.ps.dim_stress, self.ps.dim, dtype=float)
        self.f_stress_grad = ti.Vector.field(self.ps.dim, dtype=float)
        self.f_u_grad = ti.Vector.field(self.ps.dim_stress, dtype=float)
        self.f_ext = ti.Vector.field(self.ps.dim, dtype=float)
        self.g_p = ti.Vector.field(self.ps.dim_stress, dtype=float)     # item in constitutive equation
        self.g_DP = ti.field(dtype=float)               # the value of checking stress state
        self.s = ti.Vector.field(self.ps.dim_stress, dtype=float)       # the deviatoric stress
        self.p = ti.field(dtype=float)                  # the hydrostatic pressure
        self.I1 = ti.field(dtype=float)                 # the firse invariant of the stress tensor
        self.sJ2 = ti.field(dtype=float)                # sqrt of the second invariant of the deviatoric stress tensor
        self.r_sigma = ti.field(dtype=float)            # the scaling factor
        self.spin = ti.field(dtype=float)               # the spin rate tensor
        self.Jaumann = ti.Vector.field(self.ps.dim_stress, dtype=float)     # the Jaumann stress rate, tilde σ
        self.F1 = ti.Vector.field(self.ps.dim, dtype=float)
        self.F2 = ti.Vector.field(self.ps.dim_stress, dtype=float)
        self.u1234 = ti.Vector.field(self.ps.dim, dtype=float)
        self.stress1234 = ti.Vector.field(self.ps.dim_stress, dtype=float)
        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.f_stress, self.f_u, self.f_stress_grad, self.f_u_grad, self.f_ext, self.g_p, self.g_DP, self.s, self.p, self.I1, self.sJ2, self.spin, self.Jaumann, self.u1234, self.stress1234)
        particle_node.dense(ti.j, 4).place(self.F1, self.F2)

    @ti.kernel
    def init_data(self):
        for p_i in range(self.ps.particle_num[None]):
            for m in range(4):
                self.F1[p_i, m] = ti.Vector([0.0 for _ in range(self.ps.dim)])
                self.F2[p_i, m] = ti.Vector([0.0 for _ in range(self.ps.dim_stress)])

    @ti.kernel
    def update_u_stress_1(self, m: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            if m > 0:
                print('m1 =', m, end='; ')
                print('p_i =', p_i)
                continue
            # assert m > 0, 'My Error: m > 0 when it should be 0!'
            self.u1234[p_i] = self.ps.v[p_i]
            self.stress1234[p_i] = self.ps.stress[p_i]

    @ti.kernel
    def update_u_stress_234(self, m: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            if m == 0 or m > 3:
                print('m2 =', m, end='; ')
                print('p_i =', p_i)
                continue
            # assert m == 0, 'My Error: m = 0 when it should be 1, 2, 3!'
            # assert m > 3, 'My Error: m > 3 when it should be 1, 2, 3!'
            self.u1234[p_i] = self.ps.v[p_i] + 0.5 * self.dt[None] * self.F1[p_i, m-1]
            self.stress1234[p_i] = self.ps.stress[p_i] + 0.5 * self.dt[None] * self.F2[p_i, m-1]

    # Assign constant density
    @ti.kernel
    def compute_densities(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_soil:
                self.ps.density[p_i] = self.density_0

    # Calculate term fσ and fu
    @ti.kernel
    def compute_term_f(self):
        for p_i in range(self.ps.particle_num[None]):
            self.f_stress[p_i] = ti.Matrix(
                [[self.stress1234[p_i][0], self.stress1234[p_i][2]],
                 [self.stress1234[p_i][2], self.stress1234[p_i][1]]])
            self.f_u[p_i] = ti.Matrix(
                [[self.Depq[0, 0] * self.u1234[p_i][0], self.Depq[0, 1] * self.u1234[p_i][1]],
                 [self.Depq[1, 0] * self.u1234[p_i][0], self.Depq[1, 1] * self.u1234[p_i][1]],
                 [self.Depq[2, 2] * self.u1234[p_i][1], self.Depq[2, 2] * self.u1234[p_i][0]],
                 [self.Depq[3, 0] * self.u1234[p_i][0], self.Depq[3, 1] * self.u1234[p_i][1]]])

    # TODO: Check stress state and adapt
    @ti.kernel
    def compute_g_DP(self):
        for p_i in range(self.ps.particle_num[None]):
            self.I1[p_i] = self.stress1234[p_i][0] + self.stress1234[p_i][1] + self.stress1234[p_i][3]
            self.p[p_i] = -self.I1[p_i] / 3
            self.s[p_i] = ti.Vector([self.stress1234[p_i][0] - self.p[p_i], self.stress1234[p_i][1] - self.p[p_i], self.stress1234[p_i][2], self.stress1234[p_i][3] - self.p[p_i]])
            self.sJ2[p_i] = ti.sqrt(0.5 * (self.s[p_i][0]**2 + self.s[p_i][1]**2 + 2 * self.s[p_i][2]**2 + self.s[p_i][3]**2))
            self.g_DP[p_i] = self.sJ2[p_i] + self.alpha_fric * self.I1[p_i] - self.kc

    @ti.func
    def adapt_stress(self, p_i):
        flag_state = -self.alpha_fric * self.I1[p_i] + self.kc
        return flag_state

    @ti.kernel
    def check_adapt_stress_DP(self):
        pass

    # Update boundary particles
    @ti.kernel
    def update_boundary(self):
        pass


    # Calculate gradients of fσ and fu
    @ti.func
    def compute_f_stress_grad(self, p_i, p_j, r):
        tmp = self.mass * (self.f_stress[p_i] / self.ps.density[p_i]**2 + self.f_stress[p_j] / self.ps.density[p_j]**2)
        tmp_ckd = self.cubic_kernel_derivative(r)
        res = tmp@tmp_ckd
        return res

    @ti.func
    def compute_f_u_grad(self, p_i, p_j, r):
        tmp = self.mass / self.ps.density[p_j] * (self.f_u[p_j] - self.f_u[p_i])
        tmp_ckd = self.cubic_kernel_derivative(r)
        res = tmp@tmp_ckd
        return res

    @ti.func
    def cal_d_BA(self, p_i, p_j):
        x_i = self.ps.x[p_i]
        x_j = self.ps.x[p_j]
        boundary = ti.Vector([
            self.ps.bound[1] - self.ps.padding, self.ps.padding,
            self.ps.bound[0] - self.ps.padding, self.ps.padding])
        db_i = ti.Vector([x_i[1] - boundary[0], x_i[1] - boundary[1], x_i[0] - boundary[2], x_i[0] - boundary[3]])
        db_j = ti.Vector([x_j[1] - boundary[0], x_j[1] - boundary[1], x_j[0] - boundary[2], x_j[0] - boundary[3]])

        flag_b = db_i * db_j
        flag_dir = flag_b < 0

        if sum(flag_dir) > 1:
            flag_choose = abs(flag_dir * db_i)
            tmp_max = 0
            for i in ti.static(range(4)):
                tmp_max = max(tmp_max, flag_choose[i])
            flag_choose -= tmp_max
            flag_choose = flag_choose == 0.0
            flag_dir -= flag_choose     # will cause a warning: Local store may lose precision & Atomic add (i32 to f32) may lose precision

        d_A = abs(db_i.dot(flag_dir))
        d_B = abs(db_j.dot(flag_dir))
        return d_B / d_A

    @ti.func
    def update_boundary_particles(self, p_i, p_j):
        self.ps.density[p_j] = self.density_0

        d_BA = self.cal_d_BA(p_i, p_j)
        beta_max = 1.5
        beta = min(beta_max, 1 + d_BA)
        self.u1234[p_j] = (1 - beta) * self.u1234[p_i]

        self.stress1234[p_j] = self.stress1234[p_i]

        self.f_stress[p_j] = ti.Matrix([[self.stress1234[p_i][0], self.stress1234[p_i][2]],
                                        [self.stress1234[p_i][2], self.stress1234[p_i][1]]])
        self.f_u[p_j] = ti.Matrix([[self.Depq[0, 0] * self.u1234[p_i][0], self.Depq[0, 1] * self.u1234[p_i][1]],
                                   [self.Depq[1, 0] * self.u1234[p_i][0], self.Depq[1, 1] * self.u1234[p_i][1]],
                                   [self.Depq[2, 2] * self.u1234[p_i][1], self.Depq[2, 2] * self.u1234[p_i][0]],
                                   [self.Depq[3, 0] * self.u1234[p_i][0], self.Depq[3, 1] * self.u1234[p_i][1]]])

    @ti.kernel
    def compute_f_grad(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            x_i = self.ps.x[p_i]
            f_stress_grad_i = ti.Vector([0.0 for _ in range(self.ps.dim)])
            f_u_grad_i = ti.Vector([0.0 for _ in range(self.ps.dim_stress)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                if self.ps.material[p_j] == self.ps.material_dummy:
                    self.update_boundary_particles(p_i, p_j)
                f_stress_grad_i += self.compute_f_stress_grad(p_i, p_j, x_i - x_j)
                f_u_grad_i += self.compute_f_u_grad(p_i, p_j, x_i - x_j)
            self.f_stress_grad[p_i] = f_stress_grad_i
            self.f_u_grad[p_i] = f_u_grad_i

    # Assign external forces
    @ti.func
    def compute_f_ext(self, p_i):
        self.f_ext[p_i] = ti.Vector([0.0, self.g] if self.ps.dim == 2 else [0.0, 0.0, self.g])

    # TODO: Calculate plastic strain
    @ti.func
    def compute_g_p(self, p_i):
        self.g_p[p_i] = ti.Vector([0.0 for _ in range(self.ps.dim_stress)])

    # TODO: Calculate the Jaumann stress rate
    @ti.func
    def compute_Jaumann(self, p_i):
        self.Jaumann[p_i] = ti.Vector([0.0 for _ in range(self.ps.dim_stress)])

    # Compute F1 and F2
    @ti.kernel
    def compute_F(self, m: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            self.compute_f_ext(p_i)
            self.compute_Jaumann(p_i)
            self.compute_g_p(p_i)
            self.F1[p_i, m] = self.f_stress_grad[p_i] + self.f_ext[p_i]
            self.F2[p_i, m] = self.Jaumann[p_i] + self.f_u_grad[p_i] - self.g_p[p_i]

    # Update u, σ, x through RK4
    @ti.kernel
    def update_particle(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_soil:
                continue
            if self.ps.material[p_i] == self.ps.material_soil:
                self.ps.v[p_i] += self.dt[None] / 6 * (
                    self.F1[p_i, 0] + 2 * self.F1[p_i, 1] +
                    2 * self.F1[p_i, 2] + self.F1[p_i, 3])
                self.ps.stress[p_i] += self.dt[None] / 6 * (
                    self.F2[p_i, 0] + 2 * self.F2[p_i, 1] +
                    2 * self.F2[p_i, 2] + self.F2[p_i, 3])
                self.ps.x[p_i] += self.dt[None] * self.ps.v[p_i]

    def RK4_one_step(self, m):
        # print('RK4 start to compute step', m)
        self.update_boundary()
        self.check_adapt_stress_DP()
        self.compute_term_f()
        self.compute_f_grad()
        self.compute_F(m)

    def advect_RK4(self):
        for m in ti.static(range(4)):
            if m == 0:
                self.update_u_stress_1(m)
            elif m < 4:
                self.update_u_stress_234(m)
            self.RK4_one_step(m)
        self.update_particle()

    def substep(self):
        self.init_data()
        self.compute_densities()
        self.advect_RK4()
