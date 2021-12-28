from numpy.lib.function_base import piecewise
import taichi as ti
import numpy as np
from .sph_solver import SPHSolver

class SoilSPHSolver(SPHSolver):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        print("Hallo, class soil SPH Solver 2D starts to serve!")

        # Basic paras
        self.density_0 = 1850.0     # reference density of soil, kg/m3
        self.cohesion = 32720       # the material cohesion, Pa
        self.friction_deg = 36      # the angle of internal friction, DEG
        self.poisson = 0.3          # Poisson’s ratio
        self.E = 8e7                # Young’s modulus, Pa

        # Paras based on basic paras
        self.mass = self.ps.m_V * self.density_0            # the mass of each particle, kg
        self.friction = self.friction_deg / 180 * np.pi     # the angle of internal friction, RAD
        self.Depq = self.E / (1 + self.poisson) / (1 - self.poisson) * ti.Matrix(
                [[1 - self.poisson, self.poisson, 0, self.poisson],
                 [self.poisson, 1 - self.poisson, 0, self.poisson],
                 [0, 0, (1 - 2 * self.poisson) / 2, 0],
                 [self.poisson, self.poisson, 0, 1 - self.poisson]])
        self.alpha_fric = ti.tan(self.friction) / ti.sqrt(9 + 12 * (ti.tan(self.friction))**2)
        self.kc = 3 * self.cohesion / ti.sqrt(9 + 12 * (ti.tan(self.friction))**2)

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
        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.f_stress, self.f_u, self.f_stress_grad, self.f_u_grad, self.f_ext, self.g_p, self.g_DP, self.s, self.p, self.I1, self.sJ2, self.spin, self.Jaumann)

    # Assign density
    @ti.kernel
    def compute_densities(self):
        for p_i in range(self.ps.particle_num[None]):
            self.ps.density[p_i] = self.density_0

    # Calculate term fσ and fu
    @ti.kernel
    def compute_term_f(self):
        for p_i in range(self.ps.particle_num[None]):
            self.f_stress[p_i] = ti.Matrix(
                [[self.ps.stress[p_i][0], self.ps.stress[p_i][2]],
                 [self.ps.stress[p_i][2], self.ps.stress[p_i][1]]])
            self.f_u[p_i] = ti.Matrix(
                [[self.Depq[1, 1] * self.ps.v[p_i][0], self.Depq[1, 2] * self.ps.v[p_i][1]],
                 [self.Depq[2, 1] * self.ps.v[p_i][0], self.Depq[2, 2] * self.ps.v[p_i][1]],
                 [self.Depq[3, 3] * self.ps.v[p_i][1], self.Depq[3, 3] * self.ps.v[p_i][0]],
                 [self.Depq[4, 1] * self.ps.v[p_i][0], self.Depq[4, 2] * self.ps.v[p_i][1]]])

    # Check stress state and adaptation
    @ti.kernel
    def compute_g_DP(self):
        for p_i in range(self.ps.particle_num[None]):
            self.I1[p_i] = self.ps.stress[p_i][0] + self.ps.stress[p_i][1] + self.ps.stress[p_i][3]
            self.p[p_i] = -self.I1[p_i] / 3
            self.s[p_i] = ti.Vector([self.ps.stress[p_i][0] - self.p[p_i], self.ps.stress[p_i][1] - self.p[p_i], self.ps.stress[p_i][2], self.ps.stress[p_i][3] - self.p[p_i]])
            self.sJ2[p_i] = ti.sqrt(0.5 * (self.s[p_i][0]**2 + self.s[p_i][1]**2 + 2 * self.s[p_i][2]**2 + self.s[p_i][3]**2))
            self.g_DP[p_i] = self.sJ2[p_i] + self.alpha_fric * self.I1[p_i] - self.kc

    @ti.fun
    def adapt_stress(self, p_i):
        flag_state = -self.alpha_fric * self.I1[p_i] + self.kc
        return flag_state

    @ti.kernel
    def check_adapt_stress_DP(self):
        pass

    # Update boundary particles
    @ti.kernel
    def update_boundary(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_boundary:
                pass


    # Calculate gradients of fσ and fu
    @ti.fun
    def compute_f_stress_grad(self, p_i, p_j, r):
        tmp = self.mass * (self.f_stress[p_i] / self.ps.density[p_i]**2 + self.f_stress[p_j] / self.ps.density[p_j]**2)
        tmp_x = [tmp[0,0], tmp[0,1]]
        tmp_y = [tmp[1,0], tmp[1,1]]
        tmp_ckd = self.cubic_kernel_derivative(r)
        res = ti.Vector([tmp_x.dot(tmp_ckd), tmp_y.dot(tmp_ckd)])
        return res

    @ti.fun
    def compute_f_u_grad(self, p_i, p_j, r):
        tmp = self.mass / self.ps.density[p_j] * (self.f_u[p_j] - self.f_u[p_i])
        tmp_xx = [tmp[0,0], tmp[0,1]]
        tmp_yy = [tmp[1,0], tmp[1,1]]
        tmp_xy = [tmp[2,0], tmp[2,1]]
        tmp_zz = [tmp[3,0], tmp[3,1]]
        tmp_ckd = self.cubic_kernel_derivative(r)
        res = ti.Vector([tmp_xx.dot(tmp_ckd), tmp_yy.dot(tmp_ckd), tmp_xy.dot(tmp_ckd), tmp_zz.dot(tmp_ckd)])
        return res

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
                f_stress_grad_i += self.compute_f_stress_grad(p_i, p_j, x_i - x_j)
                f_u_grad_i += self.compute_f_u_grad(p_i, p_j, x_i - x_j)
            self.f_stress_grad[p_i] += f_stress_grad_i
            self.f_u_grad[p_i] += f_u_grad_i

    # Assign external forces
    @ti.kernel
    def compute_f_ext(self):
        for p_i in range(self.ps.particle_num[None]):
            self.f_ext[p_i] = ti.Vector([0.0, self.g] if self.ps.dim == 2 else [0.0, 0.0, self.g])

    # Calculate plastic strain

    # Calculate the Jaumann stress rate

    # Compute F1 and F2

    # Update u, σ, x through RK4
    @ti.kernel
    def advect_RK4(self):
        pass

    def substep(self):
        self.compute_densities()
        self.advect_RK4()