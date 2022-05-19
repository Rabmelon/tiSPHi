import taichi as ti
from .sph_solver import SPHSolver

# TODO: update code. now it is old code

class WCSPHSolver(SPHSolver):
    def __init__(self, particle_system, TDmethod):
        super().__init__(particle_system, TDmethod)
        print("Hallo, class WCSPH Solver starts to serve!")

        # Basic paras
        self.density_0 = 1000.0  # reference density
        self.mass = self.ps.m_V * self.density_0
        self.viscosity = 0.05  # viscosity

        self.d_uelocity = ti.Vector.field(self.ps.dim, dtype=float)
        self.u1234 = ti.Vector.field(self.ps.dim, dtype=float)
        self.F = ti.Vector.field(self.ps.dim, dtype=float)
        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.d_uelocity, self.u1234)
        particle_node.dense(ti.j, 4).place(self.F)

        # Two paras in taichiWCSPH code
        self.stiffness = 50000.0   # k1 for world unit cm
        # self.stiffness = 500.0   # k1 for world unit m
        self.exponent = 7.0     # k2

    ###########################################################################
    @ti.kernel
    def init_data(self):
        for p_i in range(self.ps.particle_num[None]):
            for m in range(4):
                self.F[p_i, m] = ti.Vector([0.0 for _ in range(self.ps.dim)])

    ###########################################################################
    @ti.kernel
    def update_u_1(self, m: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            if m > 0:
                print('m1 =', m, end='; ')
                print('p_i =', p_i)
                continue
            self.u1234[p_i] = self.ps.u[p_i]

    @ti.kernel
    def update_u_234(self, m: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            if m == 0 or m > 3:
                print('m2 =', m, end='; ')
                print('p_i =', p_i)
                continue
            self.u1234[p_i] = self.ps.u[p_i] + 0.5 * self.dt[None] * self.F[p_i, m-1]

    ###########################################################################
    @ti.func
    def update_boundary_particles(self, p_i, p_j):
        self.ps.density[p_j] = self.ps.density[p_i]
        self.u1234[p_j] = (1.0 - min(1.5, 1.0 + self.cal_d_BA(p_i, p_j))) * self.u1234[p_i]
        self.ps.pressure[p_j] = self.ps.pressure[p_i]

    ###########################################################################
    # Evaluate density
    @ti.kernel
    def compute_densities(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            x_i = self.ps.x[p_i]
            self.ps.density[p_i] = 0.0
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                if self.ps.material[p_j] == self.ps.material_dummy:
                    self.update_boundary_particles(p_i, p_j)
                self.ps.density[p_i] += self.ps.m_V * self.cubic_kernel(x_i - x_j)
            self.ps.density[p_i] *= self.density_0
            self.ps.density[p_i] = ti.max(self.ps.density[p_i], self.density_0)

    # Compute the viscosity force contribution, Anti-symmetric formula
    @ti.func
    def viscosity_force(self, p_i, p_j, r):
        v_xy = (self.u1234[p_i] - self.u1234[p_j]).dot(r)
        res = 2 * (self.ps.dim + 2) * self.viscosity * (self.mass / (self.ps.density[p_j])) * v_xy / (r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(r)
        return res

    # Add repulsive forces
    @ti.func
    def compute_repulsive_force(self, r):
        r_norm = r.norm()
        chi = 1 if (r_norm >= 0 and r_norm < 1.5 * self.ps.particle_diameter) else 0
        c = 60000
        gamma = r_norm / (0.75 * self.ps.support_radius)
        f = 0
        if gamma > 0 and gamma <= 2 / 3:
            f = 2 / 3
        elif gamma > 2 / 3 and gamma <= 1:
            f = 2 * gamma - 1.5 * gamma**2
        elif gamma > 1 and gamma < 2:
            f = 0.5 * (2 - gamma)**2
        elif gamma >= 2:
            f = 0
        res = 0.01 * c**2 * chi * f / (r_norm**2) * r
        return res

    # Evaluate viscosity and add gravity
    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            x_i = self.ps.x[p_i]
            d_u = ti.Vector([0.0 for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                if self.ps.material[p_j] == self.ps.material_dummy:
                    self.update_boundary_particles(p_i, p_j)
                d_u += self.viscosity_force(p_i, p_j, x_i - x_j)
                if self.ps.material[p_j] == self.ps.material_repulsive or self.ps.material[p_j] == self.ps.material_dummy:
                    d_u += self.compute_repulsive_force(x_i - x_j)

            # Add body force
            if self.ps.material[p_i] == self.ps.material_fluid:
                # d_u += ti.Vector([0.0, self.g] if self.ps.dim == 2 else [0.0, 0.0, self.g])
                if self.ps.dim == 2:
                    d_u += ti.Vector([0.0, self.g])
                else:
                    print("!!!!!My Error: cannot used in 3D now!")
            self.d_uelocity[p_i] = d_u

    # Compute the pressure force contribution, Symmetric formula
    @ti.func
    def pressure_force(self, p_i, p_j, r):
        res = -self.mass * (self.ps.pressure[p_i] / self.ps.density[p_i]**2 + self.ps.pressure[p_j] / self.ps.density[p_j]**2) * self.cubic_kernel_derivative(r)
        return res

    # Evaluate pressure force
    @ti.kernel
    def compute_pressure_forces(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            self.ps.pressure[p_i] = ti.max(self.stiffness * (ti.pow(self.ps.density[p_i] / self.density_0, self.exponent) - 1.0), 0.0)
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            x_i = self.ps.x[p_i]
            d_u = ti.Vector([0.0 for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                if self.ps.material[p_j] == self.ps.material_dummy:
                    self.update_boundary_particles(p_i, p_j)
                d_u += self.pressure_force(p_i, p_j, x_i - x_j)
            self.d_uelocity[p_i] += d_u

    ###########################################################################
    # Symplectic Euler
    @ti.kernel
    def advect(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.u[p_i] += self.dt[None] * self.d_uelocity[p_i]
                self.ps.x[p_i] += self.dt[None] * self.ps.u[p_i]

    # Update u, x through RK4
    @ti.kernel
    def compute_F(self, m: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            self.F[p_i, m] = self.d_uelocity[p_i]

    @ti.kernel
    def update_particle(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            self.ps.u[p_i] += self.dt[None] / 6 * (
                self.F[p_i, 0] + 2 * self.F[p_i, 1] +
                2 * self.F[p_i, 2] + self.F[p_i, 3])
            self.ps.x[p_i] += self.dt[None] * self.ps.u[p_i]

    def RK4_one_step(self, m):
        # print('RK4 start to compute step', m)
        self.compute_non_pressure_forces()
        self.compute_pressure_forces()
        self.compute_F(m)

    def advect_RK4(self):
        for m in ti.static(range(4)):
            if m == 0:
                self.update_u_1(m)
            elif m < 4:
                self.update_u_234(m)
            self.RK4_one_step(m)
        self.update_particle()

    def substep_RK4(self):
        self.init_data()
        self.compute_densities()
        self.advect_RK4()

    def substep_SympEuler(self):
        self.update_u_1(0)
        self.compute_densities()
        self.compute_non_pressure_forces()
        self.compute_pressure_forces()
        self.advect()