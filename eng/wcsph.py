import taichi as ti
from .sph_solver import SPHSolver

class WCSPHSolver(SPHSolver):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        print("Hallo, class WCSPH Solver starts to serve!")

        # Basic paras
        self.density_0 = 1000.0  # reference density
        self.mass = self.ps.m_V * self.density_0
        self.viscosity = 0.05  # viscosity

        self.d_velocity = ti.Vector.field(self.ps.dim, dtype=float)
        self.v1234 = ti.Vector.field(self.ps.dim, dtype=float)
        self.F = ti.Vector.field(self.ps.dim, dtype=float)
        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.d_velocity, self.v1234)
        particle_node.dense(ti.j, 4).place(self.F)

        # Two paras in taichiWCSPH code
        self.stiffness = 50000.0   # k1 for world unit cm
        # self.stiffness = 500.0   # k1 for world unit m
        self.exponent = 7.0     # k2

    @ti.kernel
    def init_data(self):
        for p_i in range(self.ps.particle_num[None]):
            for m in range(4):
                self.F[p_i, m] = ti.Vector([0.0 for _ in range(self.ps.dim)])

    @ti.kernel
    def update_v_1(self, m: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_water:
                continue
            if m > 0:
                print('m1 =', m, end='; ')
                print('p_i =', p_i)
                continue
            # assert m > 0, 'My Error: m > 0 when it should be 0!'
            self.v1234[p_i] = self.ps.v[p_i]

    @ti.kernel
    def update_v_234(self, m: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_water:
                continue
            if m == 0 or m > 3:
                print('m2 =', m, end='; ')
                print('p_i =', p_i)
                continue
            # assert m == 0, 'My Error: m = 0 when it should be 1, 2, 3!'
            # assert m > 3, 'My Error: m > 3 when it should be 1, 2, 3!'
            self.v1234[p_i] = self.ps.v[p_i] + 0.5 * self.dt[None] * self.F[p_i, m-1]

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
        self.ps.density[p_j] = self.ps.density[p_i]
        self.v1234[p_j] = (1.0 - min(1.5, 1.0 + self.cal_d_BA(p_i, p_j))) * self.v1234[p_i]
        self.ps.pressure[p_j] = self.ps.pressure[p_i]

    # Evaluate density
    @ti.kernel
    def compute_densities(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_water:
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
        v_xy = (self.v1234[p_i] - self.v1234[p_j]).dot(r)
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
            if self.ps.material[p_i] != self.ps.material_water:
                continue
            x_i = self.ps.x[p_i]
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                if self.ps.material[p_j] == self.ps.material_dummy:
                    self.update_boundary_particles(p_i, p_j)
                d_v += self.viscosity_force(p_i, p_j, x_i - x_j)
                if self.ps.material[p_j] == self.ps.material_repulsive or self.ps.material[p_j] == self.ps.material_dummy:
                    d_v += self.compute_repulsive_force(x_i - x_j)

            # Add body force
            if self.ps.material[p_i] == self.ps.material_water:
                d_v += ti.Vector([0.0, self.g] if self.ps.dim == 2 else [0.0, 0.0, self.g])
            self.d_velocity[p_i] = d_v

    # Compute the pressure force contribution, Symmetric formula
    @ti.func
    def pressure_force(self, p_i, p_j, r):
        res = -self.mass * (self.ps.pressure[p_i] / self.ps.density[p_i]**2 + self.ps.pressure[p_j] / self.ps.density[p_j]**2) * self.cubic_kernel_derivative(r)
        return res

    # Evaluate pressure force
    @ti.kernel
    def compute_pressure_forces(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_water:
                continue
            self.ps.pressure[p_i] = ti.max(self.stiffness * (ti.pow(self.ps.density[p_i] / self.density_0, self.exponent) - 1.0), 0.0)
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_water:
                continue
            x_i = self.ps.x[p_i]
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                if self.ps.material[p_j] == self.ps.material_dummy:
                    self.update_boundary_particles(p_i, p_j)
                d_v += self.pressure_force(p_i, p_j, x_i - x_j)
            self.d_velocity[p_i] += d_v

    # Symplectic Euler
    @ti.kernel
    def advect(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_water:
                self.ps.v[p_i] += self.dt[None] * self.d_velocity[p_i]
                self.ps.x[p_i] += self.dt[None] * self.ps.v[p_i]

    # Update v, x through RK4
    @ti.kernel
    def compute_F(self, m: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_water:
                continue
            self.F[p_i, m] = self.d_velocity[p_i]

    @ti.kernel
    def update_particle(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_water:
                continue
            self.ps.v[p_i] += self.dt[None] / 6 * (
                self.F[p_i, 0] + 2 * self.F[p_i, 1] +
                2 * self.F[p_i, 2] + self.F[p_i, 3])
            self.ps.x[p_i] += self.dt[None] * self.ps.v[p_i]

    def RK4_one_step(self, m):
        # print('RK4 start to compute step', m)
        self.compute_non_pressure_forces()
        self.compute_pressure_forces()
        self.compute_F(m)

    def advect_RK4(self):
        for m in ti.static(range(4)):
            if m == 0:
                self.update_v_1(m)
            elif m < 4:
                self.update_v_234(m)
            self.RK4_one_step(m)
        self.update_particle()

    def substep_RK4(self):
        self.init_data()
        self.compute_densities()
        self.advect_RK4()

    def substep(self):
        self.update_v_1(0)
        self.compute_densities()
        self.compute_non_pressure_forces()
        self.compute_pressure_forces()
        self.advect()