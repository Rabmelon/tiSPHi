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
        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.d_velocity)

        # Two paras in taichiWCSPH code
        self.stiffness = 50000.0   # k1 for world unit cm
        # self.stiffness = 500.0   # k1 for world unit m
        self.exponent = 7.0     # k2


    # Compute the viscosity force contribution, Anti-symmetric formula
    @ti.func
    def viscosity_force(self, p_i, p_j, r):
        v_xy = (self.ps.v[p_i] - self.ps.v[p_j]).dot(r)
        res = 2 * (self.ps.dim + 2) * self.viscosity * (self.mass / (self.ps.density[p_j])) * v_xy / (r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(r)
        return res

    # Compute the pressure force contribution, Symmetric formula
    @ti.func
    def pressure_force(self, p_i, p_j, r):
        res = -self.mass * (self.ps.pressure[p_i] / self.ps.density[p_i]**2 + self.ps.pressure[p_j] / self.ps.density[p_j]**2) * self.cubic_kernel_derivative(r)
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
        self.ps.density[p_j] = self.ps.density[p_i]
        d_BA = self.cal_d_BA(p_i, p_j)
        beta_max = 1.5
        beta = min(beta_max, 1 + d_BA)
        self.ps.v[p_j] = (1 - beta) * self.ps.v[p_i]
        self.ps.pressure[p_j] = self.ps.pressure[p_i]

    # Evaluate density
    @ti.kernel
    def compute_densities(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_boundary:
                self.ps.density[p_i] = self.density_0
                continue
            x_i = self.ps.x[p_i]
            self.ps.density[p_i] = 0.0
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                self.ps.density[p_i] += self.ps.m_V * self.cubic_kernel(x_i - x_j)
            self.ps.density[p_i] *= self.density_0
            self.ps.density[p_i] = ti.max(self.ps.density[p_i], self.density_0)

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
                d_v += self.viscosity_force(p_i, p_j, x_i - x_j)

            # Add body force
            if self.ps.material[p_i] == self.ps.material_water:
                d_v += ti.Vector([0.0, self.g] if self.ps.dim == 2 else [0.0, 0.0, self.g])
            self.d_velocity[p_i] = d_v

    # Evaluate pressure force
    @ti.kernel
    def compute_pressure_forces(self):
        for p_i in range(self.ps.particle_num[None]):
            self.ps.pressure[p_i] = ti.max(self.stiffness * (ti.pow(self.ps.density[p_i] / self.density_0, self.exponent) - 1.0), 0.0)
        for p_i in range(self.ps.particle_num[None]):
            x_i = self.ps.x[p_i]
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                if self.ps.material[p_i] == self.ps.material_water:
                    # Compute Pressure force contribution
                    d_v += self.pressure_force(p_i, p_j, x_i - x_j)
            self.d_velocity[p_i] += d_v

    # Symplectic Euler
    @ti.kernel
    def advect(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_water:
                self.ps.v[p_i] += self.dt[None] * self.d_velocity[p_i]
                self.ps.x[p_i] += self.dt[None] * self.ps.v[p_i]

    def substep(self):
        self.compute_densities()
        self.compute_non_pressure_forces()
        self.compute_pressure_forces()
        self.advect()