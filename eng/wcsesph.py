import taichi as ti
from .sph_solver import SPHSolver

class WCSESPHSolver(SPHSolver):
    def __init__(self, particle_system, kernel, rho, visco, stiff, expo):
        super().__init__(particle_system, kernel)
        print("Hallo, class WCSPH Solver starts to serve!")

        # Basic paras
        self.density_0 = rho  # reference density
        self.mass = self.ps.m_V * self.density_0
        self.viscosity = visco  # viscosity

        self.pressure = ti.field(dtype=float)
        self.d_velocity = ti.Vector.field(self.ps.dim, dtype=float)
        self.d_density = ti.field(dtype=float)
        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.pressure, self.d_density, self.d_velocity)

        # Two paras in taichiWCSPH code
        self.stiffness = stiff   # k1
        self.exponent = expo     # k2

    @ti.kernel
    def init_value(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] < 10:
                self.ps.val[p_i] = self.ps.v[p_i].norm()
                # self.ps.val[p_i] = -self.ps.x[p_i][1]
                # self.ps.val[p_i] = self.ps.density[p_i]
                # self.ps.val[p_i] = self.pressure[p_i]
                # self.ps.val[p_i] = p_i

    @ti.func
    def update_boundary_particles(self, p_i, p_j):
        self.ps.density[p_j] = self.density_0
        self.ps.v[p_j] = (1.0 - min(1.5, 1.0 + self.cal_d_BA(p_i, p_j))) * self.ps.v[p_i]
        self.pressure[p_j] = self.pressure[p_i]

    # Evaluate density
    @ti.kernel
    def compute_d_density(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            x_i = self.ps.x[p_i]
            drho = 0.0
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                if self.ps.material[p_j] == self.ps.material_dummy:
                    self.update_boundary_particles(p_i, p_j)
                tmp = (self.ps.v[p_i] - self.ps.v[p_j]).transpose() @ self.kernel_derivative(x_i - x_j)
                # tmp = (self.ps.v[p_i] - self.ps.v[p_j]).transpose() @ (self.ps.L[p_i] @ self.kernel_derivative(x_i - x_j))
                drho += self.ps.density[p_j] * self.ps.m_V * tmp[0]
            self.d_density[p_i] = drho

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
                self.ps.density[p_i] += self.ps.m_V * self.kernel(x_i - x_j)
            self.ps.density[p_i] *= self.density_0

    # Compute the viscosity force contribution, Anti-symmetric formula
    @ti.func
    def viscosity_force(self, p_i, p_j, r):
        v_xy = (self.ps.v[p_i] - self.ps.v[p_j]).dot(r)
        res = 2 * (self.ps.dim + 2) * self.viscosity * (self.mass / (self.ps.density[p_j])) * v_xy / (r.norm()**2 + 0.01 * self.ps.smoothing_len**2) * self.kernel_derivative(r)
        # res = 2 * (self.ps.dim + 2) * self.viscosity * (self.mass / (self.ps.density[p_j])) * v_xy / (r.norm()**2 + 0.01 * self.ps.smoothing_len**2) * (self.ps.L[p_i] @ self.kernel_derivative(r))
        return res

    # Evaluate viscosity and add gravity
    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            x_i = self.ps.x[p_i]
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                if self.ps.material[p_j] == self.ps.material_dummy:
                    self.update_boundary_particles(p_i, p_j)
                d_v += self.viscosity_force(p_i, p_j, x_i - x_j)

            # Add body force
            if self.ps.material[p_i] == self.ps.material_fluid:
                # d_v += ti.Vector([0.0, self.g] if self.ps.dim == 2 else [0.0, 0.0, self.g])
                if self.ps.dim == 2:
                    d_v += ti.Vector([0.0, self.g])
                else:
                    print("!!!!!My Error: cannot used in 3D now!")
            self.d_velocity[p_i] = d_v

    # Compute the pressure force contribution, Symmetric formula
    @ti.func
    def pressure_force(self, p_i, p_j, r):
        res = -self.ps.density[p_j] * self.ps.m_V * (self.pressure[p_i] / self.ps.density[p_i]**2 + self.pressure[p_j] / self.ps.density[p_j]**2) * self.kernel_derivative(r)
        return res

    # Evaluate pressure force
    @ti.kernel
    def compute_pressure_forces(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            self.pressure[p_i] = ti.max(self.stiffness * (ti.pow(self.ps.density[p_i] / self.density_0, self.exponent) - 1.0), 0.0)
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
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
    def advect_density(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.density[p_i] += self.dt[None] * self.d_density[p_i]

    @ti.kernel
    def chk_density(self):
        for p_i in range(self.ps.particle_num[None]):
            self.ps.density[p_i] = ti.max(self.density_0, self.ps.density[p_i])
            if self.ps.density[p_i] > self.density_0 * 1.25:
                print("stop because particle", p_i, "has a density", self.ps.density[p_i], "and pressure", self.pressure[p_i], "with neighbour num", self.ps.particle_neighbors_num[p_i])
            assert self.ps.density[p_i] < self.density_0 * 1.25

    @ti.kernel
    def advect(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.v[p_i] += self.dt[None] * self.d_velocity[p_i]
                self.ps.x[p_i] += self.dt[None] * self.ps.v[p_i]

    def substep(self):
        self.compute_densities()
        # self.compute_d_density()
        # self.advect_density()
        self.chk_density()
        self.compute_non_pressure_forces()
        self.compute_pressure_forces()
        self.advect()