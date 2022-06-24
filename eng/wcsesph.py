import taichi as ti
from .sph_solver import SPHSolver

class WCSESPHSolver(SPHSolver):
    def __init__(self, particle_system, TDmethod, kernel, visco, stiff, expo):
        super().__init__(particle_system, 1, kernel)
        print("Hallo, class WCSPH Solver with SE starts to serve!")

        # Basic paras
        self.density_0 = 1000.0  # reference density
        self.viscosity = visco  # viscosity
        self.mass = self.ps.m_V * self.density_0

        # Two paras in taichiWCSPH code
        self.stiffness = stiff   # k1
        self.exponent = expo     # k2

        self.d_density = ti.field(dtype=float)
        self.pressure = ti.field(dtype=float)
        self.d_velocity = ti.Vector.field(self.ps.dim, dtype=float)
        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.d_density, self.pressure, self.d_velocity)

    @ti.kernel
    def init_value(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] < 10:
                # self.ps.val[p_i] = self.ps.u[p_i].norm()
                # self.ps.val[p_i] = -self.ps.x[p_i][1]
                self.ps.val[p_i] = self.ps.density[p_i]
                # self.ps.val[p_i] = self.pressure[p_i]
                # self.ps.val[p_i] = p_i

    #######################################################
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
                tmp = (self.ps.u[p_i] - self.ps.u[p_j]).transpose() @ (self.ps.L[p_i] @ self.kernel_derivative(x_i - x_j))
                drho += self.ps.density[p_j] * self.ps.m_V * tmp[0]
            self.d_density[p_i] = drho

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
                r = x_i - x_j
                d_v += 2 * (self.ps.dim + 2) * self.viscosity * self.ps.m_V * (self.ps.u[p_i]-self.ps.u[p_j]).dot(r) / (r.norm()**2 + 0.01 * self.ps.smoothing_len**2) * self.kernel_derivative(r)

            # Add body force
            if self.ps.material[p_i] == self.ps.material_fluid:
                d_v += ti.Vector([0.0, self.g] if self.ps.dim == 2 else [0.0, 0.0, self.g])
            self.d_velocity[p_i] = d_v

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
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                d_v += -self.ps.m_V * self.ps.density[p_j] * (self.pressure[p_i] / self.ps.density[p_i]**2 + self.pressure[p_j] / self.ps.density[p_j]**2) * self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
            self.d_velocity[p_i] += d_v

    # Symplectic Euler
    @ti.kernel
    def advect(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.density[p_i] += self.dt[None] * self.d_density[p_i]
                self.ps.u[p_i] += self.dt[None] * self.d_velocity[p_i]
                self.ps.x[p_i] += self.dt[None] * self.ps.u[p_i]

    def substep_SympEuler(self):
        self.compute_d_density()
        self.compute_non_pressure_forces()
        self.compute_pressure_forces()
        self.advect()
