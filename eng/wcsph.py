import taichi as ti
from .sph_solver import SPHSolver

class WCSPHSolver(SPHSolver):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        print("Hallo, class WCSPH Solver starts to serve!")

        # Two paras in taichiWCSPH code
        self.stiffness = 50000.0   # k1
        self.exponent = 7.0     # k2

        self.viscosity = 0.05  # viscosity
        self.d_velocity = ti.Vector.field(self.ps.dim, dtype=float)
        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.d_velocity)


    # Compute the viscosity force contribution, Anti-symmetric formula
    @ti.func
    def viscosity_force(self, p_i, p_j, r):
        v_xy = (self.ps.v[p_i] - self.ps.v[p_j]).dot(r)
        res = 2 * (self.ps.dim + 2) * self.viscosity * (
            self.mass / (self.ps.density[p_j])) * v_xy / (
                r.norm()**2 + 0.01 *
                self.ps.support_radius**2) * self.cubic_kernel_derivative(r)
        return res

    # Compute the pressure force contribution, Symmetric formula
    @ti.func
    def pressure_force(self, p_i, p_j, r):
        res = -self.mass * (self.ps.pressure[p_i] / self.ps.density[p_i]**2 +
                            self.ps.pressure[p_j] / self.ps.density[p_j]**2
                            ) * self.cubic_kernel_derivative(r)
        return res

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