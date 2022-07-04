import taichi as ti
from .sph_solver import SPHSolver

@ti.data_oriented
class WCLFSPHSolver(SPHSolver):
    def __init__(self, particle_system, TDmethod, kernel, visco, stiff, expo):
        super().__init__(particle_system, TDmethod, kernel)
        print("Hallo, class WCSPH Solver starts to serve!")

        # Basic paras
        self.density_0 = 1000.0  # reference density
        self.mass = self.ps.m_V * self.density_0
        self.viscosity = visco  # viscosity

        self.density2 = ti.field(dtype=float)
        self.u2 = ti.Vector.field(self.ps.dim, dtype=float)
        self.pressure = ti.field(dtype=float)
        self.d_velocity = ti.Vector.field(self.ps.dim, dtype=float)
        self.d_density = ti.field(dtype=float)
        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.density2, self.u2, self.pressure, self.d_density, self.d_velocity)

        # Two paras in taichiWCSPH code
        self.stiffness = stiff   # k1
        self.exponent = expo     # k2

    @ti.kernel
    def init_value(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] < 10:
                # self.ps.val[p_i] = self.ps.u[p_i].norm()
                # self.ps.val[p_i] = -self.ps.x[p_i][1]
                self.ps.val[p_i] = self.ps.density[p_i]
                # self.ps.val[p_i] = self.pressure[p_i]
                # self.ps.val[p_i] = p_i

    @ti.kernel
    def init_LF_f(self):
        for p_i in range(self.ps.particle_num[None]):
            self.density2[p_i] = self.ps.density[p_i]
            self.u2[p_i] = self.ps.u[p_i]

    @ti.func
    def update_boundary_particles(self, p_i, p_j):
        self.density2[p_j] = self.density_0
        self.u2[p_j] = (1.0 - min(1.5, 1.0 + self.cal_d_BA(p_i, p_j))) * self.u2[p_i]
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
                tmp = (self.u2[p_i] - self.u2[p_j]).transpose() @ self.kernel_derivative(x_i - x_j)
                # tmp = (self.u2[p_i] - self.u2[p_j]).transpose() @ (self.ps.L[p_i] @ self.kernel_derivative(x_i - x_j))
                drho += self.density2[p_j] * self.ps.m_V * tmp[0]
            self.d_density[p_i] = drho

    @ti.kernel
    def compute_densities(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            x_i = self.ps.x[p_i]
            self.density2[p_i] = 0.0
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                if self.ps.material[p_j] == self.ps.material_dummy:
                    self.update_boundary_particles(p_i, p_j)
                self.density2[p_i] += self.ps.m_V * self.kernel(x_i - x_j)
            self.density2[p_i] *= self.density_0
            self.density2[p_i] = ti.max(self.density2[p_i], self.density_0)

    # Compute the viscosity force contribution, Anti-symmetric formula
    @ti.func
    def viscosity_force(self, p_i, p_j):
        r = self.ps.x[p_i] - self.ps.x[p_j]
        v_xy = (self.u2[p_i] - self.u2[p_j]).dot(r)
        res = 2 * (self.ps.dim + 2) * self.viscosity * (self.mass / (self.density2[p_j])) * v_xy / (r.norm()**2 + 0.01 * self.ps.smoothing_len**2) * self.kernel_derivative(r)
        # res = 2 * (self.ps.dim + 2) * self.viscosity * (self.mass / (self.density2[p_j])) * v_xy / (r.norm()**2 + 0.01 * self.ps.smoothing_len**2) * (self.ps.L[p_i] @ self.kernel_derivative(r))
        return res

    # Evaluate viscosity and add gravity
    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                if self.ps.material[p_j] == self.ps.material_dummy:
                    self.update_boundary_particles(p_i, p_j)
                d_v += self.viscosity_force(p_i, p_j)

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
    def pressure_force(self, p_i, p_j):
        res = -self.density2[p_j] * self.ps.m_V * (self.pressure[p_i] / self.density2[p_i]**2 + self.pressure[p_j] / self.density2[p_j]**2) * self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
        return res

    # Evaluate pressure force
    @ti.kernel
    def compute_pressure_forces(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            self.pressure[p_i] = ti.max(self.stiffness * (ti.pow(self.density2[p_i] / self.density_0, self.exponent) - 1.0), 0.0)
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            d_v = ti.Vector([0.0 for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                if self.ps.material[p_j] == self.ps.material_dummy:
                    self.update_boundary_particles(p_i, p_j)
                d_v += self.pressure_force(p_i, p_j)
            self.d_velocity[p_i] += d_v

    @ti.kernel
    def chk_density(self):
        for p_i in range(self.ps.particle_num[None]):
            self.ps.density[p_i] = ti.max(self.density_0, self.ps.density[p_i])
            if self.ps.density[p_i] > self.density_0 * 1.25:
                print("stop because particle", p_i, "has a density", self.ps.density[p_i], "and pressure", self.pressure[p_i], "with neighbour num", self.ps.particle_neighbors_num[p_i])
            assert self.ps.density[p_i] < self.density_0 * 1.25

    @ti.kernel
    def advect_LF_half(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.density2[p_i] += self.d_density[p_i] * self.dt[None] * 0.5
                self.u2[p_i] += self.d_velocity[p_i] * self.dt[None] * 0.5

    @ti.kernel
    def advect_LF(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.density[p_i] += self.d_density[p_i] * self.dt[None]
                self.ps.u[p_i] += self.d_velocity[p_i] * self.dt[None]
                self.ps.x[p_i] += self.ps.u[p_i] * self.dt[None]

    def LF_one_step(self):
        self.compute_d_density()
        self.compute_non_pressure_forces()
        self.compute_pressure_forces()

    def substep_LeapFrog(self):
        self.init_LF_f()
        self.LF_one_step()
        self.advect_LF_half()
        self.LF_one_step()
        self.advect_LF()
        self.chk_density()