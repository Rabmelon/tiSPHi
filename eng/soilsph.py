import taichi as ti
import numpy as np
from .sph_solver import SPHSolver

class SoilSPHSolver(SPHSolver):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        print("Hallo, class soil SPH Solver starts to serve!")

        # Basic paras
        self.density_0 = 1850.0     # reference density of soil, kg/m3
        self.cohesion = 32720       # the material cohesion, Pa
        self.friction_deg = 36      # the angle of internal friction, DEG
        self.poisson = 0.3          # Poisson’s ratio
        self.E = 8e7                # Young’s modulus, Pa

        # Paras based on basic paras
        self.mass = self.ps.m_V * self.density_0            # the mass of each particle, kg
        self.friction = self.friction_deg / 180 * np.pi     # the angle of internal friction, RAD
        self.Depq = self.E / (1 + self.poisson) / (1 - self.poisson) * (1)
        self.alpha_fric = 1
        self.kc = 1

        # Allocate memories
        self.f_stress = ti.Matrix.field(self.ps.dim, self.ps.dim, dtype=float)
        self.f_strain_p = ti.Matrix.field(self.ps.dim_stress, self.ps.dim, dtype=float)
        self.f_ext = ti.Vector.field(self.ps.dim, dtype=float)
        self.g_p = ti.Vector.field(self.ps.dim_stress, dtype=float)
        self.g_DP = ti.field(dtype=float)
        self.spin = ti.field(dtype=float)
        self.Jaumann = ti.Vector.field(self.ps.dim_stress, dtype=float)
        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.f_stress, self.f_strain_p, self.f_ext, self.g_p, self.g_DP, self.spin, self.Jaumann)

    # Assign density
    @ti.kernel
    def compute_densities(self):
        for p_i in range(self.ps.particle_num[None]):
            self.ps.density[p_i] = self.density_0

    # Calculate term fσ and fu

    # Check stress state

    # Update boundary particles

    # Calculate gradients of fσ and fu

    # Assign external forces

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