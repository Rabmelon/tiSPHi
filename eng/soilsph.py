import taichi as ti
from .sph_solver import SPHSolver

class SoilSPHSolver(SPHSolver):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        print("Hallo, class soil SPH Solver starts to serve!")

        # Basic paras

        # Allocate memories

    # Evaluate density