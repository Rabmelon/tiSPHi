import taichi as ti
from .sph_solver import SPHSolver

class SandSPHSolver(SPHSolver):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        print("Hallo, class sand SPH Solver starts to serve!")
