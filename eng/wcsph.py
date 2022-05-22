import taichi as ti
from .sph_solver import SPHSolver

class WCSPHSolver(SPHSolver):
    def __init__(self, particle_system, TDmethod):
        super().__init__(particle_system, TDmethod)
        print("Hallo, class WCSPH Solver starts to serve!")
