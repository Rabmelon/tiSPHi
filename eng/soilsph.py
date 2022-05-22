import taichi as ti
from .sph_solver import SPHSolver

class SoilSPHSolver(SPHSolver):
    def __init__(self, particle_system, TDmethod, density, coh, fric):
        super().__init__(particle_system, TDmethod)
        print("Class SoilSPH Solver starts to serve!")

