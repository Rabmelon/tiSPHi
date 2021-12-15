import taichi as ti
import numpy as np
from .particle_system import ParticleSystem

# TODO: understand the code of wcsph then rewrite by self

ti.require_version(0, 8, 1) # Why use this?????

@ti.data_oriented
class SPHSolver:
    def __init__(self):
        print("Hallo, class SPH Solver starts to serve!")
