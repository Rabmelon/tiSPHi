import taichi as ti
import numpy as np
from eng.particle_system import *
from eng.sph_solver import *

ti.init(arch=ti.cpu)

if __name__ == "__main__":
    print("hallo test kernel function accuracy!")

    rec_world = [120, 80]   # a rectangle world start from (0, 0) to this pos
    particle_radius = 1
    case1 = ParticleSystem(rec_world, particle_radius)
    case1.add_cube(lower_corner=[0, 0], cube_size=[80, 80])



