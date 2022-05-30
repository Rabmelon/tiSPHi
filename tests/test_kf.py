import taichi as ti
import numpy as np
from eng.particle_system import *
from eng.sph_solver import *

ti.init(arch=ti.cpu)

class FunSPHSolver(SPHSolver):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        print("Hallo, class function SPH Solver starts to serve!")


if __name__ == "__main__":
    print("hallo test kernel function accuracy!")

    screen_to_world_ratio = 5   # exp: world = (150, 100), ratio = 4, screen res = (600, 400)
    rec_world = [120, 80]   # a rectangle world start from (0, 0) to this pos
    particle_radius = 1
    cube_size = [80, 80]
    case1 = ParticleSystem(rec_world, particle_radius)
    case1.add_cube(lower_corner=[0, 0],
                   cube_size=cube_size,
                   color=(149/255,99/255,51/255),
                   material=1)



