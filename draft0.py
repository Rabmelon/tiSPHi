import taichi as ti
import numpy as np
from eng.guishow import *
from eng.gguishow import *
from eng.particle_system import *

# TODO: make unit testing for basic functions of SPH

ti.init(arch=ti.cpu, debug=True)
# ti.init(arch=ti.cuda, packed=True, device_memory_fraction=0.75)     # MEMORY max 4G in GUT, 6G in Legion

if __name__ == "__main__":
    print("hallo tiSPHi!")

    # init particle system paras, world unit is cm (BUT not cm actually! maybe still m)
    screen_to_world_ratio = 4   # exp: world = (150, 100), ratio = 4, screen res = (600, 400)
    world = [120, 100]
    particle_radius = 2
    cube_size = [20, 40]

    TDmethod = 1    # 1 Symp Euler; 2 RK4
    write_to_disk = False

    case1 = ParticleSystem(world, particle_radius)
    case1.add_cube(lower_corner=[case1.padding, case1.padding],
                   cube_size=cube_size,
                   color=0x956333,
                   material=1)

    case1.initialize_particle_system()
    # wcsph_solver = WCSPHSolver(case1, TDmethod)

    guishow(case1, world, screen_to_world_ratio, write_to_disk)
