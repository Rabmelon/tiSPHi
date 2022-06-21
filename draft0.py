import taichi as ti
import numpy as np
from eng.guishow import *
from eng.gguishow import *
from eng.particle_system import *
from eng.muIsph import *
from eng.wcsph import *
from eng.wcsesph import *

# TODO: make unit testing for basic functions of SPH

ti.init(arch=ti.cpu, debug=True)
# ti.init(arch=ti.cuda, packed=True, device_memory_fraction=0.75)     # MEMORY max 4G in GUT, 6G in Legion
# ti.init(arch=ti.vulkan)

if __name__ == "__main__":
    print("hallo tiSPHi!")

    # init particle system paras, world unit is cm (BUT not cm actually! maybe still m)
    screen_to_world_ratio = 800   # exp: world = (150, 100), ratio = 4, screen res = (600, 400)
    rec_world = [0.8, 0.8]   # a rectangle world start from (0, 0) to this pos
    particle_radius = 0.002
    cube_size = [0.2, 0.4]

    mat = 1
    rho = 1000.0
    TDmethod = 1    # 1 Symp Euler; 2 RK4
    flag_kernel = 2 # 1 cubic-spline; 2 Wendland C2

    case1 = ParticleSystem(rec_world, particle_radius)
    case1.add_cube(lower_corner=[0.0, 0], cube_size=cube_size, material=mat, density=rho)

    if mat == 1:
        solver = WCSESPHSolver(case1, TDmethod, flag_kernel, 1.1e-6, 50000, 7)
    elif mat == 2:
        solver = MCmuISPHSolver(case1, TDmethod, flag_kernel, rho, 0, 29, 0)

    gguishow(case1, solver, rec_world, screen_to_world_ratio, stepwise=20, iparticle=None, color_title="pressure Pa", kradius=1.25, write_to_disk=0)

    # color title: pressure Pa; velocity m/s; density N/m3; d density N/m3/s;
