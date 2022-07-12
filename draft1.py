import sys
import taichi as ti
import numpy as np
from eng.guishow import *
from eng.gguishow import *
from eng.particle_system import *
from eng.wcsph import *
from eng.wcsesph import *
from eng.wclfsph import *
from eng.wcrksph import *
from eng.muIsesph import *
from eng.muIlfsph import *

# TODO: water db here

sys.tracebacklimit = 0
# ti.init(arch=ti.cpu, debug=True)
ti.init(arch=ti.cuda, packed=True, device_memory_fraction=0.75)     # MEMORY max 4G in GUT, 6G in Legion
# ti.init(arch=ti.vulkan)

if __name__ == "__main__":
    print("hallo tiSPHi!")

    # init particle system paras, world unit is cm (BUT not cm actually! maybe still m)
    screen_to_world_ratio = 1200   # exp: world = (150, 100), ratio = 4, screen res = (600, 400)
    rec_world = [0.584, 0.8]   # a rectangle world start from (0, 0) to this pos
    particle_radius = 0.001
    cube_size = [0.146, 0.292]

    mat = 1
    rho = 1000.0
    TDmethod = 4    # 1 Symp Euler; 2 Leap Frog; 4 RK4
    flag_kernel = 2 # 1 cubic-spline; 2 Wendland C2

    case1 = ParticleSystem(rec_world, particle_radius)
    case1.add_cube(lower_corner=[0.0, 0], cube_size=cube_size, material=mat, density=rho)

    if mat == 1:
        viscosity = 0.00005
        stiffness = 50000
        powcomp = 7
        if TDmethod == 1:
            solver = WCSESPHSolver(case1, TDmethod, flag_kernel, viscosity, stiffness, powcomp)
        elif TDmethod == 2:
            solver = WCLFSPHSolver(case1, TDmethod, flag_kernel, viscosity, stiffness, powcomp)
        elif TDmethod == 4:
            solver = WCRKSPHSolver(case1, TDmethod, flag_kernel, viscosity, stiffness, powcomp)
    elif mat == 2:
        coh = 0
        fric = 29
        eta0 = 0
        if TDmethod == 1:
            solver = MCmuISESPHSolver(case1, TDmethod, flag_kernel, rho, coh, fric, eta0)
        elif TDmethod == 2:
            solver = MCmuILFSPHSolver(case1, TDmethod, flag_kernel, rho, coh, fric, eta0)
        elif TDmethod == 4:
            pass

    gguishow(case1, solver, rec_world, screen_to_world_ratio, color_title="velocity m/s",
             kradius=1.5, stepwise=20, iparticle=None, save_png=0, pause=True, grid_line=0.146)

    # color title: pressure Pa; velocity m/s; density N/m3; d density N/m3/s;
