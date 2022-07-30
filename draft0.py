import sys
import taichi as ti
import numpy as np
from eng.gguishow import *
from eng.particle_system import *
from eng.wcsph import *
from eng.wcsesph import *
from eng.muIsesph import *
from eng.muIlfsph import *
from eng.muIrksph import *
from eng.dpsesph import *
from eng.dplfsph import *

# TODO: sand cc here

sys.tracebacklimit = 0
ti.init(arch=ti.cpu, debug=True, default_fp=ti.f64)
# ti.init(arch=ti.cuda, packed=True, device_memory_fraction=0.75, default_fp=ti.f64)     # MEMORY max 4G in GUT, 6G in Legion
# ti.init(arch=ti.vulkan)

if __name__ == "__main__":
    print("hallo tiSPHi!")

    # init particle system paras, world unit is cm (BUT not cm actually! maybe still m)
    screen_to_world_ratio = 1400   # exp: world = (150, 100), ratio = 4, screen res = (600, 400)
    rec_world = [0.55, 0.20]   # a rectangle world start from (0, 0) to this pos
    particle_radius = 0.001
    cube_size = [0.2, 0.1]

    mat = 2
    rho = 2040.0
    cmodel = 2      # for water, 1 WC; for soil, 1 muI, 2 DP
    TDmethod = 1    # 1 Symp Euler; 2 Leap Frog; 4 RK4
    flag_kernel = 2 # 1 cubic-spline; 2 Wendland C2

    case1 = ParticleSystem(rec_world, particle_radius)
    case1.add_cube(lower_corner=[0.0, 0.0], cube_size=cube_size, material=mat, density=rho)

    if mat == 1 and cmodel == 1:
        viscosity = 0.00005
        stiffness = 50000.0
        powcomp = 7.0
        if TDmethod == 1:
            solver = WCSESPHSolver(case1, TDmethod, flag_kernel, viscosity, stiffness, powcomp)
        elif TDmethod == 2:
            solver = WCLFSPHSolver(case1, TDmethod, flag_kernel, viscosity, stiffness, powcomp)
        elif TDmethod == 4:
            solver = WCSPHSolver(case1, TDmethod, flag_kernel, viscosity, stiffness, powcomp)
    elif mat == 2 and cmodel == 1:
        coh = 0.0
        fric = 21.9
        eta0 = 0.0
        if TDmethod == 1:
            solver = MCmuISESPHSolver(case1, TDmethod, flag_kernel, rho, coh, fric, eta0)
        elif TDmethod == 2:
            solver = MCmuILFSPHSolver(case1, TDmethod, flag_kernel, rho, coh, fric, eta0)
        elif TDmethod == 4:
            solver = MCmuIRKSPHSolver(case1, TDmethod, flag_kernel, rho, coh, fric, eta0)
    elif mat == 2 and cmodel == 2:
        coh = 0.0
        fric = 21.9
        E = 5.84e6
        if TDmethod == 1:
            solver = DPSESPHSolver(case1, TDmethod, flag_kernel, rho, coh, fric, E)
        elif TDmethod == 2:
            solver = DPLFSPHSolver(case1, TDmethod, flag_kernel, rho, coh, fric, E)
        elif TDmethod == 4:
            pass

    gguishow(case1, solver, rec_world, screen_to_world_ratio, color_title="stress yy Pa",
             kradius=1.5, stepwise=1, iparticle=2385, save_png=0, pause=0, grid_line=0.1, givenmax=-1)

    # color title: pressure Pa; velocity m/s; density N/m3; d density N/m3/s; stress yy Pa; index; displacement m
