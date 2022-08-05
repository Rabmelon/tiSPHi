import taichi as ti
from eng.gguishow import *
from eng.particle_system import *
from eng.chooseSolver import *

# ti.init(arch=ti.cpu, debug=True, default_fp=ti.f64, cpu_max_num_threads=1)
ti.init(arch=ti.cuda, packed=True, device_memory_fraction=0.75, default_fp=ti.f64)     # MEMORY max 4G in GUT, 6G in Legion

if __name__ == "__main__":
    print("hallo tiSPHi! This is for water dambreak test!")

    # init particle system paras, world unit is cm (BUT not cm actually! maybe still m)
    screen_to_world_ratio = 800   # exp: world = (150, 100), ratio = 4, screen res = (600, 400)
    rec_world = [0.584, 0.8]   # a rectangle world start from (0, 0) to this pos
    particle_radius = 0.001
    cube_size = [0.146, 0.292]

    mat = 1         # 1 water; 2 soil
    cmodel = 1      # for water, 1 WC; for soil, 1 muI, 2 DP
    TDmethod = 2    # 1 Symp Euler; 2 Leap Frog; 4 RK4
    flag_kernel = 2 # 1 cubic-spline; 2 Wendland C2

    rho = 1000.0
    viscosity = 0.00005
    stiffness = 50000.0
    powcomp = 7.0

    case1 = ParticleSystem(rec_world, particle_radius)
    case1.gen_rangeary_particles()
    case1.add_cube(lower_corner=[0.0, 0.0], cube_size=cube_size, material=mat, density=rho)

    solver = chooseSolver(case1, mat, cmodel, TDmethod, flag_kernel, para1=rho, para2=viscosity, para3=stiffness, para4=powcomp)

    gguishow(case1, solver, rec_world, screen_to_world_ratio, color_title=3,
             kradius=1.5, step_ggui=20, iparticle=-1, save_png=0, pause_init=True, grid_line=0.146)

    # color title: pressure Pa; velocity m/s; density N/m3; d density N/m3/s;
