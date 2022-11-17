import taichi as ti
from eng.gguishow import *
from eng.particle_system import *
from eng.choose_solver import *

# ti.init(arch=ti.cpu, debug=True, default_fp=ti.f64, cpu_max_num_threads=1)
ti.init(arch=ti.cuda, packed=True, device_memory_fraction=0.75, default_fp=ti.f64)     # MEMORY max 4G in GUT, 6G in Legion

if __name__ == "__main__":
    print("hallo tiSPHi! This is for water dambreak test!")

    # init particle system paras, world unit is cm (BUT not cm actually! maybe still m)
    screen_to_world_ratio = 200   # exp: world = (150, 100), ratio = 4, screen res = (600, 400)
    rec_world = [5.0, 2.0]   # a rectangle world start from (0, 0) to this pos
    particle_radius = 0.01
    cube_size = [1.0, 0.9]

    mat = 1         # 1 water; 2 soil
    cmodel = 1      # for water, 1 WC; for soil, 1 muI, 2 DP
    TDmethod = 1    # 1 Symp Euler; 2 Leap Frog; 4 RK4
    flag_kernel = 2 # 1 cubic-spline; 2 Wendland C2

    rho = 1000.0
    viscosity = 0.01
    stiffness = 50000.0
    powcomp = 7.0

    case1 = ParticleSystem(rec_world, particle_radius)
    case1.gen_boundary_dummy()
    case1.add_cube(lower_corner=[0.0, 0.0], cube_size=cube_size, material=mat, density=rho)

    solver = choose_solver(case1, mat, cmodel, TDmethod, flag_kernel, para1=rho, para2=viscosity, para3=stiffness, para4=powcomp)

    gguishow(case1, solver, rec_world, screen_to_world_ratio, color_title=2,
             kradius=1.05, step_ggui=20, iparticle=-1, save_png=0, pause_flag=True, grid_line=0.1)

    '''
    color title:
    1 index
    2 density
        21 d density
    3 velocity norm
        31 x 32 y 33 z
    4 position
        41 x 42 y 43 z
    5 stress
        51 xx 52 yy 53 zz 54 xy 55 yz 56 zx
        57 hydrostatic stress 58 deviatoric stress
    6 strain
        61 equivalent plastic strain
    7 displacement
    8 pressure
    otherwise Null
    '''
