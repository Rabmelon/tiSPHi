import taichi as ti
from eng.gguishow import *
from eng.particle_system import *
from eng.choose_solver import *

# ti.init(arch=ti.cpu, debug=True, default_fp=ti.f64, cpu_max_num_threads=1)
ti.init(arch=ti.cuda, packed=True, device_memory_fraction=0.3, default_fp=ti.f64)

if __name__ == "__main__":
    print("hallo tiSPHi! This is for sand column collapse test!")

    rec_world = [0.56, 0.2]        # a rectangle world start from (0, 0) to this pos
    screen_to_world_ratio = 800 / max(rec_world)   # exp: world = (150m, 100m), ratio = 4, screen res = (600, 400)
    particle_radius = 0.001
    cube_size = [0.2, 0.1]

    mat = 2         # 1 water; 2 soil
    cmodel = 2      # for water, 1 WC; for soil, 1 muI, 2 DP
    TDmethod = 2    # 1 Symp Euler; 2 Leap Frog; 4 RK4
    flag_kernel = 2 # 1 cubic-spline; 2 Wendland C2

    # rho, coh, fric, E = 2040.0, 0.0, 21.9, 5.84e6     # aluminium rods
    rho, coh, fric, E = 2650.0, 0.0, 22.0, 15.0e6     # granular column
    av_alpha_Pi, av_beta_Pi = 1.0, 0.0

    case1 = ParticleSystem(rec_world, particle_radius)
    case1.gen_boundary_dummy()
    # case1.gen_boundary_rep()
    case1.add_cube(lower_corner=[0.0, 0.0], cube_size=cube_size, material=mat, density=rho)
    solver = choose_solver(case1, mat, cmodel, TDmethod, flag_kernel, para1=rho, para2=coh, para3=fric, para4=E, para5=av_alpha_Pi, para6=av_beta_Pi)

    gguishow(case1, solver, rec_world, screen_to_world_ratio,
             pause_flag=0, stop_step=100001, step_ggui=10, exit_flag=0,
            #  save_png=0,  save_msg=0, iparticle=[2316, 2365, 4840, 7266, 7315], # for cc test
            #  save_png=0,  save_msg=0, iparticle=[1236, 1260, 1285, 2486, 2510, 2535, 3736, 3760, 3785], # for is test
             kradius=1.25, grid_line=0.05, color_title=2,
             given_max=-1, given_min=-1, fix_max=1, fix_min=1)

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
