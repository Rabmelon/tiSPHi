import taichi as ti
import numpy as np
from eng.particle_system import *
from eng.wcsph import *

# TODO: Change the boundary conditions

# ti.init(arch=ti.cpu, debug=True)
ti.init(arch=ti.gpu, packed=True, device_memory_GB=4)

if __name__ == "__main__":
    # init particle system paras
    res = (600, 400)
    screen_to_world_ratio = 40
    particle_radius = 0.05
    kh = 4.0
    case1 = ParticleSystem(res, screen_to_world_ratio, particle_radius, kh)
    print('padding =', case1.padding)

    case1.add_cube(lower_corner=[0.45, 0.45],
                   cube_size=[2.0, 4.0],
                   velocity=[.0, .0],
                   density=1000.0,
                   color=0x956333,
                   material=1)

    case1.initialize_particle_system()

    wcsph_solver = WCSPHSolver(case1)

    gui = ti.GUI('SPH window', res=(max(res), max(res)), background_color=0xFFFFFF)
    while gui.running:
        for i in range(10):
            wcsph_solver.step()
        particle_info = case1.dump()

        # draw
        gui.circles(particle_info['position'] * case1.screen_to_world_ratio / max(res),
                    radius=case1.particle_radius * case1.screen_to_world_ratio, color=0x956333)

        corner_dl = (np.zeros(2) + case1.padding - particle_radius) * case1.screen_to_world_ratio / max(res)
        corner_tr = (case1.bound - case1.padding + particle_radius) * case1.screen_to_world_ratio / max(res)
        gui.line(corner_dl, [corner_tr[0], corner_dl[1]], radius=0.5*particle_radius*case1.screen_to_world_ratio, color=0xff0000)
        gui.line([corner_tr[0], corner_dl[1]], corner_tr, radius=0.5*particle_radius*case1.screen_to_world_ratio, color=0xff0000)
        gui.line(corner_tr, [corner_dl[0], corner_tr[1]], radius=0.5*particle_radius*case1.screen_to_world_ratio, color=0xff0000)
        gui.line([corner_dl[0], corner_tr[1]], corner_dl, radius=0.5*particle_radius*case1.screen_to_world_ratio, color=0xff0000)

        gui.show()
