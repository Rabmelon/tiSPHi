import taichi as ti
import numpy as np
from eng.particle_system import *
from eng.wcsph import *

# TODO: Change the rule from N-S to DP, material from water to sand.
# TODO: Or add DFSPH solver first? Don't sure if there is enough time remains for a real physical experiment of sand flow.

# ti.init(arch=ti.cpu, debug=True)
ti.init(arch=ti.gpu, packed=True, device_memory_GB=4)

if __name__ == "__main__":
    # init particle system paras, world unit is cm
    screen_to_world_ratio = 6   # exp: world = (150, 100), ratio = 4, screen res = (600, 400)
    world = (150, 100)
    particle_radius = 0.1
    kh = 6.0
    cube_size = [20, 40]

    case1 = ParticleSystem(world, particle_radius, kh)
    case1.add_cube(lower_corner=[case1.padding, case1.padding],
                   cube_size=cube_size,
                   velocity=[.0, .0],
                   density=1000.0,
                   color=0x956333,
                   material=1)
    case1.initialize_particle_system()

    wcsph_solver = WCSPHSolver(case1)

    res = np.array(world) * screen_to_world_ratio
    gui = ti.GUI('SPH window', res=(max(res), max(res)), background_color=0xFFFFFF)
    flag_step = 0
    flag_pause = True
    while gui.running:
        if not flag_pause:
            for i in range(20):
                wcsph_solver.step()
                flag_step += 1

        particle_info = case1.dump()

        # draw particles
        draw_radius = case1.particle_radius * screen_to_world_ratio * 1.0
        gui.circles(particle_info['position'] * screen_to_world_ratio / max(res),
                    radius=draw_radius, color=particle_info['color'])

        # show text
        gui.text('Total particle number: {pnum:,}'.format(pnum=case1.particle_num[None]), (0.05, 0.9), font_size=24, color=0x055555)
        gui.text('Step: {step:,}'.format(step=flag_step), (0.05, 0.95), font_size=24, color=0x055555)

        # control
        for e in gui.get_events(gui.PRESS):
            if e.key == gui.ESCAPE:
                gui.running = False
            elif e.key == gui.SPACE:
                flag_pause = not flag_pause

        gui.show()
