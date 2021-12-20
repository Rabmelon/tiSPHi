import taichi as ti
import numpy as np
from eng.particle_system import *
from eng.wcsph import *

# TODO: Change the boundary conditions

# ti.init(arch=ti.cpu, debug=True)
ti.init(arch=ti.gpu, packed=True, device_memory_GB=4)

if __name__ == "__main__":
    # init particle system paras, world unit is cm
    world = (150, 100)
    particle_radius = 0.25
    kh = 4.0
    case1 = ParticleSystem(world, particle_radius, kh)
    print('padding =', case1.padding)

    case1.add_cube(lower_corner=[case1.padding, case1.padding],
                   cube_size=[20, 40],
                   velocity=[.0, .0],
                   density=1000.0,
                   color=0x956333,
                   material=1)

    case1.initialize_particle_system()

    wcsph_solver = WCSPHSolver(case1)

    screen_to_world_ratio = 5   # exp: world = (150, 100), ratio = 4, screen res = (600, 400)
    res = np.array(world) * screen_to_world_ratio
    gui = ti.GUI('SPH window', res=(max(res), max(res)), background_color=0xFFFFFF)
    flag_step = 0
    flag_pause = False
    while gui.running:
        if not flag_pause:
            for i in range(10):
                wcsph_solver.step()
                flag_step += 1

        particle_info = case1.dump()

        # draw particles
        draw_radius = case1.particle_radius * screen_to_world_ratio
        gui.circles(particle_info['position'] * screen_to_world_ratio / max(res),
                    radius=draw_radius, color=0x956333)

        # draw init cube
        cube_dl = (np.zeros(2) + case1.padding) * screen_to_world_ratio / max(res)
        cube_tr = (np.zeros(2) + case1.padding + [20, 40]) * screen_to_world_ratio / max(res)
        gui.line(cube_dl, [cube_tr[0], cube_dl[1]], radius=draw_radius, color=0x44cccc)
        gui.line([cube_tr[0], cube_dl[1]], cube_tr, radius=draw_radius, color=0x44cccc)
        gui.line(cube_tr, [cube_dl[0], cube_tr[1]], radius=draw_radius, color=0x44cccc)
        gui.line([cube_dl[0], cube_tr[1]], cube_dl, radius=draw_radius, color=0x44cccc)

        # draw boundary
        corner_dl = (np.zeros(2) + case1.padding) * screen_to_world_ratio / max(res)
        corner_tr = (case1.bound - case1.padding) * screen_to_world_ratio / max(res)
        gui.line(corner_dl, [corner_tr[0], corner_dl[1]], radius=draw_radius, color=0xff0000)
        gui.line([corner_tr[0], corner_dl[1]], corner_tr, radius=draw_radius, color=0xff0000)
        gui.line(corner_tr, [corner_dl[0], corner_tr[1]], radius=draw_radius, color=0xff0000)
        gui.line([corner_dl[0], corner_tr[1]], corner_dl, radius=draw_radius, color=0xff0000)

        # show step
        if flag_step % 100 == 0: print('step:', flag_step)
        gui.text('Step: {step:,}'.format(step=flag_step), (0.05, 0.95), font_size=25, color=0x055555)
        # gui.text('Step: {time: .5f} s'.format(time=flag_step * wcsph_solver.dt[None]), (0.05, 0.95), font_size=25, color=0x055555)

        # control
        for e in gui.get_events(gui.PRESS):
            if e.key == gui.ESCAPE:
                gui.running = False
            elif e.key == gui.SPACE:
                flag_pause = not flag_pause

        gui.show()
