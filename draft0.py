import taichi as ti
import numpy as np
from eng.particle_system import *

# TODO: make unit testing for basic functions of SPH

ti.init(arch=ti.cpu, debug=True)
# ti.init(arch=ti.cuda, packed=True, device_memory_fraction=0.75)     # MEMORY max 4G in GUT, 6G in Legion

if __name__ == "__main__":
    print("hallo tiSPHi!")

    # init particle system paras, world unit is cm (BUT not cm actually! maybe still m)
    screen_to_world_ratio = 4   # exp: world = (150, 100), ratio = 4, screen res = (600, 400)
    world = (150, 100)
    particle_radius = 2
    cube_size = [20, 40]

    flag_pause = True
    write_to_disk = False

    case1 = ParticleSystem(world, particle_radius)
    case1.add_cube(lower_corner=[case1.padding, case1.padding],
                   cube_size=cube_size,
                   color=0x956333,
                   material=1)
    case1.initialize_particle_system()

    res = (np.array(world) * screen_to_world_ratio).astype(int)
    gui = ti.GUI('SPH window', res=(max(res), max(res)), background_color=0xFFFFFF)
    flag_step = 0
    show_pos = (0.0, 0.0)
    while gui.running:
        if not flag_pause:
            print('----step:', flag_step)
            for i in range(20):
                flag_step += 1

        particle_info = case1.dump()

        # draw particles
        draw_radius = case1.particle_radius * screen_to_world_ratio * 1.25
        gui.circles(particle_info['position'] * screen_to_world_ratio / max(res),
                    radius=draw_radius, color=particle_info['color'])

        # draw world
        res_world = [i * screen_to_world_ratio / max(res) for i in world]
        draw_world = np.array([[0.0, 0.0], [res_world[0], 0.0], res_world, [0.0, res_world[1]]])
        gui.circles(draw_world, radius=2*draw_radius, color=0xFF0000)
        draw_world_end = np.array([[res_world[0], 0.0], res_world, [0.0, res_world[1]], [0.0, 0.0]])
        gui.lines(begin=draw_world, end=draw_world_end, radius=0.5*draw_radius, color=0xFF0000)

        # control
        for e in gui.get_events(gui.PRESS):
            if e.key == gui.ESCAPE:
                gui.running = False
            elif e.key == gui.SPACE:
                flag_pause = not flag_pause
            elif e.key == ti.GUI.LMB:
                show_pos = e.pos

        # show text
        gui.text('Total particle number: {pnum:,}'.format(pnum=case1.particle_num[None]), (0.05, 0.9), font_size=24, color=0x055555)
        gui.text('Step: {step:,}'.format(step=flag_step), (0.05, 0.95), font_size=24, color=0x055555)
        gui.text('Pos: {px:.3f}, {py:.3f}'.format(px=show_pos[0], py=show_pos[1]), (0.05, 0.85), font_size=24, color=0x055555)

        gui.show(f'{flag_step:06d}.png' if write_to_disk else None)
