import taichi as ti
import numpy as np

def show(world, screen_to_world_ratio, write_to_disk):
    res = (np.array(world) * screen_to_world_ratio).astype(int)
    gui = ti.GUI('SPH window', res=(max(res), max(res)), background_color=0xFFFFFF)

    flag_pause = True
    flag_step = 0
    show_pos = (0.0, 0.0)

    while gui.running:
        if not flag_pause:
            print('----step:', flag_step)
            for i in range(20):
                flag_step += 1


        # draw particles
        draw_radius = 2 * screen_to_world_ratio * 1.25

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
        gui.text('Pos: {px:.3f}, {py:.3f}'.format(px=show_pos[0], py=show_pos[1]), (0.05, 0.85), font_size=24, color=0x055555)

        gui.show(f'{flag_step:06d}.png' if write_to_disk else None)
