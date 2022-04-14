import taichi as ti
import numpy as np

def guishow(radius, world, screen_to_world_ratio, write_to_disk):
    print("Hallo, gui starts to serve!")

    grid_size = 6 * radius
    drawworld = [i + 2 * grid_size for i in world]
    cor_dl = [grid_size, grid_size]
    res = (np.array(drawworld) * screen_to_world_ratio).astype(int)
    gui = ti.GUI('window', res=(max(res), max(res)), background_color=0xFFFFFF)

    flag_pause = True
    flag_step = 0
    show_pos = [0.0, 0.0]

    while gui.running:
        # draw world
        draw_radius = radius * screen_to_world_ratio * 1.25
        res_world = [(i + grid_size) * screen_to_world_ratio / max(res) for i in world]
        res_cor_ld = [i * screen_to_world_ratio / max(res) for i in cor_dl]
        draw_world = np.array([res_cor_ld, [res_world[0], res_cor_ld[1]], res_world, [res_cor_ld[0], res_world[1]]])
        gui.circles(draw_world, radius=1.0*draw_radius, color=0xFF0000)
        draw_world_end = np.array([[res_world[0], res_cor_ld[1]], res_world, [res_cor_ld[0], res_world[1]], res_cor_ld])
        gui.lines(begin=draw_world, end=draw_world_end, radius=0.25*draw_radius, color=0xFF0000)

        # control
        for e in gui.get_events(gui.PRESS):
            if e.key == gui.ESCAPE:
                gui.running = False
            elif e.key == gui.SPACE:
                flag_pause = not flag_pause
            elif e.key == ti.GUI.LMB:
                show_pos = [i / screen_to_world_ratio * max(res) - grid_size for i in e.pos]

        # show text
        gui.text('Pos: {px:.3f}, {py:.3f}'.format(px=show_pos[0], py=show_pos[1]), (0.05, 0.85), font_size=24, color=0x055555)

        gui.show(f'{flag_step:06d}.png' if write_to_disk else None)

