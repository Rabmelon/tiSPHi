import taichi as ti
import numpy as np

# TODO: find how ggui works in taichi example codes, then change my SPH show to ggui!
# TODO: clarify the drawing coordinate system

def gguishow(case, solver, world, s2w_ratio, write_to_disk):
    print("ggui starts to serve!")

    drawworld = [i + 2 * case.grid_size for i in world]
    cor_dl = [case.grid_size, case.grid_size]
    res = (np.array(drawworld) * s2w_ratio).astype(int)
    window = ti.ui.Window('SPH window', res=(max(res), max(res)))
    canvas = window.get_canvas()

    flag_pause = True
    flag_step = 0
    show_pos = [0.0, 0.0]
    show_grid = [0, 0]

    while window.running:
        if not flag_pause:
            print('----step:', flag_step)
            for i in range(20):
                solver.step()
                flag_step += 1

        particle_info = case.dump()

        # draw particles
        draw_radius = case.particle_radius * s2w_ratio * 1.25
        canvas.circles(case.x, radius=draw_radius, color=particle_info['color'])

        # draw world


        # control
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
            elif e.key == ti.ui.SPACE:
                flag_pause = not flag_pause
            elif e.key == ti.ui.LMB:
                show_pos = [i / s2w_ratio * max(res) - case.grid_size for i in window.get_cursor_pos()]
                show_grid = [(i - j) // case.grid_size for i,j in zip(show_pos, case.bound[0])]

        # show text
        window.GUI.begin("Info", 0.03, 0.03, 0.3, 0.2)
        window.GUI.text('Total particle number: {pnum:,}'.format(pnum=case.particle_num[None]))
        window.GUI.text('Step: {step:,}'.format(step=flag_step))
        window.GUI.text('Pos: {px:.3f}, {py:.3f}'.format(px=show_pos[0], py=show_pos[1]))
        window.GUI.text('Grid: {gx:.1f}, {gy:.1f}'.format(gx=show_grid[0], gy=show_grid[1]))
        window.GUI.end()

        window.show()
