import taichi as ti
import numpy as np

def gguishow(case, solver, world, s2w_ratio, write_to_disk=False, stepwise=20):
    print("ggui starts to serve!")

    drawworld = [i + 2 * case.grid_size for i in world]
    res = (np.array(drawworld) * s2w_ratio).astype(int)
    window = ti.ui.Window('SPH window', res=(max(res), max(res)))
    canvas = window.get_canvas()
    canvas.set_background_color((1,1,1))

    flag_pause = True
    flag_step = 0
    show_pos = [0.0, 0.0]
    show_grid = [0, 0]

    iparticle = 656

    while window.running:
        if not flag_pause:
            print('---- step %d, x[%d] = (%.3f, %.3f)' % (flag_step, iparticle, case.x[iparticle][0], case.x[iparticle][1]))
            # print('---- step %d' % (flag_step))
            for i in range(stepwise):
                solver.step()
                flag_step += 1

        # draw particles
        case.copy2vis(5.0, 720.0)
        solver.init_value()
        case.v_maxmin()
        case.set_color()
        draw_radius = case.particle_radius * s2w_ratio * 1.25 / max(res)
        canvas.circles(case.pos2vis, radius=draw_radius, per_vertex_color=case.color)

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
        window.GUI.begin("Info", 0.03, 0.03, 0.4, 0.25)
        window.GUI.text('Total particle number: {pnum:,}'.format(pnum=case.particle_num[None]))
        window.GUI.text('Step: {step:,}'.format(step=flag_step))
        window.GUI.text('Pos: {px:.3f}, {py:.3f}'.format(px=show_pos[0], py=show_pos[1]))
        window.GUI.text('Grid: {gx:.1f}, {gy:.1f}'.format(gx=show_grid[0], gy=show_grid[1]))
        window.GUI.text('max value: {maxv:.3f}'.format(maxv=case.vmax[None]))
        window.GUI.text('min value: {minv:.3f}'.format(minv=case.vmin[None]))
        window.GUI.end()

        window.show()
        # window.show(f'{flag_step:06d}.png' if write_to_disk else None)
