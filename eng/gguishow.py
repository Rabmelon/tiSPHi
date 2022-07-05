import taichi as ti
import numpy as np
import time
import os
from datetime import datetime

# TODO: add different color choice
# TODO: add background grids
# TODO:

def gguishow(case, solver, world, s2w_ratio, kradius=1.0, pause=True, save_png=False, stepwise=20, iparticle=None, color_title="Null", color_particle=None, grid_line=None):
    print("ggui starts to serve!")

    # basic paras
    drawworld = [i + 2 * case.grid_size for i in world]
    res = (np.array(drawworld) * s2w_ratio).astype(int)
    w2s=s2w_ratio / res.max()
    window = ti.ui.Window('SPH window', res=(max(res), max(res)))
    canvas = window.get_canvas()
    canvas.set_background_color((1,1,1))

    # draw grid line
    if grid_line is not None and grid_line != 0.0:
        dim = len(world)
        num_grid_point = [int((i - 1e-8) // grid_line) for i in world]
        num_all_grid_point = sum(num_grid_point)
        num_all2_grid_point = 2 * num_all_grid_point
        np_pos_line = np.array([[0.0 for _ in range(dim)] for _ in range(num_all2_grid_point)])
        np_indices_line = np.array([[i, i + num_all_grid_point] for i in range(num_all_grid_point)])
        pos_line = ti.Vector.field(dim, float, shape=num_all2_grid_point)
        indices_line = ti.Vector.field(2, int, shape=num_all_grid_point)
        indices_line.from_numpy(np_indices_line)
        for id in range(dim):
            id2 = dim - 1 - id
            for i in range(num_grid_point[id]):
                np_pos_line[i + sum(num_grid_point[0:id])][id] = (i + 1) * grid_line
                np_pos_line[i + sum(num_grid_point[0:id]) + num_all_grid_point][id] = (i + 1) * grid_line
                np_pos_line[i + sum(num_grid_point[0:id]) + num_all_grid_point][id2] = world[id2]
        pos_line.from_numpy((np_pos_line + case.grid_size) * w2s)

    # control paras
    flag_pause = pause
    flag_step = 0
    show_pos = [0.0, 0.0]
    show_grid = [0, 0]
    max_res = int(res.max())

    # save png
    cappath = os.getcwd() + r"\screenshots"
    if save_png:
        timestamp = datetime.today().strftime('%Y_%m_%d_%H%M%S')
        simpath = os.getcwd() + "\\sim_" + timestamp
        if not os.path.exists(simpath):
            os.mkdir(simpath)
        os.chdir(simpath)

    # main loop
    while window.running:
        if not flag_pause:
            if iparticle is None:
                print('---- step %d' % (flag_step))
            else:
                print('---- step %d, p[%d]: x=(%.3f, %.3f), u=(%.3f, %.3f), rho=%.3f, neighbour=%d' % (flag_step, iparticle, case.x[iparticle][0], case.x[iparticle][1], case.u[iparticle][0], case.u[iparticle][1], case.density[iparticle], case.particle_neighbors_num[iparticle]))
            for i in range(stepwise):
                solver.step()
                flag_step += 1

        # draw world

        # draw grids
        if grid_line is not None and grid_line != 0.0:
            canvas.lines(pos_line, 0.0025, indices_line, (0.8, 0.8, 0.8))

        # draw particles
        case.copy2vis(s2w_ratio, max_res)
        solver.init_value()
        case.v_maxmin()
        case.set_color()
        draw_radius = case.particle_radius * s2w_ratio * kradius / max_res
        canvas.circles(case.pos2vis, radius=draw_radius, per_vertex_color=case.color)

        # show text
        window.GUI.begin("Info", 0.03, 0.03, 0.4, 0.3)
        window.GUI.text('Total particle number: {pnum:,}'.format(pnum=case.particle_num[None]))
        window.GUI.text('Step: {step:,}'.format(step=flag_step))
        window.GUI.text('Time: {t:.6f}s'.format(t=solver.dt[None] * flag_step))
        window.GUI.text('Pos: {px:.3f}, {py:.3f}'.format(px=show_pos[0], py=show_pos[1]))
        window.GUI.text('Grid: {gx:.1f}, {gy:.1f}'.format(gx=show_grid[0], gy=show_grid[1]))
        window.GUI.text('colorbar: {str}'.format(str=color_title))
        window.GUI.text('max value: {maxv:.3f}'.format(maxv=case.vmax[None]))
        window.GUI.text('min value: {minv:.3f}'.format(minv=case.vmin[None]))
        window.GUI.end()

        # control
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
            elif e.key == ti.ui.SPACE:
                flag_pause = not flag_pause
            elif e.key == ti.ui.LMB:
                show_pos = [i / s2w_ratio * max_res - case.grid_size for i in window.get_cursor_pos()]
                show_grid = [(i - j) // case.grid_size for i,j in zip(show_pos, case.bound[0])]
        if window.is_pressed('p'):
            timestamp = datetime.today().strftime('%Y_%m_%d_%H%M%S')
            fname = os.path.join(cappath, f"screenshot{timestamp}.jpg")
            window.write_image(fname)
            print(f"Screenshot has been saved to {fname}")

        if save_png:
            window.write_image(f"{flag_step:06d}.png")

        window.show()
