import taichi as ti
import numpy as np
import time
import os
from datetime import datetime

# TODO: add different color choice
# TODO: add background grids
# TODO:

def gguishow(case, solver, world, s2w_ratio, kradius=1.0, pause=True, save_png=False, stepwise=20, iparticle=None, color_title="Null", grid_line=None, given_max=-1):
    print("ggui starts to serve!")

    # basic paras
    drawworld = [i + 2 * case.grid_size for i in world]
    res = (np.array(drawworld) * s2w_ratio).astype(int)
    w2s=s2w_ratio / res.max()
    window = ti.ui.Window('SPH window', res=(max(res), max(res)))
    canvas = window.get_canvas()
    canvas.set_background_color((1,1,1))
    i_pos = ti.Vector.field(case.dim, ti.f32, shape=1)

    # draw grid line
    if grid_line is not None and grid_line != 0.0:
        dim = len(world)
        if not isinstance(grid_line,list):
            grid_line = [grid_line for _ in range(dim)]
        num_grid_point = [int((world[i] - 1e-8) // grid_line[i]) for i in range(dim)]
        num_all_grid_point = sum(num_grid_point)
        num_all2_grid_point = 2 * num_all_grid_point
        np_pos_line = np.array([[0.0 for _ in range(dim)] for _ in range(num_all2_grid_point)], dtype=np.float32)
        np_indices_line = np.array([[i, i + num_all_grid_point] for i in range(num_all_grid_point)], dtype=np.int32)
        pos_line = ti.Vector.field(dim, ti.f32, shape=num_all2_grid_point)
        indices_line = ti.Vector.field(2, ti.i32, shape=num_all_grid_point)
        indices_line.from_numpy(np_indices_line)
        for id in range(dim):
            id2 = dim - 1 - id
            for i in range(num_grid_point[id]):
                np_pos_line[i + sum(num_grid_point[0:id])][id] = (i + 1) * grid_line[id]
                np_pos_line[i + sum(num_grid_point[0:id]) + num_all_grid_point][id] = (i + 1) * grid_line[id]
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
                print('---- %06d' % (flag_step))
            else:
                print('---- %06d, p[%d]: x=(%.6f, %.6f), v=(%.6f, %.6f), ??=%.3f' % (flag_step, iparticle, solver.ps.x[iparticle][0], solver.ps.x[iparticle][1], solver.ps.v[iparticle][0], solver.ps.v[iparticle][1], solver.ps.density[iparticle]))
            for i in range(stepwise):
                solver.step()
                flag_step += 1

        # draw world

        # draw grids
        if grid_line is not None and grid_line != 0.0:
            canvas.lines(pos_line, 0.0025, indices_line, (0.8, 0.8, 0.8))   # ! WARRNING: Overriding last binding

        # draw particles
        solver.ps.copy2vis(s2w_ratio, max_res)
        solver.init_value()
        solver.ps.v_maxmin(given_max)
        solver.ps.set_color()
        draw_radius = solver.ps.particle_radius * s2w_ratio * kradius / max_res
        canvas.circles(solver.ps.pos2vis, radius=draw_radius, per_vertex_color=solver.ps.color)   # ! WARRNING: Overriding last binding
        if iparticle is not None:
            i_pos.from_numpy(np.array([solver.ps.pos2vis[iparticle]], dtype=np.float32))
            canvas.circles(i_pos, radius=1.5*draw_radius, color=(1.0, 0.0, 0.0))   # ! WARRNING: Overriding last binding

        # show text
        window.GUI.begin("Info", 0.03, 0.03, 0.4, 0.3)
        window.GUI.text('Total particle number: {pnum:,}'.format(pnum=solver.ps.particle_num[None]))
        window.GUI.text('Step: {step:,}'.format(step=flag_step))
        window.GUI.text('Time: {t:.6f}s'.format(t=solver.dt[None] * flag_step))
        window.GUI.text('Pos: {px:.3f}, {py:.3f}'.format(px=show_pos[0], py=show_pos[1]))
        window.GUI.text('Grid: {gx:.1f}, {gy:.1f}'.format(gx=show_grid[0], gy=show_grid[1]))
        window.GUI.text('colorbar: {str}'.format(str=color_title))
        window.GUI.text('max value: {maxv:.3f}'.format(maxv=solver.ps.vmax[None]))
        window.GUI.text('min value: {minv:.3f}'.format(minv=solver.ps.vmin[None]))
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
