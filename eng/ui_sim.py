import taichi as ti
import numpy as np
import os
import csv
from datetime import datetime
from eng.simulation import Simulation, SimConfiger


# TODO: need a faster exporter

def ui_sim(config: SimConfiger, case: Simulation):
    # Paras
    cfg = config
    substeps = cfg.get_cfg("stepsPerRenderUpdate")
    stop_at_step = cfg.get_cfg("stopAtStep")
    exit_at_step = cfg.get_cfg("exitAtStep")
    stop_at_time = cfg.get_cfg("stopAtTime")
    exit_at_time = cfg.get_cfg("exitAtTime")
    stop_every_step = cfg.get_cfg("stopEveryStep")
    pause_flag = cfg.get_cfg("pauseAtStart")
    kradius = cfg.get_cfg("kradius")
    given_max = cfg.get_cfg("givenMax")
    given_min = cfg.get_cfg("givenMin")
    fix_max = cfg.get_cfg("fixMax")
    fix_min = cfg.get_cfg("fixMin")
    show_pt_info = cfg.get_cfg("showParticleInfo")
    save_png = cfg.get_cfg("exportFrame")
    save_csv = cfg.get_cfg("exportCSV")

    res = (1024, 768)
    w2s_ratio = 1.0 / max(case.ps.domain_size)

    # Basic settings
    time_stamp0 = get_time_stamp()
    window = ti.ui.Window('tiSPHi simulator', res, show_window = True, vsync=False)

    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    initCamera(res, camera, domain_start=case.ps.domain_start * w2s_ratio, domain_end=case.ps.domain_end * w2s_ratio, DIM=case.ps.dim)
    scene.set_camera(camera)
    canvas = window.get_canvas()
    canvas.set_background_color((0.3, 0.3, 0.3))
    light_z_len = case.ps.domain_size[2] if case.ps.dim == 3 else min(case.ps.domain_size[0:2])

    # Simulation path
    cappath = os.getcwd() + r"\screenshots"
    if not os.path.exists(cappath):
        os.mkdir(cappath)
    if save_png > 0 or save_csv:
        simpath = os.getcwd() + "\\sim_" + time_stamp0
        if not os.path.exists(simpath):
            os.mkdir(simpath)
        os.chdir(simpath)


    # Control para
    count_step = 0
    movement_speed = 0.01 if case.ps.dim == 3 else 0.0
    substeps = 1 if substeps < 1 else substeps
    stop_at_step_tmp = stop_every_step

    # Draw box
    num_box_pt = 8 if case.ps.dim == 3 else 4
    num_box_line = 12 if case.ps.dim == 3 else 4
    box_anchors = ti.Vector.field(case.ps.dim3, dtype=ti.f32, shape=num_box_pt)
    box_lines_indices = ti.Vector.field(2, int, shape=num_box_line)
    domain_anchors_np, box_lines_indices_np = calcBoxInfo(case.ps.domain_start, case.ps.domain_end, case.ps.dim)
    domain_color = (0.99,0.68,0.28)
    box_lines_indices.from_numpy(box_lines_indices_np)

    # Draw axis
    axis_anchors = ti.Vector.field(case.ps.dim3, dtype=ti.f32, shape=4)
    axis_lines_indices = ti.Vector.field(2, int, shape=3)
    axis_color = ti.Vector.field(3, dtype=ti.f32, shape=4)
    axis_anchors_np = np.array([[0,0,0], [w2s_ratio,0,0], [0,w2s_ratio,0], [0,0,w2s_ratio]], dtype=np.float32)
    axis_lines_indices_np = np.array([[0,1], [0,2], [0,3]], dtype=np.int32)
    axis_color_np = np.array([[1,1,1], [1,0,0], [0,1,0], [0,0,1]])
    axis_anchors.from_numpy(axis_anchors_np)
    axis_lines_indices.from_numpy(axis_lines_indices_np)
    axis_color.from_numpy(axis_color_np)

    # Show information
    if case.ps.dim == 2:
        print("UI 2D starts to serve!")
    elif case.ps.dim == 3:
        print("UI 3D starts to serve!")

    # Controls at step 0
    assign_color(case, given_max, given_min, fix_max, fix_min)
    if len(show_pt_info) > 0:
        show_msg(0, show_pt_info, case)
    if save_csv:
        export_csv(0, case, simpath)

    ##############################################
    # Run
    ##############################################
    while window.running:
        # run sim
        if not pause_flag:
            # step
            for i in range(substeps):
                # print("========", count_step)
                case.solver.step()
                count_step += 1
            assign_color(case, given_max, given_min, fix_max, fix_min)

            # msg
            if len(show_pt_info) > 0:
                show_msg(count_step, show_pt_info, case)

            # csv
            if save_csv:
                export_csv(count_step, case, simpath)

        # scene basic settings
        if movement_speed > 0:
            camera.track_user_inputs(window, movement_speed=movement_speed, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.point_light(
            ((case.ps.domain_end[0] + case.ps.domain_start[0]) * w2s_ratio / 2.0,
              case.ps.domain_end[1] * w2s_ratio * 0.75,
             (case.ps.domain_end[2] + light_z_len) * w2s_ratio),
            color=(1.0, 1.0, 1.0))
        scene.point_light(
            ((case.ps.domain_end[0] + case.ps.domain_start[0]) * w2s_ratio / 2.0,
              case.ps.domain_end[1] * w2s_ratio * 0.75,
             (case.ps.domain_start[2] - light_z_len) * w2s_ratio),
            color=(1.0, 1.0, 1.0))
        scene.lines(axis_anchors, indices=axis_lines_indices, per_vertex_color=axis_color, width=3.0)

        # draw domain
        box_anchors.from_numpy(domain_anchors_np * w2s_ratio)
        scene.lines(box_anchors, indices=box_lines_indices, color=domain_color, width=1.0)

        # draw particles
        draw_radius = case.ps.particle_radius * w2s_ratio * kradius
        case.ps.copy2vis(w2s_ratio)
        scene.particles(case.ps.pt.pos2vis, radius=draw_radius, per_vertex_color=case.ps.pt.color)

        canvas.scene(scene)

        # panel
        cur_time = case.solver.dt[None] * count_step
        window.GUI.begin("Running Info", 0.03, 0.03, 0.24, 0.2)
        window.GUI.text("Total particle number: {ptnum:,}".format(ptnum=case.ps.particle_num[None]))
        window.GUI.text("Step: {fstep:,}".format(fstep=count_step))
        window.GUI.text('Time: {t:.6f}s, dt={dt:.6f}s'.format(t=cur_time, dt=case.solver.dt[None]))
        window.GUI.text("colorbar: {str}".format(str=chooseColorTitle(case.ps.color_title)))
        window.GUI.text("max value: {maxv:.6f}".format(maxv=case.ps.vmax[None]))
        window.GUI.text("min value: {minv:.6f}".format(minv=case.ps.vmin[None]))
        window.GUI.end()

        str_solver = "Weakly Compressible" if case.solver_type == 1 else "Mohr-Coulomb mu(I)" if case.solver_type == 2 else "Drucker-Prager" if case.solver_type == 3 else "None"
        str_TI = "1 Symplectic Euler" if case.solver.flagTI == 1 else "2 Leap-Frog" if case.solver.flagTI == 2 else "4 Runge-Kutta" if case.solver.flagTI == 4 else "None"
        str_bdy = "Enforced collision" if case.ps.flag_boundary == case.ps.bdy_collision else "Dummy particles" if case.ps.flag_boundary == case.ps.bdy_dummy else "Repulsive particles" if case.ps.flag_boundary == case.ps.bdy_rep else "Dummy + repulsive pts" if case.ps.flag_boundary == case.ps.bdy_dummy_rep else "None"
        str_kernel = "Cubic spline" if case.solver.flagKernel == 0 else "Wendland C2" if case.solver.flagKernel == 1 else "None"
        str_kernel_corr = "CSPM" if case.solver.flagKernelCorr == 1 else "MLS" if case.solver.flagKernelCorr == 2 else "None"
        window.GUI.begin("Simulation Info", 0.3, 0.03, 0.24, 0.2)
        window.GUI.text("Solver: " + str_solver)
        window.GUI.text("Time integ: " + str_TI)
        window.GUI.text("Boundary: " + str_bdy)
        window.GUI.text("Kernel func: " + str_kernel)
        window.GUI.text("Kernel corr: " + str_kernel_corr)
        window.GUI.end()

        window.GUI.begin("Control", 0.57, 0.03, 0.32, 0.2)
        movement_speed = window.GUI.slider_float("speed of camera", movement_speed, 0.0, 0.025)
        kradius = window.GUI.slider_float("drawing radius", kradius, 0.1, 2.0)
        # case.ps.color_title = window.GUI.slider_int("color title", case.ps.color_title, 0, 10)
        window.GUI.end()

        # control
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                print("Simulator exits!")
                window.running = False
            elif e.key == ti.ui.SPACE:
                pause_flag = not pause_flag
            elif e.key == 'v':
                initCamera(res, camera, domain_start=case.ps.domain_start * w2s_ratio, domain_end=case.ps.domain_end * w2s_ratio, DIM=case.ps.dim)
                print("Camera initialized!")
            elif e.key == 'p':
                captureScreen(window, cappath, get_time_stamp())
            elif e.key == 'r':
                case.ps.clear_particles()
                case.ps.initialize_particles()
                assign_color(case, given_max, given_min, fix_max, fix_min)
                count_step = 0
                pause_flag = True
                print("Reset ERROR NOW!!!!!")


        # stop
        if count_step >= stop_at_step and stop_at_step > 0:
            pause_flag = True
            stop_at_step = 0
        if stop_every_step > 0 and substeps > 0 and stop_every_step >= substeps:
            if count_step >= stop_at_step_tmp:
                pause_flag = True
                stop_at_step_tmp += stop_every_step
        if cur_time >= stop_at_time and stop_at_time > 0:
            pause_flag = True
            stop_at_time = 0


        # exit
        if (count_step >= exit_at_step and exit_at_step > 0) or (cur_time >= exit_at_time and exit_at_time > 0):
            captureScreen(window, cappath, get_time_stamp())
            window.running = False

        # export
        if save_png > 0 and count_step % (save_png * substeps) == 0:
            window.save_image(f"{count_step:06d}.png")

        window.show()



##############################################
# Assist
##############################################
def get_time_stamp():
    return datetime.today().strftime("%Y_%m_%d_%H%M%S")

def initCamera(res, camera, domain_start, domain_end, DIM):
    domain_size = domain_end - domain_start
    domain_center = domain_start + domain_size / 2
    if DIM == 3:
        camera.position(domain_end[0], domain_end[1] - domain_size[1] * 0.2, domain_end[2] + domain_size[2] * 2)
        camera.lookat(domain_start[0], domain_start[1], domain_start[2] - domain_size[2] * 2)
    elif DIM == 2:
        x2y_ratio = 0.5 * (res[1] / res[0])
        camera.position(domain_center[0], domain_start[1] + x2y_ratio * domain_size[0], domain_start[2] + max(domain_size[0:2]))
        camera.lookat(domain_center[0], domain_start[1] + x2y_ratio * domain_size[0], domain_center[2])
    camera.fov(45)

def calcBoxInfo(box_start, box_end, DIM):
    x_min, y_min, z_min = box_start[0], box_start[1], box_start[2]
    x_max, y_max, z_max = box_end[0], box_end[1], box_end[2]
    box_anchors_np = []
    box_lines_indices_np = []
    if DIM == 3:
        box_anchors_np = np.array([[x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_min, z_max], [x_min, y_min, z_max], [x_min, y_max, z_min], [x_max, y_max, z_min], [x_max, y_max, z_max], [x_min, y_max, z_max]], dtype=np.float32)
        box_lines_indices_np = np.array([[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4], [0,4], [1,5], [2,6], [3,7]], dtype=np.int32)
    elif DIM == 2:
        box_anchors_np = np.array([[x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min]], dtype=np.float32)
        box_lines_indices_np = np.array([[0,1], [1,2], [2,3], [3,0]], dtype=np.int32)
    return box_anchors_np, box_lines_indices_np

def captureScreen(window, cappath, time_stamp):
    fname = os.path.join(cappath, f"screenshot{time_stamp}.png")
    window.save_image(fname)
    print(f"Screenshot has been saved to {fname}")

def assign_color(case: Simulation, given_max, given_min, fix_max, fix_min):
    if case.ps.color_title > 0:
        case.solver.assign_value_color()
        case.ps.v_maxmin(given_max, given_min, fix_max, fix_min)
        case.ps.set_color()

# ! temporary low performance!
def show_msg(step: int, show_pt_info: int, case: Simulation):
    print("---- ---- ---- %06d ---- ---- ----" % step)
    for i in show_pt_info:
        pti = case.ps.pt[i]
        str_msg = "---- pt[%06d]: id0=%06d, mat=%02d, x=[%.6f, %.6f, %.6f], v=[(]%.6f, %.6f, %.6f], ρ=%.6f, pressure=%.6f, σ=[xx %.6f, yy %.6f, zz %.6f, xy %.6f, yz %.6f, zx %.6f], ∇v=[[%.6f, %.6f, %.6f], [%.6f, %.6f, %.6f], [%.6f, %.6f, %.6f]], dist_B=%.6f" % (i, pti.id0, pti.mat_type, pti.x.x, pti.x.y, pti.x.z, pti.v.x, pti.v.y, pti.v.z, pti.density, pti.pressure, pti.stress[0,0], pti.stress[1,1], pti.stress[2,2], pti.stress[0,1], pti.stress[1,2], pti.stress[2,0], pti.v_grad[0,0], pti.v_grad[0,1], pti.v_grad[0,2], pti.v_grad[1,0], pti.v_grad[1,1], pti.v_grad[1,2], pti.v_grad[2,0], pti.v_grad[2,1], pti.v_grad[2,2], pti.dist_B)
        print(str_msg)

def export_csv(step: int, case: Simulation, simpath: str):
    csv_filename = simpath + "\\csv_%06d.csv" % (step)
    fid = open(csv_filename, "w", encoding="utf-8", newline="")
    fid_writer = csv.writer(fid)
    str_title = ["pid", "id0", "gid", "mat", "posx", "posy", "posz", "vx", "vy", "vz", "rho", "p", "sxx", "syy", "szz", "sxy", "syz", "szx", "gvxx", "gvxy", "gvxz", "gvyx", "gvyy", "gvyz", "gvzx", "gvzy", "gvzz", "dist_B"]
    fid_writer.writerow(str_title)
    for i in range(case.ps.particle_num[None]):
        pti = case.ps.pt[i]
        str_i = [i, pti.id0, pti.grid_ids, pti.mat_type, pti.x.x, pti.x.y, pti.x.z, pti.v.x, pti.v.y, pti.v.z, pti.density, pti.pressure, pti.stress[0,0], pti.stress[1,1], pti.stress[2,2], pti.stress[0,1], pti.stress[1,2], pti.stress[2,0], pti.v_grad[0,0], pti.v_grad[0,1], pti.v_grad[0,2], pti.v_grad[1,0], pti.v_grad[1,1], pti.v_grad[1,2], pti.v_grad[2,0], pti.v_grad[2,1], pti.v_grad[2,2], pti.dist_B]
        fid_writer.writerow(str_i)
    fid.close()
    print("---- ---- %06d csv exported" % step)

##############################################
# Color title
##############################################
def chooseColorTitle(flag):
    if flag is not None:
        if flag == 0:
            res = "material"
        elif flag == 1:
            res = "index"
        elif flag == 2:
            res = "density kg/m3"
        elif flag == 21:
            res = "d density kg/m3/s"
        elif flag == 3:
            res = "velocity norm m/s"
        elif flag == 31:
            res = "velocity x m/s"
        elif flag == 32:
            res = "velocity y m/s"
        elif flag == 33:
            res = "velocity z m/s"
        elif flag == 34:
            res = "d velocity norm m/s2"
        elif flag == 35:
            res = "d velocity x m/s2"
        elif flag == 36:
            res = "d velocity y m/s2"
        elif flag == 37:
            res = "d velocity z m/s2"
        elif flag == 4:
            res = "position m"
        elif flag == 41:
            res = "position x m"
        elif flag == 42:
            res = "position y m"
        elif flag == 43:
            res = "position z m"
        elif flag == 44:
            res = "displacement norm m"
        elif flag == 5:
            res = "stress Pa"
        elif flag == 51:
            res = "stress xx Pa"
        elif flag == 52:
            res = "stress yy Pa"
        elif flag == 53:
            res = "stress zz Pa"
        elif flag == 54:
            res = "stress xy Pa"
        elif flag == 55:
            res = "stress yz Pa"
        elif flag == 56:
            res = "stress zx Pa"
        elif flag == 57:
            res = "stress hydro Pa"
        elif flag == 58:
            res = "stress devia Pa"
        elif flag == 6:
            res = "strain"
        elif flag == 61:
            res = "strain pla equ"
        elif flag == 62:
            res = "strain pla dev"
        elif flag == 7:
            res = "pressure Pa"
        elif flag >= 100:
            res = "test"
        else:
            res = "Null"
    return res




##############################################
# Test
##############################################

