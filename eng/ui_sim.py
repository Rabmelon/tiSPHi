import taichi as ti
import numpy as np
import os
from datetime import datetime
from eng.simulation import Simulation, SimConfiger


# TODO: need a faster exporter

def ui_sim(case: Simulation):
    # Paras
    str_comment = "Comment: " + case.cfg.get_cfg("comment")
    substeps = case.cfg.get_cfg("stepsPerRenderUpdate")
    stop_at_step = case.cfg.get_cfg("stopAtStep")
    exit_at_step = case.cfg.get_cfg("exitAtStep")
    stop_at_time = case.cfg.get_cfg("stopAtTime")
    exit_at_time = case.cfg.get_cfg("exitAtTime")
    stop_every_step = case.cfg.get_cfg("stopEveryStep")
    pause_flag = case.cfg.get_cfg("pauseAtStart")
    kradius = case.cfg.get_cfg("kradius")
    given_max = case.cfg.get_cfg("givenMax")
    given_min = case.cfg.get_cfg("givenMin")
    fix_max = case.cfg.get_cfg("fixMax")
    fix_min = case.cfg.get_cfg("fixMin")

    save_every_time = case.cfg.get_cfg("exportEveryTime")
    save_every_render = case.cfg.get_cfg("exportEveryRender")
    save_frame = case.cfg.get_cfg("exportFrame")
    save_vtk = case.cfg.get_cfg("exportVTK")
    save_csv = case.cfg.get_cfg("exportCSV")

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
    judge_sim_path = (save_every_render > 0 or save_every_time > 0) and (save_frame or save_vtk or save_csv)
    if judge_sim_path:
        simpath = os.getcwd() + "\\sim_" + time_stamp0
        if not os.path.exists(simpath):
            os.mkdir(simpath)
        os.chdir(simpath)


    # Control para
    count_step = 0
    movement_speed = 0.01 if case.ps.dim == 3 else 0.0
    substeps = 1 if substeps < 1 else substeps
    stop_at_step_tmp = stop_every_step
    save_every_time_tmp = save_every_time
    info_saved = False

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

    ##############################################
    # Run
    ##############################################
    while window.running:
        # run sim
        if not pause_flag:
            # step
            for _ in range(substeps):
                # print("========", count_step)
                case.solver.step()
                count_step += 1
            assign_color(case, given_max, given_min, fix_max, fix_min)

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
        str_pt_num = "Total particle number: {ptnum:,}".format(ptnum=case.ps.particle_num[None])
        str_step = "Step: {fstep:,}".format(fstep=count_step)
        str_dt = "dt={dt:.6f}s".format(dt=case.solver.dt[None])
        str_time = "Time: {t:.6f}s, ".format(t=cur_time) + str_dt
        str_colorbar = "colorbar: {str}".format(str=chooseColorTitle(case.ps.color_title))
        str_vmax = "max value: {maxv:.6f}".format(maxv=case.ps.vmax[None])
        str_vmin = "min value: {minv:.6f}".format(minv=case.ps.vmin[None])
        window.GUI.begin("Running Info", 0.03, 0.03, 0.24, 0.2)
        window.GUI.text(str_pt_num)
        window.GUI.text(str_step)
        window.GUI.text(str_time)
        window.GUI.text(str_colorbar)
        window.GUI.text(str_vmax)
        window.GUI.text(str_vmin)
        window.GUI.end()

        str_solver = "Solver: " + ("Weakly Compressible" if case.solver_type == 1 else "Mohr-Coulomb mu(I)" if case.solver_type == 2 else "Drucker-Prager" if case.solver_type == 3 else "None")
        str_TI = "Time integ: " + ("1 Symplectic Euler" if case.solver.flagTI == 1 else "2 Leap-Frog" if case.solver.flagTI == 2 else "4 Runge-Kutta" if case.solver.flagTI == 4 else "None")
        str_bdy = "Boundary: " + ("Enforced collision" if case.ps.flag_boundary == case.ps.bdy_collision else "Dummy particles" if case.ps.flag_boundary == case.ps.bdy_dummy else "Repulsive particles" if case.ps.flag_boundary == case.ps.bdy_rep else "Dummy + repulsive pts" if case.ps.flag_boundary == case.ps.bdy_dummy_rep else "None")
        str_kernel = "Kernel func: " + ("Cubic spline" if case.solver.flagKernel == 0 else "Wendland C2" if case.solver.flagKernel == 1 else "None")
        str_kernel_corr = "Kernel corr: " + ("CSPM" if case.solver.flagKernelCorr == 1 else "MLS" if case.solver.flagKernelCorr == 2 else "None")
        str_pos_upd = "Position upd: " + ("XSPH" if case.solver.flagXSPH == 1 else "None")
        window.GUI.begin("Simulation Info", 0.3, 0.03, 0.24, 0.2)
        window.GUI.text(str_solver)
        window.GUI.text(str_TI)
        window.GUI.text(str_bdy)
        window.GUI.text(str_kernel)
        window.GUI.text(str_kernel_corr)
        window.GUI.text(str_pos_upd)
        window.GUI.text(str_comment)
        window.GUI.end()

        window.GUI.begin("Control", 0.57, 0.03, 0.32, 0.2)
        movement_speed = window.GUI.slider_float("speed of camera", movement_speed, 0.0, 0.025)
        kradius = window.GUI.slider_float("drawing radius", kradius, 0.1, 2.0)
        # case.ps.color_title = window.GUI.slider_int("color title", case.ps.color_title, 0, 10)
        window.GUI.end()

        if judge_sim_path and not info_saved:
            save_info(simpath, "==== Running Info ====\n%s\n%s\n\n==== Simulation Info ====\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n\n==== Note ====\n\"time.secx1e6.0349220\" means the frame of 0.349220s\n\n\n\n==== Configure Info ====\n%s" % (str_pt_num, str_dt, str_solver, str_TI, str_bdy, str_kernel, str_kernel_corr, str_pos_upd, str_comment, case.cfg.config))
            info_saved = True

        # control
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                if judge_sim_path:
                    captureScreen(window, simpath, get_time_stamp())
                else:
                    captureScreen(window, cappath, get_time_stamp())
                print("Simulator exits!")
                window.running = False
            elif e.key == ti.ui.SPACE:
                pause_flag = not pause_flag
                if pause_flag:
                    print("Simulation is stopped!")
                elif not pause_flag and count_step == 0:
                    print("Simulation starts!")
                else:
                    print("Simulation is resumed!")
            elif e.key == 'v':
                initCamera(res, camera, domain_start=case.ps.domain_start * w2s_ratio, domain_end=case.ps.domain_end * w2s_ratio, DIM=case.ps.dim)
                print("Camera initialized!")
            elif e.key == 'p':
                captureScreen(window, cappath, get_time_stamp())
            elif e.key == 'r':
                # ! wrong now
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

        # export
        if save_every_time > 0 and judge_sim_path:
            if count_step == 0 and pause_flag == False:
                func_export_time(window, save_frame, save_vtk, save_csv, cur_time, simpath, case)
            elif cur_time >= save_every_time:
                func_export_time(window, save_frame, save_vtk, save_csv, cur_time, simpath, case)
                save_every_time += save_every_time_tmp
        elif save_every_render > 0 and judge_sim_path:
            if (count_step == 0 and pause_flag == False) or (count_step % (save_every_render * substeps) == 0 and count_step > 0):
                func_export_step(window, save_frame, save_vtk, save_csv, count_step, simpath, case)

        # exit
        if (count_step >= exit_at_step and exit_at_step > 0) or (cur_time >= exit_at_time and exit_at_time > 0):
            if judge_sim_path:
                captureScreen(window, simpath, get_time_stamp())
            else:
                captureScreen(window, cappath, get_time_stamp())
            window.running = False

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

def captureScreen(window, imgpath, time_stamp):
    fname = os.path.join(imgpath, f"screenshot{time_stamp}.png")
    window.save_image(fname)
    print(f"Screenshot has been saved to {fname}")

def assign_color(case: Simulation, given_max, given_min, fix_max, fix_min):
    if case.ps.color_title > 0:
        case.solver.assign_value_color()
        case.ps.v_maxmin(given_max, given_min, fix_max, fix_min)
        case.ps.set_color()



##############################################
# Export
##############################################
def export_csv(stamp: str, simpath: str, case: Simulation):
    # ! need to upd with the ps dump
    csv_filename = simpath + "\\sim.csv.%s.csv" % (stamp)
    pos_dump, data_dump = case.ps.dump()
    data_csv = np.array([data_dump["id0"], data_dump["objId"], data_dump["material"], pos_dump["pos.x"], pos_dump["pos.y"], pos_dump["pos.z"], data_dump["vel.x"], data_dump["vel.y"], data_dump["vel.z"], data_dump["density"], data_dump["stress.xx"], data_dump["stress.yy"], data_dump["stress.zz"], data_dump["stress.xy"], data_dump["stress.yz"], data_dump["stress.zx"], data_dump["strain_equ"]]).T
    np.savetxt(csv_filename, data_csv, delimiter=",", header="id0, objId, material, pos.x, pos.y, pos.z, vel.x, vel.y, vel.z, density, stress.xx, stress.yy, stress.zz, stress.xy, stress.yz, stress.zx, strain_equ")

def export_vtk(stamp: str, simpath: str, case: Simulation):
    from pyevtk.hl import pointsToVTK
    vtk_filename = simpath + "\\sim.vtk.%s" % (stamp)
    pos_dump, data_dump = case.ps.dump()
    pointsToVTK(vtk_filename, x=pos_dump["pos.x"], y=pos_dump["pos.y"], z=pos_dump["pos.z"], data=data_dump)

def func_export_step(window, save_frame, save_vtk, save_csv, count_step, simpath, case: Simulation):
    stamp = f"{count_step:06d}"
    if save_frame:
        window.save_image("%s.png" % (stamp))
    if save_vtk:
        export_vtk(stamp, simpath, case)
    if save_csv:
        export_csv(stamp, simpath, case)

def func_export_time(window, save_frame, save_vtk, save_csv, cur_time, simpath, case: Simulation):
    int_cur_time = int(cur_time * 1e6)
    stamp = f"time.secx1e6.{int_cur_time:07d}"
    if save_frame:
        window.save_image("%s.png" % (stamp))
    if save_vtk:
        export_vtk(stamp, simpath, case)
    if save_csv:
        export_csv(stamp, simpath, case)

def save_info(simpath, str):
    filepath = simpath + "\\_info.txt"
    with open(filepath, "w") as f:
        f.write(str)



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
            res = "velocity y -m/s"
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
            res = "stress yy -Pa"
        elif flag == 53:
            res = "stress zz -Pa"
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
            res = "strain equ"
        elif flag == 62:
            res = "strain equ p"
        elif flag == 7:
            res = "pressure Pa"
        elif flag == 8:
            res = "ret map proj"
        elif flag >= 100:
            res = "test"
        else:
            res = "Null"
    return res




##############################################
# Test
##############################################

