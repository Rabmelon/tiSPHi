import taichi as ti
import numpy as np
from eng.particle_system import *
from eng.wcsph import *

# TODO: Test wcsph code in taichi course and try to understand its parts.

ti.init(arch=ti.cpu, debug=True)
# ti.init(arch=ti.gpu, packed=True)

if __name__ == "__main__":
    res = [8, 8]
    case1 = ParticleSystem(res)

    case1.add_cube(lower_corner=[1, 2],
                   cube_size=[2.0, 4.0],
                   velocity=[2.0, -5.0],
                   density=1000.0,
                   color=0x956333,
                   material=1)
    # case1.add_cube(lower_corner=[0.5, 0.5],
    #                cube_size=[3.0, 5.0],
    #                velocity=[2.0, -5.0],
    #                density=1000.0,
    #                color=0x956333,
    #                material=1)

    case1.initialize_particle_system()
    particle_info = case1.dump()

    # wcsph_solver = WCSPHSolver(case1)

    # gui = ti.GUI(background_color=0xFFFFFF)
    # while gui.running:
    #     # for i in range(5):
    #     #     wcsph_solver.step()
    #     # particle_info = case1.dump()

    #     # draw
    #     gui.circles(particle_info['position'] * case1.screen_to_world_ratio / max(res),
    #                 radius=case1.particle_radius * case1.screen_to_world_ratio, color=0x956333)

    #     # 为何绘制不出top和right边界？
    #     gui.line([0, 0], [case1.bound[0], 0], radius=0.1 * case1.screen_to_world_ratio, color=0xff0000)
    #     gui.line([case1.bound[0], 0], case1.bound, radius=0.1 * case1.screen_to_world_ratio, color=0xff0000)
    #     gui.line(case1.bound, [0, case1.bound[1]], radius=0.1 * case1.screen_to_world_ratio, color=0xff0000)
    #     gui.line([0, case1.bound[1]], [0, 0], radius=0.1 * case1.screen_to_world_ratio, color=0xff0000)
    #     gui.line([0, 0], case1.bound, radius=0.1 * case1.screen_to_world_ratio, color=0xffff00)

    #     gui.show()
