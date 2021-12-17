import taichi as ti
import numpy as np
from eng.particle_system import *
from eng.wcsph import *

ti.init(arch=ti.cpu, debug=True)
# ti.init(arch=ti.gpu, device_memory_GB=3, packed=True)

if __name__ == "__main__":
    res = [512, 512]
    case1 = ParticleSystem(res)

    case1.add_cube(lower_corner=[0.5, 0.5],
                   cube_size=[3.0, 5.0],
                   velocity=[.0, 5.0],
                   density=1000.0,
                   color=0x956333,
                   material=1)

    wcsph_solver = WCSPHSolver(case1)
    gui = ti.GUI(background_color=0xFFFFFF)
    while gui.running:
        for i in range(5):
            wcsph_solver.step()
        particle_info = case1.dump()
        gui.circles(particle_info['position'] * case1.screen_to_world_ratio / res[0],
                    radius=case1.particle_radius * case1.screen_to_world_ratio,
                    color=0x956333)
        gui.show()
