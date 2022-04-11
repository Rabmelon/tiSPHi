import taichi as ti
import numpy as np
from particle_system import ParticleSystem
# from wcsph import WCSPHSolver

ti.init(arch=ti.cpu)

# Use GPU for higher peformance if available
# ti.init(arch=ti.gpu, device_memory_GB=3, packed=True)


if __name__ == "__main__":
    ps = ParticleSystem((512, 512))

    # I note my change of program with comment: -----------------------------------------------------------------------
    ps.add_cube(lower_corner=[1, 2],
                cube_size=[2.0, 4.0],
                velocity=[-5.0, -10.0],
                density=1000.0,
                color=0x956333,
                material=1)

    ps.initialize_particle_system()

    # wcsph_solver = WCSPHSolver(ps)
    gui = ti.GUI(background_color=0xFFFFFF)
    while gui.running:
        # for i in range(5):
            # wcsph_solver.step()
        particle_info = ps.dump()
        gui.circles(particle_info['position'] * ps.screen_to_world_ratio / 512,
                    radius=ps.particle_radius / 1.5 * ps.screen_to_world_ratio,
                    color=0x956333)

        # Asssitant drawing of padding -----------------------------------------------------------------------
        corner_dl = (np.zeros(2) + ps.padding - ps.particle_radius) * ps.screen_to_world_ratio / 512
        corner_tr = (ps.bound - ps.padding + ps.particle_radius) * ps.screen_to_world_ratio / 512
        gui.line(corner_dl, [corner_tr[0], corner_dl[1]], radius=0.5*ps.particle_radius*ps.screen_to_world_ratio, color=0xff0000)
        gui.line([corner_tr[0], corner_dl[1]], corner_tr, radius=0.5*ps.particle_radius*ps.screen_to_world_ratio, color=0xff0000)
        gui.line(corner_tr, [corner_dl[0], corner_tr[1]], radius=0.5*ps.particle_radius*ps.screen_to_world_ratio, color=0xff0000)
        gui.line([corner_dl[0], corner_tr[1]], corner_dl, radius=0.5*ps.particle_radius*ps.screen_to_world_ratio, color=0xff0000)

        gui.show()