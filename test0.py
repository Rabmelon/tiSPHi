import taichi as ti
from eng.particle_system import *
from eng.sph_solver import *

# TODO: Test wcsph code in taichi course and try to understand its parts.

ti.init(arch=ti.cpu)

res = [32, 32]
case1 = ParticleSystem(res)
case1.add_cube(lower_corner=[1, 2],
               cube_size=[2.0, 4.0],
               velocity=[-5.0, -10.0],
               density=1000.0,
               color=0x956333,
               material=1)

particle_info = case1.dump()
case1.initialize_particle_system()

gui = ti.GUI(background_color=0xFFFFFF)
while gui.running:
    gui.circles(particle_info['position'] * case1.screen_to_world_ratio / 512,
                radius=case1.particle_radius * case1.screen_to_world_ratio,
                color=0x956333)
    gui.line([1 * case1.screen_to_world_ratio / 512, 2 * case1.screen_to_world_ratio / 512],
             [3 * case1.screen_to_world_ratio / 512, 6 * case1.screen_to_world_ratio / 512],
             radius=0.05 * case1.screen_to_world_ratio,
             color=0xff0000)
    gui.line([0 * case1.screen_to_world_ratio / 512, 0 * case1.screen_to_world_ratio / 512],
             [1 * case1.screen_to_world_ratio / 512, 2 * case1.screen_to_world_ratio / 512],
             radius=0.05 * case1.screen_to_world_ratio,
             color=0xffff00)
    gui.show()
