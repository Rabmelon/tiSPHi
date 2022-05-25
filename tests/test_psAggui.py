import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

@ti.data_oriented
class TmpParticleSystem:
    def __init__(self, world, radius) -> None:
        print("temp particle system serve.")
        self.world = np.array(world)
        self.grid_size = 6 * radius
        self.particle_num = ti.field(int, shape=())  # record the number of current particles
        self.particle_max_num = 2**16  # the max number of all particles, as 65536
        self.x = ti.Vector.field(2, dtype=float)     # position
        self.particles_node = ti.root.dense(ti.i, self.particle_max_num)    # 使用稠密数据结构开辟每个粒子数据的存储空间，按列存储
        self.particles_node.place(self.x)

    @ti.kernel
    def set_pnum(self, num: int):
        self.particle_num[None] = num

    @ti.kernel
    def ge_x_rand(self):
        for i in range(self.particle_num[None]):
            self.x[i] = ti.Vector([ti.random(), ti.random()]) * 100


def gguishow(case, world, s2w_ratio):
    drawworld = [i + 2 * case.grid_size for i in world]
    res = (np.array(drawworld) * s2w_ratio).astype(int)
    window = ti.ui.Window('SPH window', res=(max(res), max(res)))
    canvas = window.get_canvas()
    show_pos = [0.0, 0.0]

    while window.running:
        # draw
        canvas.circles(case.x, radius=0.1, color=(0.5,0.5,0.5))

        # control
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
            elif e.key == ti.ui.SPACE:
                flag_pause = not flag_pause
            elif e.key == ti.ui.LMB:
                show_pos = [i / s2w_ratio * max(res) - case.grid_size for i in window.get_cursor_pos()]

        # show text
        window.GUI.begin("Info", 0.03, 0.03, 0.3, 0.2)
        window.GUI.text('Total particle number: {pnum:,}'.format(pnum=case.particle_num[None]))
        window.GUI.text('Pos: {px:.3f}, {py:.3f}'.format(px=show_pos[0], py=show_pos[1]))
        window.GUI.end()

        window.show()



if __name__ == "__main__":
    print("hallo test particle system and ggui show!")

    world = [120, 60]
    s2w_ratio = 5
    radius = 1
    case = TmpParticleSystem(world, radius)
    case.set_pnum(100)
    case.ge_x_rand()
    gguishow(case, world, s2w_ratio)

