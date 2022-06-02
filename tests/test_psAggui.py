import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

@ti.data_oriented
class TmpParticleSystem:
    def __init__(self, world, radius) -> None:
        print("temp particle system serve.")
        self.world = np.array(world)
        self.radius = radius
        self.grid_size = 6 * self.radius
        self.particle_num = ti.field(int, shape=())  # record the number of current particles
        self.particle_max_num = 2**16  # the max number of all particles, as 65536
        self.x = ti.Vector.field(2, dtype=float)     # position
        self.val = ti.field(dtype=float)
        self.maxv = ti.field(float, shape=())
        self.pos2vis = ti.Vector.field(2, dtype=float)     # position
        self.color = ti.Vector.field(3, dtype=float)
        self.particles_node = ti.root.dense(ti.i, self.particle_max_num)    # 使用稠密数据结构开辟每个粒子数据的存储空间，按列存储
        self.particles_node.place(self.x, self.val, self.pos2vis, self.color)
        self.line_indices = ti.field(dtype=int)
        self.line_node = ti.root.dense(ti.i, (self.particle_max_num-1) * 2)
        self.line_node.place(self.line_indices)

    @ti.kernel
    def set_pnum(self, num: int):
        self.particle_num[None] = num

    @ti.kernel
    def set_value(self):
        for i in range(self.particle_num[None]):
            self.val[i] = self.x[i].max()

    @ti.kernel
    def ge_x_rand(self):
        for i in range(self.particle_num[None]):
            self.x[i] = ti.Vector([ti.random(), ti.random()]) * self.world.min()

    @ti.kernel
    def copy2vis(self, s2w_ratio: float, max_res: float):
        for i in range(self.particle_num[None]):
            for j in ti.static(range(2)):
                self.pos2vis[i][j] = (self.x[i][j] + self.grid_size) * s2w_ratio / max_res

    @ti.kernel
    def ge_line_indices(self):
        for i in range(self.particle_num[None]-1):
            self.line_indices[2 * i] = i
            self.line_indices[2 * i + 1] = i + 1

    @ti.kernel
    def get_max_v(self):
        maxv = 0.0
        for i in range(self.particle_num[None]):
            maxv = max(self.val[i], maxv)
        self.maxv[None] = maxv

    @ti.kernel
    def set_color(self):
        maxv1 = 1 / self.maxv[None]
        for i in range(self.particle_num[None]):
            self.color[i] = ti.Vector([1, self.val[i] * maxv1, 0])

    def print_pos(self):
        for i in range(self.particle_num[None]):
            print("%d: [%.3f, %.3f]" % (i, self.x[i][0], self.x[i][1]))



def gguishow(case, world, s2w_ratio):
    drawworld = [i + 2 * case.grid_size for i in world]
    res = (np.array(drawworld) * s2w_ratio).astype(int)
    window = ti.ui.Window('window', res=(res.max(), res.max()))
    canvas = window.get_canvas()
    canvas.set_background_color((1,1,1))
    show_pos = [0.0, 0.0]

    while window.running:
        # draw
        case.copy2vis(s2w_ratio, max(res).astype(float))
        canvas.lines(case.pos2vis, 0.005, indices=case.line_indices, color=(0.5,1,0.5))
        canvas.circles(case.pos2vis, radius=case.radius * s2w_ratio / max(res), per_vertex_color=case.color)

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
        window.GUI.text('max value: {maxv:.3f}'.format(maxv=case.maxv[None]))
        window.GUI.end()

        window.show()



if __name__ == "__main__":
    print("hallo test particle system and ggui show!")

    world = [120, 80]
    s2w_ratio = 5
    radius = 1
    case = TmpParticleSystem(world, radius)
    case.set_pnum(50)
    case.ge_x_rand()
    case.print_pos()
    case.set_value()
    case.get_max_v()
    case.set_color()
    case.ge_line_indices()
    gguishow(case, world, s2w_ratio)
