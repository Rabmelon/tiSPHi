import taichi as ti
import numpy as np

ti.init(arch=ti.cpu, cpu_max_num_threads=1)

epsilon = 1e-8

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

    # TODO: trying to create a rectangular window, wrong in transfering tuple 'res'
    @ti.kernel
    def copy2vis(self, s2w_ratio: float, res: ti.ext_arr()):
        for i in range(self.particle_num[None]):
            # self.pos2vis[i][0] = (self.x[i][0] + self.grid_size) * s2w_ratio / res_x
            # self.pos2vis[i][1] = (self.x[i][1] + self.grid_size) * s2w_ratio / res_y
            for j in ti.static(range(2)):
                self.pos2vis[i][j] = (self.x[i][j] + self.grid_size) * s2w_ratio / res[j]

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

# TODO: original form of numpy function (really slow!!!)
def gen_grid_line_2d(world, grid_line, canvas, ld_size, w2s, width=0.0025, color=(0.8,0.8,0.8)):
    dim = len(world)
    if not isinstance(grid_line,list):
        grid_line = [grid_line for _ in range(dim)]
    num_grid_point = [int((world[i] - 1e-8) // grid_line[i]) for i in range(dim)]
    num_all_grid_point = sum(num_grid_point)
    num_all2_grid_point = 2 * num_all_grid_point
    np_pos_line = np.array([[0.0 for _ in range(dim)] for _ in range(num_all2_grid_point)])
    np_indices_line = np.array([[i, i + num_all_grid_point] for i in range(num_all_grid_point)])
    pos_line = ti.Vector.field(dim, float, shape=num_all2_grid_point)
    indices_line = ti.Vector.field(2, int, shape=num_all_grid_point)
    for id in range(dim):
        id2 = dim - 1 - id
        for i in range(num_grid_point[id]):
            np_pos_line[i + sum(num_grid_point[0:id])][id] = (i + 1) * grid_line[id]
            np_pos_line[i + sum(num_grid_point[0:id]) + num_all_grid_point][id] = (i + 1) * grid_line[id]
            np_pos_line[i + sum(num_grid_point[0:id]) + num_all_grid_point][id2] = world[id2]
            # print(id, i, np_pos_line[i + sum(num_grid_point[0:id])], np_pos_line[i + sum(num_grid_point[0:id]) + num_all_grid_point])
    pos_line.from_numpy((np_pos_line + ld_size) * w2s)
    indices_line.from_numpy(np_indices_line)
    canvas.lines(pos_line, width, indices_line, color)


def gguishow(case, world, s2w_ratio, grid_line=None):
    drawworld = [i + 2 * case.grid_size for i in world]
    res = tuple((np.array(drawworld) * s2w_ratio).astype(int))
    resv = ti.Vector(res)
    window = ti.ui.Window('window', res=res)
    canvas = window.get_canvas()
    canvas.set_background_color((1,1,1))
    show_pos = [0.0, 0.0]
    flag_pause = False

    while window.running:
        # draw grid line
        if grid_line is not None:
            gen_grid_line_2d(world, grid_line, canvas, case.grid_size, w2s=s2w_ratio / max(res))

        # draw main part
        case.copy2vis(s2w_ratio, resv)
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
    gguishow(case, world, s2w_ratio, grid_line=15)
