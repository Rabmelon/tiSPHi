import taichi as ti
import numpy as np
from functools import reduce    # 整数：累加；字符串、列表、元组：拼接。lambda为使用匿名函数

ti.init(arch=ti.cpu, debug=True)

@ti.data_oriented
class ParticleSystem:
    def __init__(self, world, radius):
        print("Class Particle System starts to serve!")

        # Basic information of the simulation
        self.world = np.array(world)
        self.dim = len(world)
        assert self.dim in (2, 3), "SPH solver supports only 2D and 3D particle system and 2D ractangular world from ld_pos(0,0) now."

        # Basic particle property 粒子的基本属性
        self.particle_radius = radius
        self.particle_diameter = 2.0 * self.particle_radius
        self.kh = 3   # times the support domain radius to the particle radius. Should be adapted automaticlly soon
        self.support_radius = self.kh * self.particle_diameter
        self.m_V = self.particle_diameter**self.dim     # m2 or m3 for cubic discrete
        self.particle_max_num = 2**16  # the max number of all particles, as 65536
        self.particle_max_num_per_cell = 100  # the max number of particles in each cell
        self.particle_max_num_neighbors = 100  # the max number of neighbour particles of each particle
        self.particle_num = ti.field(int, shape=())  # record the number of current particles

        # Grid property 背景格网的基本属性
        self.grid_size = self.support_radius  # 令格网边长为2倍的支持域半径，这样只需遍历4个grid就可以获取邻域粒子【不好使！】at the same equals to padding width
        # self.grid_size = self.support_radius + 1e-5 # 支持域半径加一个微小量
        self.bound = [[-self.grid_size, -self.grid_size], [i + self.grid_size for i in world]]    # Simply create a rectangular range
        self.range = np.array([self.bound[1][0] - self.bound[0][0], self.bound[1][1] - self.bound[0][1]])    # Simply create a rectangular range
        self.grid_num = np.ceil(self.range / self.grid_size).astype(int)  # 格网总数
        self.grid_particles_num = ti.field(int)  # 每个格网中的粒子总数
        self.grid_particles = ti.field(int)  # 每个格网中的粒子编号

        self.offset_check = ti.field(int)

        # Particle related property 粒子携带的属性信息
        # Basic
        self.x = ti.Vector.field(self.dim, dtype=float)     # position
        self.val = ti.field(dtype=float)                      # store a value
        self.particle_neighbors_num = ti.field(int)         # total number of neighbour particles
        self.particle_neighbors = ti.field(int)             # index of neighbour particles

        self.particles_node = ti.root.dense(ti.i, self.particle_max_num)
        self.particles_node.place(self.x, self.val)
        self.particles_node.place(self.particle_neighbors_num, self.offset_check)
        self.particle_node = self.particles_node.dense(ti.j, self.particle_max_num_neighbors)
        self.particle_node.place(self.particle_neighbors)
        self.grid_index = ti.ij if self.dim == 2 else ti.ijk          # 建立格网维度索引变量，xy or xyz
        self.grid_node = ti.root.dense(self.grid_index, self.grid_num)     # 使用稠密数据结构开辟每个格网中粒子总数的存储空间
        self.grid_node.place(self.grid_particles_num)
        self.cell_index = ti.k if self.dim == 2 else ti.l        # 建立粒子索引变量
        self.cell_node = self.grid_node.dense(self.cell_index, self.particle_max_num_per_cell)     # 使用稠密数据结构开辟每个格网中存储粒子编号的存储空间
        self.cell_node.place(self.grid_particles)

    ###########################################################################
    # NS
    ###########################################################################
    # 获取粒子位置对应的grid编号
    @ti.func
    def pos_to_index(self, pos):
        return ((pos - self.bound[0]) / self.grid_size).cast(int)

    # 检查cell的编号是否有效，即是否在grid内
    @ti.func
    def is_valid_cell(self, cell):
        # Check whether the cell is in the grid
        flag = True
        for d in ti.static(range(self.dim)):
            flag = flag and (0 <= cell[d] < self.grid_num[d])
        return flag

    # 计算每个粒子对应的grid编号，并将粒子编号加入到对应的grid中
    @ti.kernel
    def allocate_particles_to_grid(self):
        for p in range(self.particle_num[None]):
            cell = self.pos_to_index(self.x[p])                     # 当前粒子位于哪个grid
            offset = ti.atomic_add(self.grid_particles_num[cell], 1)    # 当前粒子是这个grid中的第几个粒子
            self.grid_particles[cell, offset] = p

    # 搜索邻域粒子，使用的应该是常规的基于格网的搜索方法
    @ti.kernel
    def search_neighbors(self):
        for p_i in range(self.particle_num[None]):
            center_cell = self.pos_to_index(self.x[p_i])
            print("x[", p_i, "]=(", self.x[p_i][0], ",", self.x[p_i][1], ") in cell [", center_cell[0], ",", center_cell[1], "]", end=" ns: ")
            cnt = 0
            self.offset_check[p_i] = 0
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
                print(self.offset_check[p_i], end=" ")
                self.offset_check[p_i] += 1   # -------------------------
                if cnt >= self.particle_max_num_neighbors:
                    break
                cell = center_cell + offset
                if not self.is_valid_cell(cell):
                    continue
                for j in range(self.grid_particles_num[cell]):
                    p_j = self.grid_particles[cell, j]
                    distance = (self.x[p_i] - self.x[p_j]).norm()
                    if p_i != p_j and distance < self.support_radius:
                        self.particle_neighbors[p_i, cnt] = p_j
                        print(p_j, end=",")
                        cnt += 1
            self.particle_neighbors_num[p_i] = cnt
            print("--", cnt, "endl")

    ###########################################################################
    # Add particles
    ###########################################################################
    # add one particle in p with given properties
    @ti.func
    def add_particle(self, p, val, x):
        self.val[p] = val
        self.x[p] = x

    # add particles with given properties
    @ti.kernel
    def add_particles(self, new_particles_num: int,
                      new_particles_value: ti.ext_arr(),
                      new_particles_positions: ti.ext_arr()):
        for p in range(self.particle_num[None],
                       self.particle_num[None] + new_particles_num):
            new_p = p - self.particle_num[None]
            x = ti.Vector.zero(float, self.dim)
            for d in ti.static(range(self.dim)):
                x[d] = new_particles_positions[new_p, d]
            self.add_particle(p, new_particles_value[new_p], x)
        self.particle_num[None] += new_particles_num
        assert self.particle_num[None] + new_particles_num <= self.particle_max_num, 'My Error: exceed the maximum number of particles!'

    # add particles in a cube region
    def add_cube(self, lower_corner, cube_size, value=None, offset=None):
        num_dim = []
        range_offset = offset if offset is not None else self.particle_diameter
        for i in range(self.dim):
            num_dim.append(np.arange(lower_corner[i] + self.particle_radius, lower_corner[i] + cube_size[i] + 1e-5, range_offset))
        num_new_particles = reduce(lambda x, y: x * y, [len(n) for n in num_dim])

        new_positions = np.array(np.meshgrid(*num_dim, sparse=False, indexing='ij' if self.dim == 2 else 'ijk'), dtype=np.float32)
        new_positions = new_positions.reshape(-1, reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        print("New cube's number and dim: ", new_positions.shape)

        value = np.full_like(np.zeros(num_new_particles), value if value is not None else 0.0)
        self.add_particles(num_new_particles, value, new_positions)
        self.initialize_particle_system()

    ###########################################################################
    # Initialise and Update the particle system b
    ###########################################################################
    def initialize_particle_system(self):
        self.grid_particles_num.fill(0)
        self.particle_neighbors.fill(-1)
        self.allocate_particles_to_grid()
        self.search_neighbors()





rec_world = [0.30, 0.40]   # a rectangle world start from (0, 0) to this pos
particle_radius = 0.01
case1 = ParticleSystem(rec_world, particle_radius)
case1.add_cube(lower_corner=[0.0, 0], cube_size=[0.24, 0.36])
