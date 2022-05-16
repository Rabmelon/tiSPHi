import taichi as ti
import numpy as np
from functools import reduce    # 整数：累加；字符串、列表、元组：拼接。lambda为使用匿名函数

# TODO: Unify all coordinate systems and put padding area outside the real world.

@ti.data_oriented
class ParticleSystem:
    def __init__(self, world, radius):
        print("Hallo, class Particle System starts to serve!")

        # Basic information of the simulation
        self.dim = len(world)
        assert self.dim in (2, 3), "SPH solver supports only 2D and 3D particle system and 2D ractangular world from ld_pos(0,0) now."

        # Material 材料类型定义
        self.material_fluid = 1
        self.material_solid = 2
        self.material_dummy = 10

        # Basic particle property 粒子的基本属性
        self.particle_radius = radius
        self.particle_diameter = 2.0 * self.particle_radius
        self.kh = 6.0   # times the support domain radius to the particle radius. Should be adapted automaticlly soon
        self.support_radius = self.kh * self.particle_radius
        self.m_V = self.particle_diameter**self.dim     # m2 or m3 for cubic discrete
        self.particle_max_num = 2**16  # the max number of all particles, as 65536
        self.particle_max_num_per_cell = 100  # the max number of particles in each cell
        self.particle_max_num_neighbors = 100  # the max number of neighbour particles of each particle
        self.particle_num = ti.field(int, shape=())  # record the number of current particles

        # Grid property 背景格网的基本属性
        self.grid_size = 2 * self.support_radius  # 令格网边长为2倍的支持域半径，这样只需遍历4个grid就可以获取邻域粒子【不好使！】
        # self.grid_size = self.support_radius + 1e-5 # 支持域半径加一个微小量
        self.bound = [[-self.grid_size, -self.grid_size], [i + self.grid_size for i in world]]    # Simply create a rectangular range
        self.range = np.array([self.bound[1][0] - self.bound[0][0], self.bound[1][1] - self.bound[0][1]])    # Simply create a rectangular range
        self.grid_num = np.ceil(self.range / self.grid_size).astype(int)  # 格网总数
        self.grid_particles_num = ti.field(int)  # 每个格网中的粒子总数
        self.grid_particles = ti.field(int)  # 每个格网中的粒子编号
        self.padding = self.grid_size  # 边界padding, 用在enforce_rangeary函数中

        # Particle related property 粒子携带的属性信息
        # Basic
        self.x = ti.Vector.field(self.dim, dtype=float)     # position
        self.particle_neighbors_num = ti.field(int)         # total number of neighbour particles
        self.particle_neighbors = ti.field(int)             # index of neighbour particles
        self.material = ti.field(dtype=int)                 # material type
        self.color = ti.field(dtype=int)                    # color in drawing
        # Values
        self.v = ti.field(dtype=float)               # store a value

        # Place nodes on root
        self.particles_node = ti.root.dense(ti.i, self.particle_max_num)    # 使用稠密数据结构开辟每个粒子数据的存储空间，按列存储
        self.particles_node.place(self.x, self.material, self.color)
        self.particles_node.place(self.v)
        self.particles_node.place(self.particle_neighbors_num)
        self.particle_node = self.particles_node.dense(ti.j, self.particle_max_num_neighbors)    # 使用稠密数据结构开辟每个粒子邻域粒子编号的存储空间，按行存储
        self.particle_node.place(self.particle_neighbors)

        grid_index = ti.ij if self.dim == 2 else ti.ijk          # 建立格网维度索引变量，xy or xyz
        grid_node = ti.root.dense(grid_index, self.grid_num)     # 使用稠密数据结构开辟每个格网中粒子总数的存储空间
        grid_node.place(self.grid_particles_num)
        cell_index = ti.k if self.dim == 2 else ti.l        # 建立粒子索引变量
        cell_node = grid_node.dense(cell_index, self.particle_max_num_per_cell)     # 使用稠密数据结构开辟每个格网中存储粒子编号的存储空间
        cell_node.place(self.grid_particles)

        # Create rangeary particles
        self.gen_rangeary_particles()


    ###########################################################################
    # 增加单个粒子，或者说第p个粒子，2/3d通用
    @ti.func
    def add_particle(self, p, x, v, material, color):
        self.x[p] = x
        self.material[p] = material
        self.color[p] = color
        self.v[p] = v

    # 增加一群粒子，2/3d通用
    @ti.kernel
    def add_particles(self, new_particles_num: int,
                      new_particles_positions: ti.ext_arr(),
                      new_particles_value: ti.ext_arr(),
                      new_particles_material: ti.ext_arr(),
                      new_particles_color: ti.ext_arr()):
        for p in range(self.particle_num[None],
                       self.particle_num[None] + new_particles_num):
            x = ti.Vector.zero(float, self.dim)
            for d in ti.static(range(self.dim)):
                x[d] = new_particles_positions[p - self.particle_num[None], d]
            self.add_particle(p, x,
                new_particles_value[p - self.particle_num[None]],
                new_particles_material[p - self.particle_num[None]],
                new_particles_color[p - self.particle_num[None]])
        self.particle_num[None] += new_particles_num

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
            # Skip rangeary particles
            if self.material[p_i] == self.material_dummy:
                continue
            center_cell = self.pos_to_index(self.x[p_i])
            cnt = 0
            offset_check = 0
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
                # assert offset_check > 9, 'My Error: offset loop die for endless in NS!'
                if offset_check > 9:
                    print('!!!!My warning: offset loop die for endless in NS!')
                    break
                offset_check += 1   # -------------------------
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
                        cnt += 1
            self.particle_neighbors_num[p_i] = cnt

    ###########################################################################
    # 根据当前的粒子位置，初始化粒子系统
    def initialize_particle_system(self):
        self.grid_particles_num.fill(0)
        self.particle_neighbors.fill(-1)
        self.allocate_particles_to_grid()
        self.search_neighbors()

    ###########################################################################
    # transfer data from ti to np
    # 数据交换至numpy方法：向量数据
    @ti.kernel
    def copy_to_numpy_nd(self, np_arr: ti.ext_arr(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            for j in ti.static(range(self.dim)):
                np_arr[i, j] = src_arr[i][j]

    # 数据交换至numpy方法：标量数据
    @ti.kernel
    def copy_to_numpy(self, np_arr: ti.ext_arr(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            np_arr[i] = src_arr[i]

    # 数据交换至numpy
    def dump(self):
        np_x = np.ndarray((self.particle_num[None], self.dim), dtype=np.float32)
        self.copy_to_numpy_nd(np_x, self.x)

        np_material = np.ndarray((self.particle_num[None],), dtype=np.int32)
        self.copy_to_numpy(np_material, self.material)

        np_color = np.ndarray((self.particle_num[None],), dtype=np.int32)
        self.copy_to_numpy(np_color, self.color)

        np_value = np.ndarray((self.particle_num[None],), dtype=np.float32)
        self.copy_to_numpy(np_value, self.v)

        return {
            'position': np_x,
            'value': np_value,
            'material': np_material,
            'color': np_color
        }

    ###########################################################################
    # 增加 padding region 中所有方向上矩形边界的粒子，2d
    def gen_one_rangeary_cube(self, dl, tr, color, type, voff):
        self.add_cube(lower_corner=dl,
                      cube_size=tr - dl,
                      material=type,
                      color=color,
                      offset=voff,
                      flag_print=False)

    def gen_rangeary_particles(self):
        Dummy_color = 0x9999FF
        Dummy_type = 10
        Dummy_off = self.particle_diameter
        Dummy_cube_d_dl = np.array([i + self.grid_size - self.support_radius for i in self.bound[0]])
        Dummy_cube_d_tr = np.array([self.bound[1][0] - self.grid_size + self.support_radius, 0])
        Dummy_cube_u_dl = np.array([self.bound[0][0] + self.grid_size - self.support_radius, self.bound[1][1] - self.grid_size])
        Dummy_cube_u_tr = np.array([i - self.grid_size + self.support_radius for i in self.bound[1]])
        Dummy_cube_l_dl = np.array([self.bound[0][0] + self.grid_size - self.support_radius, 0])
        Dummy_cube_l_tr = np.array([self.bound[0][0] + self.grid_size, self.bound[1][1] - self.grid_size])
        Dummy_cube_r_dl = np.array([self.bound[1][0] - self.grid_size, 0])
        Dummy_cube_r_tr = np.array([self.bound[1][0] - self.grid_size + self.support_radius, self.bound[1][1] - self.grid_size])
        self.gen_one_rangeary_cube(Dummy_cube_d_dl, Dummy_cube_d_tr, Dummy_color, Dummy_type, Dummy_off)
        self.gen_one_rangeary_cube(Dummy_cube_u_dl, Dummy_cube_u_tr, Dummy_color, Dummy_type, Dummy_off)
        self.gen_one_rangeary_cube(Dummy_cube_l_dl, Dummy_cube_l_tr, Dummy_color, Dummy_type, Dummy_off)
        self.gen_one_rangeary_cube(Dummy_cube_r_dl, Dummy_cube_r_tr, Dummy_color, Dummy_type, Dummy_off)
        print("rangeary dummy particles' number: ", self.particle_num)

    ###########################################################################
    # 增加一个cube区域的粒子，2/3d通用。
    # 目前矩形的左下角会自动加一个半径
    def add_cube(self,
                 lower_corner,
                 cube_size,
                 material,
                 color=0xFFFFFF,
                 value=None,
                 offset=None,
                 flag_print=True):

        num_dim = []
        range_offset = offset if offset is not None else self.particle_diameter
        for i in range(self.dim):
            num_dim.append(np.arange(lower_corner[i] + self.particle_radius, lower_corner[i] + cube_size[i] + 1e-5, range_offset))
        num_new_particles = reduce(lambda x, y: x * y, [len(n) for n in num_dim])
        assert self.particle_num[None] + num_new_particles <= self.particle_max_num, 'My Error: exceed the maximum number of particles!'

        new_positions = np.array(np.meshgrid(*num_dim, sparse=False, indexing='ij' if self.dim == 2 else 'ijk'), dtype=np.float32)
        new_positions = new_positions.reshape(-1, reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        if flag_print:
            print("New cube's number and dim: ", new_positions.shape)

        materials = np.full_like(np.zeros(num_new_particles), material)
        colors = np.full_like(np.zeros(num_new_particles), color)
        value = np.full_like(np.zeros(num_new_particles), value if value is not None else 0.0)
        self.add_particles(num_new_particles, new_positions, value, materials, colors)
