import taichi as ti
import numpy as np
from functools import reduce    # 整数：累加；字符串、列表、元组：拼接。lambda为使用匿名函数

@ti.data_oriented
class ParticleSystem:
    def __init__(self, world, radius, kh):
        print("Hallo, class Particle System starts to serve!")

        # Basic information of the simulation
        self.dim = len(world)
        assert self.dim in (2, 3), "SPH solver supports only 2D simulations."
        self.dim_stress = 4 if self.dim == 2 else 9     # Temporary 9 for 3d
        self.bound = np.array(world)

        # Material 材料类型定义
        self.material_boundary = 0
        self.material_water = 1
        self.material_soil = 2
        self.material_rigid = 3
        self.material_solid_e = 4
        self.material_solid_p = 5

        # Basic particle property 粒子的基本属性
        self.particle_radius = radius
        self.particle_diameter = 2.0 * self.particle_radius
        self.support_radius = kh * self.particle_radius
        self.m_V = self.particle_diameter**self.dim        # m2 or m3
        # self.m_V = (np.pi / 4.0 if self.dim == 2 else 3 * np.pi / 32) * self.particle_diameter**self.dim  # 2d为pi/4≈0.8，3d为3π/32≈0.3     然而这会导致体积丧失？？？
        self.particle_max_num = 2**16  # 粒子上限数目，65536
        self.particle_max_num_per_cell = 100  # 每格网最多100个
        self.particle_max_num_neighbor = 100  # 每个粒子的neighbour粒子最多100个
        self.particle_num = ti.field(int, shape=())  # 记录当前的粒子总数

        # Grid property 背景格网的基本属性
        self.grid_size = 2 * self.support_radius  # 令格网边长为2倍的支持域半径，这样只需遍历4个grid就可以获取邻域粒子【不好使！】
        # self.grid_size = self.support_radius + 1e-5 # 支持域半径加一个微小量
        self.grid_num = np.ceil(np.array(world) / self.grid_size).astype(int)  # 格网总数？
        self.grid_particles_num = ti.field(int)  # 格网中的粒子总数？
        self.grid_particles = ti.field(int)  # 格网中的粒子编号？
        self.padding = self.grid_size  # padding是什么用途？用在enforce_boundary函数中

        # Particle related property 粒子携带的属性信息
        self.x = ti.Vector.field(self.dim, dtype=float)     # 粒子的位置
        self.v = ti.Vector.field(self.dim, dtype=float)     # 粒子的速度
        self.density = ti.field(dtype=float)                # 粒子的密度
        self.stress = ti.Vector.field(self.dim_stress, dtype=float)       # 粒子的应力项
        self.strain_p = ti.Vector.field(self.dim_stress, dtype=float)     # 粒子的应变项
        self.pressure = ti.field(dtype=float)               # 粒子的压力项
        self.material = ti.field(dtype=int)                 # 粒子的材料类型
        self.color = ti.field(dtype=int)                    # 粒子的绘制颜色
        self.particle_neighbors = ti.field(int)             # 粒子的邻域粒子编号？
        self.particle_neighbors_num = ti.field(int)         # 粒子的邻域粒子总数目

        # New memory space?
        self.particles_node = ti.root.dense(ti.i, self.particle_max_num)    # 使用稠密数据结构开辟粒子中每个数据的存储空间？？？？？
        self.particles_node.place(self.x, self.v, self.density, self.stress, self.strain_p, self.pressure, self.material, self.color)
        self.particles_node.place(self.particle_neighbors_num)
        self.particle_node = self.particles_node.dense(ti.j, self.particle_max_num_neighbor)    # 使用稠密数据结构开辟每个粒子的存储空间？？？？？是否相当于挂一列粒子，每一个粒子的各项数据用行存储？
        self.particle_node.place(self.particle_neighbors)

        index = ti.ij if self.dim == 2 else ti.ijk          # 建立格网维度索引变量，xy or xyz
        grid_node = ti.root.dense(index, self.grid_num)     # 使用稠密数据结构开辟所有格网中存储最多粒子所需的空间？？？？？
        grid_node.place(self.grid_particles_num)

        cell_index = ti.k if self.dim == 2 else ti.l        # 建立粒子索引变量
        cell_node = grid_node.dense(cell_index, self.particle_max_num_per_cell)     # 使用稠密数据结构开辟每个格网中存储粒子所需的空间？？？？？
        cell_node.place(self.grid_particles)

        # Create boundary particles
        self.gen_boundary_particles()

    # 增加单个粒子，或者说第p个粒子，2/3d通用
    @ti.func
    def add_particle(self, p, x, v, density, stress, strain_p, pressure, material, color):
        self.x[p] = x
        self.v[p] = v
        self.density[p] = density
        self.stress[p] = stress
        self.strain_p[p] = strain_p
        self.pressure[p] = pressure
        self.material[p] = material
        self.color[p] = color

    # 增加一群粒子，2/3d通用
    @ti.kernel
    def add_particles(self, new_particles_num: int,
                      new_particles_positions: ti.ext_arr(),
                      new_particles_velocity: ti.ext_arr(),
                      new_particles_density: ti.ext_arr(),
                      new_particles_stress: ti.ext_arr(),
                      new_particles_strain_p: ti.ext_arr(),
                      new_particles_pressure: ti.ext_arr(),
                      new_particles_material: ti.ext_arr(),
                      new_particles_color: ti.ext_arr()):
        for p in range(self.particle_num[None],
                       self.particle_num[None] + new_particles_num):
            x = ti.Vector.zero(float, self.dim)
            v = ti.Vector.zero(float, self.dim)
            stress = ti.Vector.zero(float, self.dim_stress)
            strain_p = ti.Vector.zero(float, self.dim_stress)
            for d in ti.static(range(self.dim)):
                x[d] = new_particles_positions[p - self.particle_num[None], d]
                v[d] = new_particles_velocity[p - self.particle_num[None], d]
            for d in ti.static(range(self.dim_stress)):
                stress[d] = new_particles_stress[p - self.particle_num[None], d]
                strain_p[d] = new_particles_strain_p[p - self.particle_num[None], d]
            self.add_particle(
                p, x, v, new_particles_density[p - self.particle_num[None]], stress, strain_p,
                new_particles_pressure[p - self.particle_num[None]],
                new_particles_material[p - self.particle_num[None]],
                new_particles_color[p - self.particle_num[None]])
        self.particle_num[None] += new_particles_num

    # 获取粒子位置对应的grid编号
    @ti.func
    def pos_to_index(self, pos):
        return (pos / self.grid_size).cast(int)

    # 检查cell的编号是否有效，即是否在grid内
    @ti.func
    def is_valid_cell(self, cell):
        # Check whether the cell is in the grid
        flag = True
        for d in ti.static(range(self.dim)):
            flag = flag and (0 <= cell[d] < self.grid_num[d])
        return flag

    # 计算每个粒子对应的grid编号？？？并将粒子编号加入到对应的grid的链表中？？？
    @ti.kernel
    def allocate_particles_to_grid(self):
        for p in range(self.particle_num[None]):
            cell = self.pos_to_index(self.x[p])                     # 当前粒子位于哪个grid
            offset = self.grid_particles_num[cell].atomic_add(1)    # 当前粒子是这个grid中的第几个粒子
            self.grid_particles[cell, offset] = p

    # 搜索邻域粒子，使用的应该是常规的基于格网的搜索方法
    @ti.kernel
    def search_neighbors(self):
        for p_i in range(self.particle_num[None]):
            # Skip boundary particles
            # 若多介质混合，如何限制搜索条件？【查看一叶扁舟程序学习】
            if self.material[p_i] == self.material_boundary:
                continue
            center_cell = self.pos_to_index(self.x[p_i])
            # print('p', p_i, end=', ')   # -------------------------
            # print('center cell', center_cell, end=', ')   # -------------------------
            # print('NS cell:', end='')   # -------------------------
            cnt = 0
            offset_check = 0
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
                if offset_check > 9:   # -------------------------
                    # print('dieLoop!', end='')   # -------------------------
                    break   # -------------------------
                offset_check += 1   # -------------------------
                if cnt >= self.particle_max_num_neighbor:
                    break
                cell = center_cell + offset
                # print(cell, end='; ')   # -------------------------
                if not self.is_valid_cell(cell):
                    continue        # still be a big problem!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                for j in range(self.grid_particles_num[cell]):
                    p_j = self.grid_particles[cell, j]
                    distance = (self.x[p_i] - self.x[p_j]).norm()
                    if p_i != p_j and distance < self.support_radius:
                        self.particle_neighbors[p_i, cnt] = p_j
                        cnt += 1
            self.particle_neighbors_num[p_i] = cnt
            # print('')   # -------------------------

    # 根据当前的粒子位置，初始化粒子系统
    def initialize_particle_system(self):
        self.grid_particles_num.fill(0)
        self.particle_neighbors.fill(-1)
        self.allocate_particles_to_grid()
        self.search_neighbors()

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

        np_v = np.ndarray((self.particle_num[None], self.dim), dtype=np.float32)
        self.copy_to_numpy_nd(np_v, self.v)

        np_stress = np.ndarray((self.particle_num[None], self.dim_stress), dtype=np.float32)
        self.copy_to_numpy_nd(np_stress, self.stress)

        np_strain_p = np.ndarray((self.particle_num[None], self.dim_stress), dtype=np.float32)
        self.copy_to_numpy_nd(np_strain_p, self.strain_p)

        np_material = np.ndarray((self.particle_num[None],), dtype=np.int32)
        self.copy_to_numpy(np_material, self.material)

        np_color = np.ndarray((self.particle_num[None],), dtype=np.int32)
        self.copy_to_numpy(np_color, self.color)

        return {
            'position': np_x,
            'velocity': np_v,
            'stress': np_stress,
            'strain_p': np_strain_p,
            'material': np_material,
            'color': np_color
        }

    # 增加 padding region 中所有方向上矩形边界的粒子，2d
    def gen_boundary_particles(self):
        Bound_color = 0x9999FF
        Bound_cube_d_dl = np.zeros(2) + self.padding - self.support_radius
        Bound_cube_d_tr = np.array([self.bound[0] - self.padding + self.support_radius, self.padding])
        self.add_cube(lower_corner=Bound_cube_d_dl,
                      cube_size=Bound_cube_d_tr - Bound_cube_d_dl,
                      color=Bound_color,
                      material=0,flag_print=False)
        Bound_cube_u_dl = np.array([self.padding - self.support_radius, self.bound[1] - self.padding])
        Bound_cube_u_tr = self.bound - self.padding + self.support_radius
        self.add_cube(lower_corner=Bound_cube_u_dl,
                      cube_size=Bound_cube_u_tr - Bound_cube_u_dl,
                      color=Bound_color,
                      material=0,flag_print=False)
        Bound_cube_l_dl = np.array([self.padding - self.support_radius, self.padding])
        Bound_cube_l_tr = np.array([self.padding, self.bound[1] - self.padding])
        self.add_cube(lower_corner=Bound_cube_l_dl,
                      cube_size=Bound_cube_l_tr - Bound_cube_l_dl,
                      color=Bound_color,
                      material=0,flag_print=False)
        Bound_cube_r_dl = np.array([self.bound[0] - self.padding, self.padding])
        Bound_cube_r_tr = np.array([self.bound[0] - self.padding + self.support_radius, self.bound[1] - self.padding])
        self.add_cube(lower_corner=Bound_cube_r_dl,
                      cube_size=Bound_cube_r_tr - Bound_cube_r_dl,
                      color=Bound_color,
                      material=0,flag_print=False)
        print("Boundary particles' number: ", self.particle_num)


    # 增加一个cube区域的粒子，2/3d通用。
    # 目前矩形的左下角会自动加一个半径！！！
    # 具体实现仍需仔细学习！！！
    def add_cube(self,
                 lower_corner,
                 cube_size,
                 material,
                 color=0xFFFFFF,
                 stress=None,
                 strain_p=None,
                 density=None,
                 pressure=None,
                 velocity=None,
                 flag_print=True):

        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(lower_corner[i] + self.particle_radius,
                          lower_corner[i] + cube_size[i] + 1e-5,
                          self.particle_diameter))
        num_new_particles = reduce(lambda x, y: x * y,
                                   [len(n) for n in num_dim])
        assert self.particle_num[None] + num_new_particles <= self.particle_max_num

        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)
        new_positions = new_positions.reshape(
            -1, reduce(lambda x, y: x * y,
                       list(new_positions.shape[1:]))).transpose()
        if flag_print:
            print("New cube's number and dim: ", new_positions.shape)
        if velocity is None:
            velocity = np.full_like(new_positions, 0)
        else:
            velocity = np.array([velocity for _ in range(num_new_particles)], dtype=np.float32)

        if stress is None:
            stress = np.full_like(new_positions, 0)
        else:
            stress = np.array([stress for _ in range(num_new_particles)], dtype=np.float32)

        if strain_p is None:
            strain_p = np.full_like(new_positions, 0)
        else:
            strain_p = np.array([strain_p for _ in range(num_new_particles)], dtype=np.float32)

        material = np.full_like(np.zeros(num_new_particles), material)
        color = np.full_like(np.zeros(num_new_particles), color)
        density = np.full_like(np.zeros(num_new_particles), density if density is not None else 1000.)
        pressure = np.full_like(np.zeros(num_new_particles), pressure if pressure is not None else 0.)
        self.add_particles(num_new_particles, new_positions, velocity, density, stress, strain_p, pressure, material, color)
