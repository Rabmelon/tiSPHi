import taichi as ti
import numpy as np
from functools import reduce    # 整数：累加；字符串、列表、元组：拼接。

@ti.data_oriented
class ParticleSystem:
    def __init__(self, res):
        self.res = res
        self.dim = len(res)
        assert self.dim > 1 & self.dim < 4
        self.screen_to_world_ratio = 50
        self.bound = np.array(res) / self.screen_to_world_ratio     # 这个应该是一个绘图用的边界？具体指代什么呢？

        # Material
        self.material_dummy = 0
        self.material_fluid = 1

        self.particle_radius = 0.05  # particle radius
        self.particle_diameter = 2 * self.particle_radius
        self.support_radius = self.particle_radius * 4.0  # support radius
        self.m_V = 0.8 * self.particle_diameter ** self.dim     # 这里是只适用于2d的情况，pi/4≈0.8，3d为3π/32≈0.3
        self.particle_max_num = 2 ** 15             # 粒子上限数目2^15个
        self.particle_max_num_per_cell = 100        # 每格网最多100个
        self.particle_max_num_neighbor = 100        # 每个粒子的neighbour粒子最多100个
        self.particle_num = ti.field(int, shape=())     # 记录粒子总数

        # Grid related properties
        self.grid_size = self.support_radius        # 令格网边长为支持域半径，即4倍粒子半径。似乎可优化为加一个微小量
        self.grid_num = np.ceil(np.array(res) / self.grid_size).astype(int)     # 格网总数？
        self.grid_particles_num = ti.field(int)     # 格网中的粒子总数？
        self.grid_particles = ti.field(int)         # 格网中的粒子编号？
        self.padding = self.grid_size               # padding是什么用途？用在enforce_boundary函数中

        # Particle related properties
        self.x = ti.Vector.field(self.dim, dtype=float)     # 粒子的位置
        self.v = ti.Vector.field(self.dim, dtype=float)     # 粒子的速度
        self.density = ti.field(dtype=float)                # 粒子的密度
        self.pressure = ti.field(dtype=float)               # 粒子所受压力？
        self.material = ti.field(dtype=int)                 # 粒子的材料类型
        self.color = ti.field(dtype=int)                    # 粒子的绘制颜色
        self.particle_neighbors = ti.field(int)             # 粒子的邻域粒子编号？
        self.particle_neighbors_num = ti.field(int)         # 粒子的邻域粒子总数目

        self.particles_node = ti.root.dense(ti.i, self.particle_max_num)    # 使用稠密数据结构开辟粒子存储空间？？？？？
        self.particles_node.place(self.x, self.v, self.density, self.pressure, self.material, self.color)
        self.particles_node.place(self.particle_neighbors_num)
        self.particle_node = self.particles_node.dense(ti.j, self.particle_max_num_neighbor)    # 使用稠密数据结构开辟每个粒子的邻域粒子编号的存储空间？？？？？
        self.particle_node.place(self.particle_neighbors)

        index = ti.ij if self.dim == 2 else ti.ijk          # 建立格网索引变量
        grid_node = ti.root.dense(index, self.grid_num)     # 使用稠密数据结构开辟所有格网中存储最多粒子所需的空间？？？？？
        grid_node.place(self.grid_particles_num)

        cell_index = ti.k if self.dim == 2 else ti.l        # 建立粒子索引变量
        cell_node = grid_node.dense(cell_index, self.particle_max_num_per_cell)     # 使用稠密数据结构开辟每个格网中存储粒子所需的空间？？？？？
        cell_node.place(self.grid_particles)

    @ti.func
    def add_particle(self, p, x, v, density, pressure, material, color):
        self.x[p] = x
        self.v[p] = v
        self.density[p] = density
        self.pressure[p] = pressure
        self.material[p] = material
        self.color[p] = color

    @ti.kernel
    def add_particles(self, new_particles_num: int,
                      new_particles_positions: ti.ext_arr(),
                      new_particles_velocity: ti.ext_arr(),
                      new_particle_density: ti.ext_arr(),
                      new_particle_pressure: ti.ext_arr(),
                      new_particles_material: ti.ext_arr(),
                      new_particles_color: ti.ext_arr()):
        for p in range(self.particle_num[None], self.particle_num[None] + new_particles_num):
            v = ti.Vector.zero(float, self.dim)
            x = ti.Vector.zero(float, self.dim)
            for d in ti.static(range(self.dim)):
                v[d] = new_particles_velocity[p - self.particle_num[None], d]
                x[d] = new_particles_positions[p - self.particle_num[None], d]
            self.add_particle(p, x, v,
                              new_particle_density[p - self.particle_num[None]],
                              new_particle_pressure[p - self.particle_num[None]],
                              new_particles_material[p - self.particle_num[None]],
                              new_particles_color[p - self.particle_num[None]])
        self.particle_num[None] += new_particles_num

    @ti.func
    def pos_to_index(self, pos):
        return (pos / self.grid_size).cast(int)

    @ti.func
    def is_valid_cell(self, cell):
        # Check whether the cell is in the grid
        flag = True
        for d in ti.static(range(self.dim)):
            flag = flag and (0 <= cell[d] < self.grid_num[d])
        return flag

    @ti.kernel
    def allocate_particles_to_grid(self):
        for p in range(self.particle_num[None]):
            cell = self.pos_to_index(self.x[p])
            offset = self.grid_particles_num[cell].atomic_add(1)
            self.grid_particles[cell, offset] = p

    @ti.kernel
    def search_neighbors(self):
        for p_i in range(self.particle_num[None]):
            # Skip boundary particles
            if self.material[p_i] == self.material_dummy:
                continue
            center_cell = self.pos_to_index(self.x[p_i])
            cnt = 0
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
                if cnt >= self.particle_max_num_neighbor:
                    break
                cell = center_cell + offset
                if not self.is_valid_cell(cell):
                    break
                for j in range(self.grid_particles_num[cell]):
                    p_j = self.grid_particles[cell, j]
                    distance = (self.x[p_i] - self.x[p_j]).norm()
                    if p_i != p_j and distance < self.support_radius:
                        self.particle_neighbors[p_i, cnt] = p_j
                        cnt += 1
            self.particle_neighbors_num[p_i] = cnt

    def initialize_particle_system(self):
        self.grid_particles_num.fill(0)
        self.particle_neighbors.fill(-1)
        self.allocate_particles_to_grid()
        self.search_neighbors()

    @ti.kernel
    def copy_to_numpy_nd(self, np_arr: ti.ext_arr(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            for j in ti.static(range(self.dim)):
                np_arr[i, j] = src_arr[i][j]

    @ti.kernel
    def copy_to_numpy(self, np_arr: ti.ext_arr(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            np_arr[i] = src_arr[i]

    def dump(self):
        np_x = np.ndarray((self.particle_num[None], self.dim), dtype=np.float32)
        self.copy_to_numpy_nd(np_x, self.x)

        np_v = np.ndarray((self.particle_num[None], self.dim), dtype=np.float32)
        self.copy_to_numpy_nd(np_v, self.v)

        np_material = np.ndarray((self.particle_num[None],), dtype=np.int32)
        self.copy_to_numpy(np_material, self.material)

        np_color = np.ndarray((self.particle_num[None],), dtype=np.int32)
        self.copy_to_numpy(np_color, self.color)

        return {
            'position': np_x,
            'velocity': np_v,
            'material': np_material,
            'color': np_color
        }

    def add_cube(self,
                 lower_corner,
                 cube_size,
                 material,
                 color=0xFFFFFF,
                 density=None,
                 pressure=None,
                 velocity=None):

        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(lower_corner[i], lower_corner[i] + cube_size[i],
                          self.particle_radius))
        num_new_particles = reduce(lambda x, y: x * y,
                                   [len(n) for n in num_dim])
        assert self.particle_num[
                   None] + num_new_particles <= self.particle_max_num

        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)
        new_positions = new_positions.reshape(-1,
                                              reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        print("new position shape ", new_positions.shape)
        if velocity is None:
            velocity = np.full_like(new_positions, 0)
        else:
            velocity = np.array([velocity for _ in range(num_new_particles)], dtype=np.float32)

        material = np.full_like(np.zeros(num_new_particles), material)
        color = np.full_like(np.zeros(num_new_particles), color)
        density = np.full_like(np.zeros(num_new_particles), density if density is not None else 1000.)
        pressure = np.full_like(np.zeros(num_new_particles), pressure if pressure is not None else 0.)
        self.add_particles(num_new_particles, new_positions, velocity, density, pressure, material, color)