import taichi as ti
import numpy as np
from functools import reduce    # 整数：累加；字符串、列表、元组：拼接。lambda为使用匿名函数
ti.init(arch=ti.cpu)


@ti.data_oriented
class ParticleSystem:
    def __init__(self, world, radius):

        # Basic information of the simulation 模拟世界矩形范围信息
        self.dim = len(world)
        assert self.dim in (2, 3), "SPH solver supports only 2D and 3D particle system."
        self.bound = np.array(world)    # Simply create a rectangular bound

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
        self.grid_num = np.ceil(self.bound / self.grid_size).astype(int)  # 格网总数
        self.grid_particles_num = ti.field(int)  # 每个格网中的粒子总数
        self.grid_particles = ti.field(int)  # 每个格网中的粒子编号
        self.padding = self.grid_size  # 边界padding, 用在enforce_boundary函数中

        # Particle related property 粒子携带的属性信息
        # Basic
        self.particle_neighbors_num = ti.field(int)         # total number of neighbour particles
        self.particle_neighbors = ti.field(int)             # index of neighbour particles
        self.material = ti.field(dtype=int)                 # material type
        self.color = ti.field(dtype=int)                    # color in drawing
        # Values
        self.x = ti.Vector.field(self.dim, dtype=float)     # position

        # Place nodes on root
        self.particles_node = ti.root.dense(ti.i, self.particle_max_num)    # 使用稠密数据结构开辟每个粒子数据的存储空间，按列存储
        self.particles_node.place(self.x, self.material, self.color)
        self.particles_node.place(self.particle_neighbors_num)
        self.particle_node = self.particles_node.dense(ti.j, self.particle_max_num_neighbors)    # 使用稠密数据结构开辟每个粒子邻域粒子编号的存储空间，按行存储
        self.particle_node.place(self.particle_neighbors)

        grid_index = ti.ij if self.dim == 2 else ti.ijk          # 建立格网维度索引变量，xy or xyz
        grid_node = ti.root.dense(grid_index, self.grid_num)     # 使用稠密数据结构开辟每个格网中粒子总数的存储空间
        grid_node.place(self.grid_particles_num)
        cell_index = ti.k if self.dim == 2 else ti.l        # 建立粒子索引变量
        cell_node = grid_node.dense(cell_index, self.particle_max_num_per_cell)     # 使用稠密数据结构开辟每个格网中存储粒子编号的存储空间
        cell_node.place(self.grid_particles)


    @ti.kernel
    def add_particles(self, new_particles_num: int):
        for p in range(self.particle_num[None], self.particle_num[None] + new_particles_num):
            print(p)
        self.particle_num[None] += new_particles_num
        print(self.particle_num[None])

    def add_cube(self):
        num_new_particles = 12
        assert self.particle_num[None] + num_new_particles <= self.particle_max_num, 'My Error: exceed the maximum number of particles!'
        self.add_particles(num_new_particles)

world = (150, 100)
particle_radius = 0.5
case = ParticleSystem(world, particle_radius)
case.add_cube()