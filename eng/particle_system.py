import taichi as ti
import numpy as np
# from eng.scan_single_buffer import parallel_prefix_sum_inclusive_inplace
from eng.configer_builder import SimConfiger
from eng.colormap import color_map
from eng.particle_func import *


@ti.data_oriented
class ParticleSystem:
    def __init__(self, config: SimConfiger) -> None:
        self.cfg = config
        self.dim = 3 if not self.cfg.get_cfg("is2D") else 2
        self.dim3 = 3
        self.domain_start = np.array(self.cfg.get_cfg("domainStart"))
        self.domain_end = np.array(self.cfg.get_cfg("domainEnd"))
        self.domain_size = self.domain_end - self.domain_start
        self.color_title = self.cfg.get_cfg("colorTitle")
        self.flag_boundary = self.cfg.get_cfg("boundary")   # 0: none, 1: collision, 2: dummy particles, 3: dummy + repulsive particles
        self.show_bdy = self.cfg.get_cfg("showBdyPts")
        self.mat_dummy_type, self.mat_rep_type = 0, -1
        self.mat_fluid_type, self.mat_soil_type, self.mat_rigid_type = 1, 2, 11
        self.bdy_none, self.bdy_collision, self.bdy_dummy, self.bdy_rep, self.bdy_dummy_rep = 0, 1, 2, 3, 4

        ##############################################
        # Basics
        ##############################################
        # Discretization info
        self.particle_radius = self.cfg.get_cfg("particleRadius")
        self.kappa = self.cfg.get_cfg("kappa")
        self.kh = self.cfg.get_cfg("kh")
        self.particle_diameter = 2 * self.particle_radius
        self.smoothing_len = self.kh * self.particle_diameter
        self.support_radius = self.kappa * self.smoothing_len
        self.m_V0 = self.particle_diameter ** self.dim
        self.particle_num = ti.field(int, shape=())

        # info of whole particle systems
        self.vmax = ti.field(float, shape=())
        self.vmin = ti.field(float, shape=())

        # Grid related properties
        # self.grid_size = self.support_radius
        self.grid_size = ti.ceil(self.kappa * self.kh) * self.particle_diameter
        self.vdomain_start = self.domain_start - self.grid_size
        self.vdomain_end = self.domain_end + self.grid_size
        self.vdomain_size = self.vdomain_end - self.vdomain_start
        if self.dim == 2:
            self.domain_end[2] = self.domain_start[2] + self.particle_diameter
            self.domain_size[2] = self.particle_diameter
            self.vdomain_start[2] = self.domain_start[2]
            self.vdomain_end[2] = self.vdomain_start[2] + self.particle_diameter
            self.vdomain_size[2] = self.particle_diameter
        self.grid_num = np.ceil(self.vdomain_size / self.grid_size).astype(int)
        self.grid_num_total = np.prod(self.grid_num[0:self.dim]).item()
        print("Grid num:", [self.grid_num[i] for i in range(self.dim)], "total:", self.grid_num_total)

        # All objects id and its particle num
        self.object_collection = dict()
        self.object_id_rigid_body = set()

        # Materials
        self.mat_index, self.mat_fluid, self.mat_soil, self.mat_rigid = set_material(self)

        ##############################################
        # Compute number of particles
        ##############################################
        # Process Fluid Blocks
        self.fluid_blocks = self.cfg.get_fluid_blocks()
        fluid_particle_num = 0
        for fluid in self.fluid_blocks:
            chk_block_in_domain(self.domain_start, self.domain_end, fluid["translation"], fluid["size"], self.dim)
            particle_num, _ = calc_cube_particle_num(fluid["translation"], fluid["size"], self.dim, offset=self.particle_diameter)
            fluid["particleNum"] = particle_num
            self.object_collection[fluid["objectId"]] = fluid
            fluid_particle_num += particle_num
            print("Fluid block %d particle number: %d" % (fluid["objectId"], particle_num))

        # Process Fluid Bodies
        self.fluid_bodies = self.cfg.get_fluid_bodies()
        for fluid_body in self.fluid_bodies:
            voxelized_points_np = load_body(fluid_body, self.particle_diameter)
            fluid_body["particleNum"] = voxelized_points_np.shape[0]
            fluid_body["voxelizedPoints"] = voxelized_points_np
            self.object_collection[fluid_body["objectId"]] = fluid_body
            fluid_particle_num += voxelized_points_np.shape[0]
            print("Fluid body %d particle number: %d" % (fluid_body["objectId"], voxelized_points_np.shape[0]))

        # Process Rigid Blocks
        self.rigid_blocks = self.cfg.get_rigid_blocks()
        rigid_particle_num = 0
        for rigid in self.rigid_blocks:
            chk_block_in_domain(self.domain_start, self.domain_end, rigid["translation"], rigid["size"], self.dim)
            particle_num, _ = calc_cube_particle_num(rigid["translation"], rigid["size"], self.dim, offset=self.particle_diameter)
            rigid["particleNum"] = particle_num
            self.object_collection[rigid["objectId"]] = rigid
            rigid_particle_num += particle_num
            print("Rigid block %d particle number: %d" % (rigid["objectId"], particle_num))

        # Process Rigid Bodies
        self.rigid_bodies = self.cfg.get_rigid_bodies()
        for rigid_body in self.rigid_bodies:
            voxelized_points_np = load_body(rigid_body, self.particle_diameter)
            rigid_body["particleNum"] = voxelized_points_np.shape[0]
            rigid_body["voxelizedPoints"] = voxelized_points_np
            self.object_collection[rigid_body["objectId"]] = rigid_body
            rigid_particle_num += voxelized_points_np.shape[0]
            print("Rigid body %d particle number: %d" % (rigid_body["objectId"], voxelized_points_np.shape[0]))

        # Check dummy particle number
        dummy_particle_num = 0
        if self.flag_boundary == self.bdy_dummy or self.flag_boundary == self.bdy_dummy_rep:
            self.dummy_boundary = calc_dummy_boundary(self.dim, self.domain_start, self.domain_end, self.vdomain_start, self.vdomain_end)
            dummy_particle_num = count_boundary(self.dummy_boundary, self.dim, self.particle_diameter)
            print("Dummy particle number: %d" % (dummy_particle_num))

        # Check repulsive particle number
        rep_particle_num = 0
        if self.flag_boundary == self.bdy_rep or self.flag_boundary == self.bdy_dummy_rep:
            self.rep_boundary = calc_rep_boundary(self.dim, self.domain_start, self.domain_end, self.particle_radius)
            rep_particle_num = count_boundary(self.rep_boundary, self.dim, self.particle_radius)
            print("Repulsive particle number: %d" % (rep_particle_num))

        self.particle_max_num = fluid_particle_num + rigid_particle_num + dummy_particle_num + rep_particle_num
        print(f"Particle total num: {self.particle_max_num}")

        ##############################################
        # Allocate memory
        ##############################################
        self.pt = Particle.field(shape=self.particle_max_num)
        self.pt_buf = Particle.field(shape=self.particle_max_num)
        self.pt_empty = Particle.field(shape=())

        # Particle num of each grid
        self.grid_particle_num = ti.field(int, shape=self.grid_num_total)
        self.grid_particle_num_temp = ti.field(int, shape=self.grid_num_total)
        # grid_node = ti.root.dense(ti.i, self.grid_num_total)
        # grid_node.place(self.grid_particle_num, self.grid_particle_num_temp)

        # rigid body
        self.rigid_rest_cm = type_vec3f.field(shape=ti.max(len(self.rigid_blocks)+len(self.rigid_bodies), 1))

        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.grid_particle_num.shape[0])

        ##############################################
        # Initialize particles
        ##############################################
        self.initialize_particles()
        self.set_id0()

        print("Particle system construction complete!")


    def initialize_particles(self):
        # Fluid block
        for fluid in self.fluid_blocks:
            obj_id = fluid["objectId"]
            start = np.array(fluid["translation"])
            size = np.array(fluid["size"])
            velocity = fluid["velocity"]
            mat_Id = fluid["materialId"]
            mat_fluid = get_material(self, mat_Id)
            mat_type = mat_fluid["matType"]
            density = mat_fluid["density0"]
            color = np.array([ic / 255 for ic in mat_fluid["color"]], dtype=np.float32)
            add_cube(self,
                object_id=obj_id,
                lower_corner=start,
                cube_size=size,
                velocity=velocity,
                density=density,
                is_dynamic=1,  # enforce dynamic
                color=color,
                mat_id=mat_Id,
                mat_type=mat_type)

        # Fluid bodies

        # Rigid blocks
        for rigid in self.rigid_blocks:
            obj_id = rigid["objectId"]
            self.object_id_rigid_body.add(obj_id)
            num_particles_obj = rigid["particleNum"]
            start = np.array(rigid["translation"])
            size = np.array(rigid["size"])
            velocity = rigid["velocity"]
            is_dynamic = rigid["isDynamic"]
            mat_Id = rigid["materialId"]
            mat_rigid = get_material(self, mat_Id)
            mat_type = mat_rigid["matType"]
            density = mat_rigid["density0"]
            color = np.array([ic / 255 for ic in mat_rigid["color"]], dtype=np.float32)
            add_cube(self,
                object_id=obj_id,
                lower_corner=start,
                cube_size=size,
                velocity=velocity,
                density=density,
                is_dynamic=is_dynamic,
                color=color,
                mat_id=mat_Id,
                mat_type=mat_type)

        # Rigid bodies
        for rigid_body in self.rigid_bodies:
            obj_id = rigid_body["objectId"]
            self.object_id_rigid_body.add(obj_id)
            num_particles_obj = rigid_body["particleNum"]
            voxelized_points_np = rigid_body["voxelizedPoints"]
            is_dynamic = rigid_body["isDynamic"]
            velocity = np.array(rigid_body["velocity"], dtype=np.float64)
            mat_Id = rigid_body["materialId"]
            mat_rigid = get_material(self, mat_Id)
            mat_type = mat_rigid["matType"]
            density = mat_rigid["density0"]
            color = np.array([ic / 255 for ic in mat_rigid["color"]], dtype=np.float32)
            add_particles(self,
                obj_id, num_particles_obj,
                np.array(voxelized_points_np, dtype=np.float64),  # position
                np.stack([velocity for _ in range(num_particles_obj)]),  # velocity
                density * np.ones(num_particles_obj, dtype=np.float64),  # density
                np.zeros(num_particles_obj, dtype=np.float64),  # pressure
                np.array([mat_Id for _ in range(num_particles_obj)], dtype=np.int32),  # material id is solid
                np.array([mat_type for _ in range(num_particles_obj)], dtype=np.int32),  # material type is solid
                is_dynamic * np.ones(num_particles_obj, dtype=np.int32),  # is_dynamic
                np.stack([color for _ in range(num_particles_obj)]))  # color

        # Boundary particles
        if self.flag_boundary == self.bdy_dummy or self.flag_boundary == self.bdy_dummy_rep:
            add_boundary(self, self.dummy_boundary, self.mat_dummy_type, color=[153, 153, 255])
        if self.flag_boundary == self.bdy_rep or self.flag_boundary == self.bdy_dummy_rep:
            add_boundary(self, self.rep_boundary, self.mat_rep_type, offset=self.particle_radius, color=[170, 17, 255])


    @ti.kernel
    def clear_particles(self):
        for i in range(self.particle_num[None]):
            self.pt[i] = self.pt_empty[None]

    @ti.kernel
    def set_id0(self):
        for i in range(self.particle_num[None]):
            self.pt[i].id0 = i

    ##############################################
    # Neighbour search
    ##############################################
    @ti.func
    def pos_to_index(self, pos):
        return ((pos - ti.Vector(self.vdomain_start)) / self.grid_size).cast(int)

    @ti.func
    def flatten_grid_index(self, grid_index):
        # res = -1
        # if self.dim == 3:
        #     res = grid_index[0] * self.grid_num[1] * self.grid_num[2] + grid_index[1] * self.grid_num[2] + grid_index[2]
        # elif self.dim == 2:
        #     res = grid_index[0] * self.grid_num[1] + grid_index[1]
        # return res
        return grid_index[0] * self.grid_num[1] * self.grid_num[2] + grid_index[1] * self.grid_num[2] + grid_index[2]

    @ti.func
    def get_flatten_grid_index(self, pos):
        return self.flatten_grid_index(self.pos_to_index(pos))

    @ti.kernel
    def update_grid_id(self):
        self.grid_particle_num.fill(0)
        for I in ti.grouped(self.pt):
            grid_index = self.get_flatten_grid_index(self.pt[I].x)
            self.pt[I].grid_ids = grid_index
            ti.atomic_add(self.grid_particle_num[grid_index], 1)
        for I in ti.grouped(self.grid_particle_num):
            self.grid_particle_num_temp[I] = self.grid_particle_num[I]

    @ti.kernel
    def counting_sort(self):
        for i in range(self.particle_num[None]):
            I = self.particle_num[None] - 1 - i
            base_offset = 0
            if self.pt[I].grid_ids - 1 >= 0:
                base_offset = self.grid_particle_num[self.pt[I].grid_ids - 1]
            self.pt[I].id_new = ti.atomic_sub(self.grid_particle_num_temp[self.pt[I].grid_ids], 1) - 1 + base_offset

        for i in range(self.particle_num[None]):
            new_index = self.pt[i].id_new
            self.pt_buf[new_index] = self.pt[i]

        for i in range(self.particle_num[None]):
            self.pt[i] = self.pt_buf[i]

    def initialize_particle_system(self):
        self.update_grid_id()
        self.prefix_sum_executor.run(self.grid_particle_num)
        # FIXME: change to taichi built-in prefix_sum_inclusive_inplace after next Taichi release i.e., 1.1.4
        # parallel_prefix_sum_inclusive_inplace(self.grid_particle_num, self.grid_particle_num.shape[0])
        self.counting_sort()

    @ti.func
    def for_all_neighbors(self, i, task: ti.template(), ret: ti.template()):
        center_cell = self.pos_to_index(self.pt[i].x)
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim3)):
            if self.dim == 2 and offset[2] != 0:
                continue
            grid_index = self.flatten_grid_index(center_cell + offset)
            for j in range(self.grid_particle_num[ti.max(0, grid_index-1)], self.grid_particle_num[grid_index]):
                if i != j and (self.pt[i].x - self.pt[j].x).norm() < self.support_radius:
                    task(i, j, ret)

    ##############################################
    # Add particles
    ##############################################
    @ti.func
    def add_particle(self, p, obj_id, x, v, density, pressure, material_id, material_type, is_dynamic, color):
        self.pt[p].obj_id = obj_id
        self.pt[p].x = x
        self.pt[p].x0 = x
        self.pt[p].v = v
        self.pt[p].density = density
        self.pt[p].m_V = self.m_V0
        self.pt[p].mass = self.m_V0 * density
        self.pt[p].pressure = pressure
        self.pt[p].mat_id = material_id
        self.pt[p].mat_type = material_type
        self.pt[p].is_dynamic = is_dynamic
        self.pt[p].color = color

    @ti.kernel
    def _add_particles(self,
                      object_id: int,
                      new_particles_num: int,
                      new_particles_positions: ti.types.ndarray(),
                      new_particles_velocity: ti.types.ndarray(),
                      new_particle_density: ti.types.ndarray(),
                      new_particle_pressure: ti.types.ndarray(),
                      new_particles_material_id: ti.types.ndarray(),
                      new_particles_material_type: ti.types.ndarray(),
                      new_particles_is_dynamic: ti.types.ndarray(),
                      new_particles_color: ti.types.ndarray()):
        for p in range(self.particle_num[None], self.particle_num[None] + new_particles_num):
            v = ti.Vector.zero(float, self.dim3)
            x = ti.Vector.zero(float, self.dim3)
            for d in ti.static(range(self.dim3)):
                v[d] = new_particles_velocity[p - self.particle_num[None], d]
                x[d] = new_particles_positions[p - self.particle_num[None], d]
            self.add_particle(p, object_id, x, v,
                              new_particle_density[p - self.particle_num[None]],
                              new_particle_pressure[p - self.particle_num[None]],
                              new_particles_material_id[p - self.particle_num[None]],
                              new_particles_material_type[p - self.particle_num[None]],
                              new_particles_is_dynamic[p - self.particle_num[None]],
                              ti.Vector([new_particles_color[p - self.particle_num[None], i] for i in range(3)]))
        self.particle_num[None] += new_particles_num


    ##############################################
    # Judge
    ##############################################
    @ti.func
    def is_fluid_particle(self, p):
        return self.pt[p].mat_type == self.mat_fluid_type

    @ti.func
    def is_soil_particle(self, p):
        return self.pt[p].mat_type == self.mat_soil_type

    @ti.func
    def is_real_particle(self, p):
        # return self.pt[p].mat_type == self.mat_fluid_type or self.pt[p].mat_type == self.mat_soil_type
        return self.pt[p].mat_type > 0

    @ti.func
    def is_dummy_particle(self, p):
        return self.pt[p].mat_type == self.mat_dummy_type

    @ti.func
    def is_rep_particle(self, p):
        return self.pt[p].mat_type == self.mat_rep_type

    @ti.func
    def is_bdy_particle(self, p):
        return self.pt[p].mat_type == self.mat_rep_type or self.pt[p].mat_type == self.mat_dummy_type

    @ti.func
    def is_rigid(self, p):
        return self.pt[p].mat_type == self.mat_rigid_type

    @ti.func
    def is_rigid_static(self, p):
        return (self.pt[p].mat_type == self.mat_rigid_type) and (not self.pt[p].is_dynamic)

    @ti.func
    def is_rigid_dynamic(self, p):
        return (self.pt[p].mat_type == self.mat_rigid_type) and self.pt[p].is_dynamic

    ##############################################
    # Visualization
    ##############################################
    @ti.kernel
    def copy2vis(self, w2s_ratio: float):
        for i in range(self.particle_num[None]):
            # if self.is_real_particle(i) or (self.show_bdy and self.is_dummy_particle(i)):
            if self.is_real_particle(i) or (self.show_bdy and self.is_dummy_particle(i)) or (self.show_bdy and self.is_rep_particle(i)):
                for j in ti.static(range(self.dim3)):
                    self.pt[i].pos2vis[j] = ti.cast((self.pt[i].x[j]) * w2s_ratio, ti.f32)

    @ti.kernel
    def v_maxmin(self, givenmax: float, givenmin: float, fixmax: int, fixmin: int):
        vmax = -float('Inf')
        vmin = float('Inf')
        for i in range(self.particle_num[None]):
            # if self.is_real_particle(i):
            if self.is_fluid_particle(i) or self.is_soil_particle(i):
                ti.atomic_max(vmax, self.pt[i].val)
                ti.atomic_min(vmin, self.pt[i].val)
        self.vmax[None] = vmax if givenmax == -1 or vmax < givenmax and not fixmax else givenmax
        self.vmin[None] = vmin if givenmin == -1 or vmin > givenmin and not fixmin else givenmin

    @ti.kernel
    def set_color(self):
        vrange1 = 1 / (self.vmax[None] - self.vmin[None])
        for i in range(self.particle_num[None]):
            # if self.is_fluid_particle(i) or self.is_soil_particle(i) or self.is_rigid_dynamic(i):
            if self.is_fluid_particle(i) or self.is_soil_particle(i):
                tmp = (self.pt[i].val - self.vmin[None]) * vrange1
                self.pt[i].color = color_map(tmp)


    ##############################################
    # Copy data
    ##############################################
    @ti.kernel
    def copy_to_numpy_nd(self, np_arr: ti.types.ndarray(), src_arr: ti.template(), dim: int):
        for i in range(self.particle_num[None]):
            for j in ti.static(range(dim)):
                np_arr[i, j] = src_arr[i][j]

    @ti.kernel
    def copy_to_numpy(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            np_arr[i] = src_arr[i]

    def dump(self):
        particle_num = self.particle_num[None]
        np_x = np.ndarray((particle_num, self.dim3), dtype=np.float64)
        self.copy_to_numpy_nd(np_x, self.pt.x)

        return {
            'pos': np_x
        }


