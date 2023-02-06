import taichi as ti
import numpy as np
from eng.configer_builder import SimConfiger
from eng.colormap import color_map
from eng.particle_func import *


# TODO: add motion. It is not easy! and needs to change the system.

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
        self.color_group = self.cfg.get_cfg("colorGroup") # 0: flow, 1: flow and rigid, 2: flow, rigid and bdy
        self.flag_boundary = self.cfg.get_cfg("boundary")   # 0: none, 1: collision, 2: dummy particles, 3: dummy + repulsive particles
        self.show_bdy = self.cfg.get_cfg("showBdyPts")
        self.mat_dummy_type, self.mat_rep_type = -1, -2
        self.mat_fluid_type, self.mat_soil_type, self.mat_rigid_type = 1, 2, 11
        self.bdy_none, self.bdy_collision, self.bdy_dummy, self.bdy_rep, self.bdy_dummy_rep = 0, 1, 2, 3, 4
        self.i_dump = ti.field(int, shape=())

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

        # Materials
        self.mat_index, self.mat_fluid, self.mat_soil, self.mat_rigid = set_material(self)

        # All objects id and its particle num
        self.object_collection = dict()
        self.object_id_rigid = set()

        ##############################################
        # Compute number of particles
        ##############################################
        # Process Blocks
        self.blocks = self.cfg.get_blocks()
        block_particle_num = 0
        for block in self.blocks:
            chk_block_in_domain(self.domain_start, self.domain_end, block["translation"], block["size"], self.dim)
            particle_num, _ = calc_cube_particle_num(block["translation"], block["size"], self.dim, offset=self.particle_diameter)
            block["particleNum"] = particle_num
            self.object_collection[block["objectId"]] = block
            block_particle_num += particle_num
            print("Block %d particle number: %d" % (block["objectId"], particle_num))

        # Process Bodies
        self.bodies = self.cfg.get_bodies()
        body_particle_num = 0
        for body in self.bodies:
            voxelized_points_np = load_body(body, self.particle_diameter)
            body["particleNum"] = voxelized_points_np.shape[0]
            body["voxelizedPoints"] = voxelized_points_np
            self.object_collection[body["objectId"]] = body
            body_particle_num += voxelized_points_np.shape[0]
            print("Body %d particle number: %d" % (body["objectId"], voxelized_points_np.shape[0]))

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

        self.particle_max_num = block_particle_num + body_particle_num + dummy_particle_num + rep_particle_num
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

        # rigid body rest center coordinate
        self.rigid_rest_cm = type_vec3f.field(shape=ti.max(len(self.blocks)+len(self.bodies), 1))

        # prefix sum
        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.grid_particle_num.shape[0])

        ##############################################
        # Initialize particles
        ##############################################
        self.initialize_particles()
        self.set_id0()

        print("Particle system construction complete!")

    def initialize_particles(self):
        # Blocks
        for block in self.blocks:
            self.init_block(block)

        # Bodies
        for body in self.bodies:
            self.init_body(body)

        # Boundary particles
        if self.flag_boundary == self.bdy_dummy or self.flag_boundary == self.bdy_dummy_rep:
            add_boundary(self, self.dummy_boundary, self.mat_dummy_type, color=[153, 153, 255])
        if self.flag_boundary == self.bdy_rep or self.flag_boundary == self.bdy_dummy_rep:
            add_boundary(self, self.rep_boundary, self.mat_rep_type, offset=self.particle_radius, color=[170, 17, 255])

    def init_block(self, block):
        obj_id = block["objectId"]
        start = np.array(block["translation"])
        size = np.array(block["size"])
        velocity = block["velocity"]
        mat_Id = block["materialId"]
        mat_fluid = get_material(self, mat_Id)
        mat_type = mat_fluid["matType"]
        density = mat_fluid["density0"]
        color = np.array([ic / 255 for ic in mat_fluid["color"]], dtype=np.float32)
        if mat_type > 10:
            self.object_id_rigid.add(obj_id)
            is_dynamic = block["isDynamic"]
        elif mat_type < 10 and mat_type > 0:
            is_dynamic = True
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

    def init_body(self, body):
        obj_id = body["objectId"]
        num_particles_obj = body["particleNum"]
        voxelized_points_np = body["voxelizedPoints"]
        velocity = np.array(body["velocity"], dtype=np.float64)
        mat_Id = body["materialId"]
        mat_body = get_material(self, mat_Id)
        mat_type = mat_body["matType"]
        density = mat_body["density0"]
        color = np.array([ic / 255 for ic in mat_body["color"]], dtype=np.float32)
        if mat_type > 10:
            self.object_id_rigid.add(obj_id)
            is_dynamic = body["isDynamic"]
        elif mat_type < 10 and mat_type > 0:
            is_dynamic = True
        add_particles(self, obj_id, num_particles_obj,
            np.array(voxelized_points_np, dtype=np.float64),  # position
            np.stack([velocity for _ in range(num_particles_obj)]),  # velocity
            density * np.ones(num_particles_obj, dtype=np.float64),  # density
            np.zeros(num_particles_obj, dtype=np.float64),  # pressure
            np.array([mat_Id for _ in range(num_particles_obj)], dtype=np.int32),  # material id is solid
            np.array([mat_type for _ in range(num_particles_obj)], dtype=np.int32),  # material type is solid
            is_dynamic * np.ones(num_particles_obj, dtype=np.int32),  # is_dynamic
            np.stack([color for _ in range(num_particles_obj)]))  # color


    # ! want to clear and reset the particles, wrong now
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
        self.counting_sort()

    @ti.func
    def for_all_neighbors(self, i, task: ti.template(), ret: ti.template()):
        center_cell = self.pos_to_index(self.pt[i].x)
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim3)):
            if self.dim == 2 and offset[2] != 0:
                continue
            grid_index = self.flatten_grid_index(center_cell + offset)
            j_range_begin = self.grid_particle_num[grid_index-1] if grid_index > 0 else 0
            for j in range(j_range_begin, self.grid_particle_num[grid_index]):
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
    def is_flow_particle(self, p):
        return self.pt[p].mat_type == self.mat_fluid_type or self.pt[p].mat_type == self.mat_soil_type

    @ti.func
    def is_real_particle(self, p):
        return self.pt[p].mat_type > 0

    @ti.func
    def is_real_particle_dynamic(self, p):
        return (self.pt[p].mat_type > 0) and self.pt[p].is_dynamic

    @ti.func
    def is_dummy_particle(self, p):
        return self.pt[p].mat_type == self.mat_dummy_type

    @ti.func
    def is_rep_particle(self, p):
        return self.pt[p].mat_type == self.mat_rep_type

    @ti.func
    def is_bdy_particle(self, p):
        return self.pt[p].mat_type == self.mat_dummy_type or self.pt[p].mat_type == self.mat_rep_type

    @ti.func
    def is_rigid(self, p):
        return self.pt[p].mat_type == self.mat_rigid_type

    @ti.func
    def is_rigid_static(self, p):
        return (self.pt[p].mat_type == self.mat_rigid_type) and (not self.pt[p].is_dynamic)

    @ti.func
    def is_rigid_dynamic(self, p):
        return (self.pt[p].mat_type == self.mat_rigid_type) and self.pt[p].is_dynamic

    @ti.func
    def is_object_id(self, p, oid):
        return self.pt[p].obj_id == oid

    @ti.func
    def is_material_type(self, p, mtype):
        return self.pt[p].mat_type == mtype

    @ti.func
    def is_same_type(self, p, q):
        return self.pt[p].mat_type == self.pt[q].mat_type


    ##############################################
    # Visualization
    ##############################################
    @ti.kernel
    def copy2vis(self, w2s_ratio: float):
        for i in range(self.particle_num[None]):
            if self.is_real_particle(i) or (self.show_bdy and self.is_bdy_particle(i)):
                for j in ti.static(range(self.dim3)):
                    self.pt[i].pos2vis[j] = ti.cast((self.pt[i].x[j]) * w2s_ratio, ti.f32)

    @ti.kernel
    def v_maxmin(self, givenmax: float, givenmin: float, fixmax: int, fixmin: int):
        vmax = -float('Inf')
        vmin = float('Inf')
        for i in range(self.particle_num[None]):
            # if self.is_real_particle(i):
            if self.is_flow_particle(i):
                ti.atomic_max(vmax, self.pt[i].val)
                ti.atomic_min(vmin, self.pt[i].val)
        self.vmax[None] = vmax if givenmax == -1 or vmax < givenmax and not fixmax else givenmax
        self.vmin[None] = vmin if givenmin == -1 or vmin > givenmin and not fixmin else givenmin

    @ti.kernel
    def set_color(self):
        vrange1 = 1 / (self.vmax[None] - self.vmin[None])
        for i in range(self.particle_num[None]):
            if (self.color_group == 0 and self.is_flow_particle(i)) or (
                self.color_group == 1 and self.is_real_particle(i)) or (
                self.color_group == 2 and (self.is_real_particle(i) or (self.show_bdy and self.is_dummy_particle(i)))):
                tmp = (self.pt[i].val - self.vmin[None]) * vrange1
                self.pt[i].color = color_map(tmp)


    ##############################################
    # Copy data
    ##############################################
    @ti.kernel
    def copy_to_numpy_mat(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            for j in ti.static(range(self.dim3)):
                for k in ti.static(range(self.dim3)):
                    np_arr[i, j, k] = src_arr[i][j,k]

    @ti.kernel
    def copy_to_numpy_vec(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            for j in ti.static(range(self.dim3)):
                np_arr[i, j] = src_arr[i][j]

    @ti.kernel
    def copy_to_numpy_matxyz(self, np_arr_xx: ti.types.ndarray(), np_arr_yy: ti.types.ndarray(), np_arr_zz: ti.types.ndarray(), np_arr_xy: ti.types.ndarray(), np_arr_yz: ti.types.ndarray(), np_arr_zx: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            np_arr_xx[i] = src_arr[i][0, 0]
            np_arr_yy[i] = src_arr[i][1, 1]
            np_arr_zz[i] = src_arr[i][2, 2]
            np_arr_xy[i] = src_arr[i][0, 1]
            np_arr_yz[i] = src_arr[i][1, 2]
            np_arr_zx[i] = src_arr[i][2, 0]

    @ti.kernel
    def copy_to_numpy_vecxyz(self, np_arr_x: ti.types.ndarray(), np_arr_y: ti.types.ndarray(), np_arr_z: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            np_arr_x[i] = src_arr[i].x
            np_arr_y[i] = src_arr[i].y
            np_arr_z[i] = src_arr[i].z

    @ti.kernel
    def copy_to_numpy(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            np_arr[i] = src_arr[i]

    @ti.kernel
    def copy_to_numpy_vecnorm(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            np_arr[i] = src_arr[i].norm()

    @ti.kernel
    def copy_to_numpy_mathydro(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            np_arr[i] = (src_arr[i][0,0] + src_arr[i][1,1] + src_arr[i][2,2]) / 3.0


    def dump(self):
        # particle_num = self.object_collection[obj_id]["particleNum"]
        particle_num = self.particle_num[None]

        id0 = np.zeros((particle_num), dtype=np.int64)
        self.copy_to_numpy(id0, self.pt.id0)

        obj_id = np.zeros((particle_num), dtype=np.int64)
        self.copy_to_numpy(obj_id, self.pt.obj_id)

        mat_type = np.zeros((particle_num), dtype=np.int64)
        self.copy_to_numpy(mat_type, self.pt.mat_type)

        density = np.zeros((particle_num), dtype=np.float64)
        self.copy_to_numpy(density, self.pt.density)

        # position = np.zeros((particle_num, self.dim3), dtype=np.float64)
        # self.copy_to_numpy_vec(position, self.pt.x)
        pos_x = np.zeros((particle_num), dtype=np.float64)
        pos_y = np.zeros((particle_num), dtype=np.float64)
        pos_z = np.zeros((particle_num), dtype=np.float64)
        self.copy_to_numpy_vecxyz(pos_x, pos_y, pos_z, self.pt.x)

        # velocity = np.zeros((particle_num, self.dim3), dtype=np.float64)
        # self.copy_to_numpy_vec(velocity, self.pt.v)
        vel_x = np.zeros((particle_num), dtype=np.float64)
        vel_y = np.zeros((particle_num), dtype=np.float64)
        vel_z = np.zeros((particle_num), dtype=np.float64)
        self.copy_to_numpy_vecxyz(vel_x, vel_y, vel_z, self.pt.v)

        vel_norm = np.zeros((particle_num), dtype=np.float64)
        self.copy_to_numpy_vecnorm(vel_norm, self.pt.v)

        # stress = np.zeros((particle_num, self.dim3, self.dim3), dtype=np.float64)
        # self.copy_to_numpy_mat(stress, self.pt.stress)
        stress_xx = np.zeros((particle_num), dtype=np.float64)
        stress_yy = np.zeros((particle_num), dtype=np.float64)
        stress_zz = np.zeros((particle_num), dtype=np.float64)
        stress_xy = np.zeros((particle_num), dtype=np.float64)
        stress_yz = np.zeros((particle_num), dtype=np.float64)
        stress_zx = np.zeros((particle_num), dtype=np.float64)
        self.copy_to_numpy_matxyz(stress_xx, stress_yy, stress_zz, stress_xy, stress_yz, stress_zx, self.pt.stress)

        stress_hydro = np.zeros((particle_num), dtype=np.float64)
        self.copy_to_numpy_mathydro(stress_hydro, self.pt.stress)

        strain_equ = np.zeros((particle_num), dtype=np.float64)
        self.copy_to_numpy(strain_equ, self.pt.strain_equ)

        strain_equ_p = np.zeros((particle_num), dtype=np.float64)
        self.copy_to_numpy(strain_equ_p, self.pt.strain_equ_p)

        pressure = np.zeros((particle_num), dtype=np.float64)
        self.copy_to_numpy(pressure, self.pt.pressure)

        retmap = np.zeros((particle_num), dtype=np.float64)
        self.copy_to_numpy(retmap, self.pt.flag_retmap)

        return {
            # 'position': position,
            'pos.x': pos_x,
            'pos.y': pos_y,
            'pos.z': pos_z
        }, {
            'id0': id0,
            'objId': obj_id,
            'material': mat_type,
            # 'velocity': velocity,
            'density': density,
            'vel.x': vel_x,
            'vel.y': vel_y,
            'vel.z': vel_z,
            'vel.norm': vel_norm,
            # 'stress': stress,
            "stress.xx": stress_xx,
            "stress.yy": stress_yy,
            "stress.zz": stress_zz,
            "stress.xy": stress_xy,
            "stress.yz": stress_yz,
            "stress.zx": stress_zx,
            "stress.hydro": stress_hydro,
            'strain_equ': strain_equ,
            'strain_equ_p': strain_equ_p,
            'pressure': pressure,
			"plas_behav": retmap

        }
