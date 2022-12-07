import taichi as ti
import numpy as np
import trimesh as tm
from functools import reduce    # 整数：累加；字符串、列表、元组：拼接。lambda为使用匿名函数
from eng.type_define import *


##############################################
# Class defination
##############################################
@ti.dataclass
class Particle:
    # basic
    obj_id: int
    mat_id: int
    mat_type: int
    color: type_vec3f32
    is_dynamic: int

    # properties
    x: type_vec3f
    val: type_f

    x0: type_vec3f
    m_V: type_f
    density: type_f
    mass: type_f
    v: type_vec3f
    pressure: type_f
    stress: type_mat3f

    # counting sort
    grid_ids: int
    id_new: int
    id0: int

    # visulization
    pos2vis: type_vec3f32

    # correction
    CSPM_f: type_f
    CSPM_L: type_mat3f
    MLS_beta: type_vec4f

    # solver
    d_density: type_f
    d_vel: type_vec3f
    d_stress: type_mat3f
    v_grad: type_mat3f
    strain_equ: type_f
    d_strain_equ: type_f
    dist_B: type_f

    # tmp
    density_tmp: type_f
    v_tmp: type_vec3f
    stress_tmp: type_mat3f

    # for RK4
    d_density_RK: type_f
    d_vel_RK: type_vec3f
    d_stress_RK: type_mat3f


    @ti.func
    def set_stress_all(self, sxx=0.0, syy=0.0, szz=0.0, sxy=0.0, syz=0.0, szx=0.0):
        self.stress = type_mat3f([sxx, sxy, szx], [sxy, syy, syz], [szx, syz, szz])

    @ti.func
    def set_stress_diag(self, sxx=0.0, syy=0.0, szz=0.0):
        self.stress[0,0] = sxx
        self.stress[1,1] = syy
        self.stress[2,2] = szz

    @ti.func
    def set_mass(self):
        self.mass = self.density * self.m_V

    @ti.func
    def set_volume(self):
        self.m_V = self.mass / self.density


class Water:
    matId: int
    matType: int
    color: list
    density0: float
    stiffness: float
    exponent: float

    def __init__(self,
                 mat_id=0,
                 mat_type=1,
                 color=[0, 0, 0],
                 density=1000,
                 stiffness=5e4,
                 exponent=7):
        self.matId = mat_id
        self.matType = mat_type
        self.color = color
        self.density0 = density
        self.stiffness = stiffness
        self.exponent = exponent

class Soil:
    matId: int
    matType: int
    color: list
    density0: float
    cohesion: float
    friction: float
    EYoungMod: float
    poison: float
    dilatancy: float

    def __init__(self,
                 mat_id=0,
                 mat_type=2,
                 color=[0, 0, 0],
                 density=2000,
                 cohesion=0,
                 friction=45,
                 EYoungMod=1e8,
                 poison=0.3,
                 dilatancy=0):
        self.matId = mat_id
        self.matType = mat_type
        self.color = color
        self.density0 = density
        self.cohesion = cohesion
        self.friction = friction
        self.EYoungMod = EYoungMod
        self.poison = poison
        self.dilatancy = dilatancy

class Rigid:
    matId: int
    matType: int
    color: list
    density0: float

    def __init__(self, mat_id=0, mat_type=11, color=[0, 0, 0], density=7000):
        self.matId = mat_id
        self.matType = mat_type
        self.color = color
        self.density0 = density


##############################################
# Add particles
##############################################
def add_particles(ps, object_id: int, new_particles_num: int,
                  new_particles_positions: ti.types.ndarray(),
                  new_particles_velocity: ti.types.ndarray(),
                  new_particle_density: ti.types.ndarray(),
                  new_particle_pressure: ti.types.ndarray(),
                  new_particles_material_id: ti.types.ndarray(),
                  new_particles_material_type: ti.types.ndarray(),
                  new_particles_is_dynamic: ti.types.ndarray(),
                  new_particles_color: ti.types.ndarray()):

    ps._add_particles(object_id, new_particles_num, new_particles_positions,
                      new_particles_velocity, new_particle_density,
                      new_particle_pressure, new_particles_material_id,
                      new_particles_material_type, new_particles_is_dynamic,
                      new_particles_color)

def add_cube(ps,
             object_id,
             lower_corner,
             cube_size,
             mat_id=0,
             mat_type=1,
             is_dynamic=True,
             color=(0, 0, 0),
             density=None,
             pressure=None,
             velocity=None,
             offset=None):

    pt_offset = offset if offset is not None else ps.particle_diameter
    num_new_particles, num_dim = calc_cube_particle_num(lower_corner, cube_size, ps.dim, offset=pt_offset)
    if ps.dim == 2:
        num_dim.append(np.array([0.0]))

    # ! be careful of dtype: float32!
    new_positions = np.array(np.meshgrid(*num_dim, sparse=False, indexing='ij'), dtype=np.float64)
    new_positions = new_positions.reshape(-1, reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
    if velocity is None:
        velocity_arr = np.full_like(new_positions, 0, dtype=np.float64)
    else:
        velocity_arr = np.array([velocity for _ in range(num_new_particles)], dtype=np.float64)

    mat_id_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), mat_id)
    mat_type_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), mat_type)
    is_dynamic_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), is_dynamic)
    color_arr = np.stack([np.full_like(np.zeros(num_new_particles, dtype=np.int32), c, dtype=np.float32) for c in color], axis=1)
    density_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float64), density if density is not None else 1000.)
    pressure_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float64), pressure if pressure is not None else 0.)
    add_particles(ps, object_id, num_new_particles, new_positions, velocity_arr, density_arr, pressure_arr, mat_id_arr, mat_type_arr, is_dynamic_arr, color_arr)

def add_cube_boundary(ps, pos_bld, pos_fru, type, offset=None, color=[0,0,0]):
    add_cube(ps=ps, lower_corner=pos_bld, cube_size=pos_fru - pos_bld, mat_type=type, color=color, object_id=type, offset=offset)



##############################################
# Assist
##############################################
def set_material(ps):
    mat_list = ps.cfg.get_materials()
    mat_index, water_list, soil_list, rigid_list = [], [], [], []
    i_water, i_soil, i_rigid = 0, 0, 0
    for mat in mat_list:
        mat["color"] = mat["color"]
        if mat["matType"] == ps.mat_fluid_type:
            water_list.append(mat)
            mat_index.append([ps.mat_fluid_type, i_water])
            i_water += 1
        elif mat["matType"] == ps.mat_soil_type:
            soil_list.append(mat)
            mat_index.append([ps.mat_soil_type, i_soil])
            i_soil += 1
        elif mat["matType"] == ps.mat_rigid_type:
            rigid_list.append(mat)
            mat_index.append([ps.mat_rigid_type, i_rigid])
            i_rigid += 1
    return mat_index, water_list, soil_list, rigid_list

def get_material(ps, i_mat_index):
    if ps.mat_index[i_mat_index][0] == ps.mat_fluid_type:
        return ps.mat_fluid[ps.mat_index[i_mat_index][1]]
    elif ps.mat_index[i_mat_index][0] == ps.mat_soil_type:
        return ps.mat_soil[ps.mat_index[i_mat_index][1]]
    elif ps.mat_index[i_mat_index][0] == ps.mat_rigid_type:
        return ps.mat_rigid[ps.mat_index[i_mat_index][1]]

def chk_block_in_domain(domain_start, domain_end, block_translation, block_size, dim):
    # true means in domain
    # ! now without block rotation !
    flag_start = all([block_translation[i] - domain_start[i] >= 0.0 for i in range(dim)])
    flag_end = all([block_translation[i] + block_size[i] - domain_end[i] <= 0.0 for i in range(dim)])
    assert flag_start and flag_end, "Block is not in domain!"

def calc_cube_particle_num(translation, size, dim=3, offset=None):
    num_dim = []
    range_offset = offset if offset is not None else 0.1
    for i in range(dim):
        off = range_offset if size[i] >= 0 else -range_offset
        num_dim.append(np.arange(translation[i] + off / 2.0, translation[i] + size[i] + 1e-5, off))
    return reduce(lambda x, y: x * y, [len(n) for n in num_dim]), num_dim

def count_cube_num(pos_bld, pos_fru, dim, offset):
    tmp = calc_cube_particle_num(pos_bld, pos_fru - pos_bld, dim, offset=offset)
    return tmp[0]

def load_body(body, vox_len):
    obj_id = body["objectId"]
    mesh = tm.load(body["geometryFile"])
    mesh.apply_scale(body["scale"])
    offset = np.array(body["translation"])

    angle = body["rotationAngle"] / 180 * ti.math.pi
    direction = body["rotationAxis"]
    rot_matrix = tm.transformations.rotation_matrix(angle, direction, mesh.vertices.mean(axis=-1))
    mesh.apply_transform(rot_matrix)

    is_dynamic = body["isDynamic"]
    if is_dynamic:
        # Backup the original mesh for exporting obj
        mesh_backup = mesh.copy()
        mesh_backup.vertices += offset
        body["mesh"] = mesh_backup
        body["restPosition"] = mesh_backup.vertices
        body["restCenterOfMass"] = mesh_backup.vertices.mean(axis=-1)
        # is_success = tm.repair.fill_holes(mesh)
        # print("Is the mesh successfully repaired? ", is_success)
    voxelized_mesh = mesh.voxelized(pitch=vox_len)
    voxelized_mesh = mesh.voxelized(pitch=vox_len).fill()
    # voxelized_mesh = mesh.voxelized(pitch=ps.particle_diameter).hollow()
    # voxelized_mesh.show()     # ! will raise AttributeError: 'SceneViewer' object has no attribute 'ubo'
    voxelized_points_np = voxelized_mesh.points + offset
    # print(f"body {obj_id} num: {voxelized_points_np.shape[-1]}")

    return voxelized_points_np


##############################################
# Boundary particle
##############################################
# ! very streightforward cube boundary now!!!

def count_boundary(boundary, dim, offset):
    num = 0
    for i in range(len(boundary)):
        num += count_cube_num(boundary[i][0], boundary[i][1], dim, offset)
    return num

def add_boundary(ps, boundary, type, offset=None, color=[255,255,255]):
    bdy_type = type
    bdy_color = np.array(color) / 255
    for i in range(len(boundary)):
        add_cube_boundary(ps, boundary[i][0], boundary[i][1], bdy_type, offset, bdy_color)

# Dummy particle
def calc_dummy_boundary(dim, domain_start, domain_end, vdomain_start, vdomain_end):
    if dim == 3:
        dummy_b_bld = np.array([vdomain_start[0], domain_start[1], vdomain_start[2]])
        dummy_b_fru = np.array([domain_start[0], domain_end[1], domain_end[2]])
        dummy_r_bld = np.array([vdomain_start[0], domain_start[1], domain_end[2]])
        dummy_r_fru = np.array([domain_end[0], domain_end[1], vdomain_end[2]])
        dummy_f_bld = np.array([domain_end[0], domain_start[1], domain_start[2]])
        dummy_f_fru = np.array([vdomain_end[0], domain_end[1], vdomain_end[2]])
        dummy_l_bld = np.array([domain_start[0], domain_start[1], vdomain_start[2]])
        dummy_l_fru = np.array([vdomain_end[0], domain_end[1], domain_start[2]])
        dummy_d_bld = vdomain_start
        dummy_d_fru = np.array([vdomain_end[0], domain_start[1], vdomain_end[2]])
        dummy_boundary = [[dummy_b_bld, dummy_b_fru], [dummy_r_bld, dummy_r_fru], [dummy_f_bld, dummy_f_fru], [dummy_l_bld, dummy_l_fru], [dummy_d_bld, dummy_d_fru]]
        # dummy_u_bld = np.array([vdomain_start[0], domain_end[1], vdomain_start[2]])
        # dummy_u_fru = vdomain_end
        # dummy_boundary = [[dummy_b_bld, dummy_b_fru], [dummy_r_bld, dummy_r_fru], [dummy_f_bld, dummy_f_fru], [dummy_l_bld, dummy_l_fru], [dummy_d_bld, dummy_d_fru], [dummy_u_bld, dummy_u_fru]]
    elif dim == 2:
        dummy_b_bd = np.array([vdomain_start[0], domain_start[1], domain_start[2]])
        dummy_b_fu = np.array([domain_start[0], domain_end[1], domain_end[2]])
        dummy_d_bd = np.array([vdomain_start[0], vdomain_start[1], domain_start[2]])
        dummy_d_fu = np.array([vdomain_end[0], domain_start[1], domain_end[2]])
        dummy_f_bd = np.array([domain_end[0], domain_start[1], domain_start[2]])
        dummy_f_fu = np.array([vdomain_end[0], domain_end[1], domain_end[2]])
        dummy_boundary = [[dummy_b_bd, dummy_b_fu], [dummy_d_bd, dummy_d_fu], [dummy_f_bd, dummy_f_fu]]
        # dummy_u_bd = np.array([vdomain_start[0], domain_end[1], domain_start[2]])
        # dummy_u_fu = np.array([vdomain_end[0], vdomain_end[1], domain_end[2]])
        # dummy_boundary = [[dummy_b_bd, dummy_b_fu], [dummy_d_bd, dummy_d_fu], [dummy_f_bd, dummy_f_fu], [dummy_u_bd, dummy_u_fu]]
    return dummy_boundary

# Repulsive particle
def calc_rep_boundary(dim, domain_start, domain_end, pt_radius):
    tmpr = pt_radius / 2
    if dim == 3:
        rep_b_bld = np.array([domain_start[0] - tmpr, domain_start[1] + tmpr, domain_start[2] - tmpr])
        rep_b_fru = np.array([domain_start[0] + tmpr, domain_end[1] - tmpr, domain_end[2] - tmpr])
        rep_r_bld = np.array([domain_start[0] - tmpr, domain_start[1] + tmpr, domain_end[2] - tmpr])
        rep_r_fru = np.array([domain_end[0] - tmpr, domain_end[1] - tmpr, domain_end[2] + tmpr])
        rep_f_bld = np.array([domain_end[0] - tmpr, domain_start[1] + tmpr, domain_start[2] + tmpr])
        rep_f_fru = np.array([domain_end[0] + tmpr, domain_end[1] - tmpr, domain_end[2] + tmpr])
        rep_l_bld = np.array([domain_start[0] + tmpr, domain_start[1] + tmpr, domain_start[2] - tmpr])
        rep_l_fru = np.array([domain_end[0] + tmpr, domain_end[1] - tmpr, domain_start[2] + tmpr])
        rep_d_bld = domain_start - tmpr
        rep_d_fru = np.array([domain_end[0], domain_start[1], domain_end[2]]) + tmpr
        rep_boundary = [[rep_b_bld, rep_b_fru], [rep_r_bld, rep_r_fru], [rep_f_bld, rep_f_fru], [rep_l_bld, rep_l_fru], [rep_d_bld, rep_d_fru]]
        # rep_u_bld = np.array([domain_start[0], domain_end[1], domain_start[2]])
        # rep_u_fru = domain_end + tmpr
        # rep_boundary = [[rep_b_bld, rep_b_fru], [rep_r_bld, rep_r_fru], [rep_f_bld, rep_f_fru], [rep_l_bld, rep_l_fru], [rep_d_bld, rep_d_fru], [rep_u_bld, rep_u_fru]]
    elif dim == 2:
        rep_b_bd = np.array([domain_start[0] - tmpr, domain_start[1] + tmpr, domain_start[2]])
        rep_b_fu = np.array([domain_start[0] + tmpr, domain_end[1] - tmpr, domain_end[2]])
        rep_d_bd = np.array([domain_start[0] - tmpr, domain_start[1] - tmpr, domain_start[2]])
        rep_d_fu = np.array([domain_end[0] + tmpr, domain_start[1] + tmpr, domain_end[2]])
        rep_f_bd = np.array([domain_end[0] - tmpr, domain_start[1] + tmpr, domain_start[2]])
        rep_f_fu = np.array([domain_end[0] + tmpr, domain_end[1] - tmpr, domain_end[2]])
        rep_boundary = [[rep_b_bd, rep_b_fu], [rep_d_bd, rep_d_fu], [rep_f_bd, rep_f_fu]]
        # rep_u_bd = np.array([domain_start[0] - tmpr, domain_end[1] - tmpr, domain_start[2]])
        # rep_u_fu = np.array([domain_end[0] + tmpr, domain_end[1] + tmpr, domain_end[2]])
        # rep_boundary = [[rep_b_bd, rep_b_fu], [rep_d_bd, rep_d_fu], [rep_f_bd, rep_f_fu], [rep_u_bd, rep_u_fu]]
    return rep_boundary



