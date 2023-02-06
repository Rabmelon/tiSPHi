import taichi as ti

type_f = ti.f64

type_vec3f32 = ti.types.vector(3, ti.f32)
type_vec3f = ti.types.vector(3, ti.f64)
type_vec4f = ti.types.vector(3, ti.f64)
type_vec2f = ti.types.vector(2, ti.f64)

type_mat3f = ti.types.matrix(3, 3, ti.f64)
type_mat4f = ti.types.matrix(4, 4, ti.f64)
type_mat2f = ti.types.matrix(2, 2, ti.f64)

@ti.func
def trans_mat_3_2(mat3):
    return type_mat2f([[mat3[0,0], mat3[0,1]], [mat3[1,0], mat3[1,1]]])

@ti.func
def trans_mat_2_3_fill0(mat2):
	return type_mat3f([[mat2[0,0], mat2[0,1], 0], [mat2[1,0], mat2[1,1], 0], [0,0,0]])

@ti.func
def trans_vec3_diag(vec3):
    return type_mat3f([[vec3[0], 0, 0], [0, vec3[1], 0], [0, 0, vec3[2]]])

@ti.func
def calc_dev_component(ts):
	return ti.sqrt((ts * ts).sum() * 2 / 3)
