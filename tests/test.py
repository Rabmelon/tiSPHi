import taichi as ti
import numpy as np

ti.init(arch=ti.cpu, debug=True, default_fp=ti.f64, kernel_profiler=True)

# mat2 and vec3 are predefined types in the ti.math module
mat2 = ti.types.matrix(2, 2, float)
vec3 = ti.types.vector(3, float)

m = mat2(1)  # [[1., 1.], [1., 1.]]
m = mat2(1, 2, 3, 4)  # [[1., 2.], [3, 4.]]
m = mat2([1, 2], [3, 4])  # [[1., 2.], [3, 4.]]
m = mat2([1, 2, 3, 4])  # [[1., 2.], [3, 4.]]
v = vec3(1, 2, 3)
m = mat2(v, 4)  # [[1., 2.], [3, 4.]]

a = ti.math.vec3(1.5, 0.5, 0.1)
print(a)

b = ti.math.mat3(2.4)
print(b)

@ti.kernel
def defc() -> ti.types.matrix(3, 3, float):
	c = ti.math.eye(3)
	return c

d = defc()
print(d)

cc = ti.Matrix(np.eye(3))
print(cc)

