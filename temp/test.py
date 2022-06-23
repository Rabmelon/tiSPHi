import taichi as ti
import numpy as np

ti.init()

a = ti.Vector([1,2])
b = ti.Vector([3,4])

c = a.transpose()@b
d = a@b.transpose() + 1

k = 1.5

dim = 2
N = 3
kk = ti.Vector.field(dim, float, shape=(N))

print(kk[0])

# @ti.func
def WendlandC2_kernel_derivative(r):
	res = ti.Vector([0.0 for _ in range(dim)])
	h1 = 1 / 12
	k = 7 / (4 * np.pi) if dim == 2 else 21 / (2 * np.pi) if dim == 3 else 0.0
	k *= h1**dim
	r_norm = r.norm()
	q = r_norm * h1
	if r_norm > 1e-8 and q <= 2.0:
		q1 = 1 - 0.5 * q
		res = k * ti.pow(q1, 3) * (-5 * q) * h1 * r / r_norm
	return res

@ti.kernel
def foo():
	for i in range(N):
		d = ti.Vector([0.0 for _ in range(dim)])
		for j in range(dim):
			d += k * WendlandC2_kernel_derivative(a)
		kk[i] += d

foo()

flag_end = 1
