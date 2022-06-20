import taichi as ti

ti.init()

a = ti.Vector([1,2])
b = ti.Vector([3,4])

c = a.transpose()@b
d = a@b.transpose() + 1




@ti.kernel
def foo():
	dd = d.inverse()
	print(dd)
	ddd = dd @ a
	print(ddd)

foo()

flag_end = 1
