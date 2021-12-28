import taichi as ti
ti.init(arch=ti.cpu)

a = ti.Matrix([[1, 2], [3, 4]])
b = ti.Matrix([[1, 5], [2, 6], [3, 7], [4, 8]])
# a = ti.Matrix.field(4, 2, dtype=ti.f32, shape=())
# a = b * 2.0

c = ti.Vector([10, 20])

d = c.dot(c)
print('b[:, 0] =', b[1, 0])

print('a =', a)
print('b =', b)
print('c =', c)
print('d =', d)
