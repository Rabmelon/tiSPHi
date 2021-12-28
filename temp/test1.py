import taichi as ti
ti.init(arch=ti.cpu)

a = ti.Matrix.field(4, 2, dtype=ti.f32, shape=())
b = ti.Matrix([[1, 5], [2, 6], [3, 7], [4, 8]])
a = b * 2.0

print('a =', a)
print('b =', b)

