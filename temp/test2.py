import taichi as ti

ti.init(ti.cpu)

d = ti.Vector.field(2, dtype=float, shape=1)
d = ti.Vector([2.5, 1.5])

padding = ti.field(dtype=float, shape=())
padding[None] = 2


for m in range(2):
    assert d[m] > padding[None], 'My Error!'

print('???')
