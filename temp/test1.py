import taichi as ti
ti.init(arch=ti.cpu)

# TODO: Test the RK4 procedure!!!

a = ti.Matrix([[1, 2], [3, 4]])
b = ti.Matrix([[1, 5], [2, 6], [3, 7], [4, 8]])
# a = ti.Matrix.field(4, 2, dtype=ti.f32, shape=())
# a = b * 2.0

c = ti.Vector([10, 20])

d = c.dot(c)
da = a@c
db = b@c

e = ti.Vector([b[1, 0], b[1, 1]])
f = e.dot(c)

g = ti.Vector.field(2, dtype=float)
h = ti.Vector.field(5, dtype=float)
N = 10
# test_node = ti.root.dense(ti.i, N).dense(ti.j, 4)
root_node = ti.root.dense(ti.i, N)
root_node.place(h)
root_node.dense(ti.j, 4).place(g)
for i in range(N):
    h[i] = ti.Vector([111, 222, 333, 444, 555]) + i*1000
    for j in range(4):
        for k in range(2):
            g[i, j][k] = i*100 + j*10 + k

print(g)
print(h)

for m in (1,2,3,4):
    print(m)

# print('a =', a)
# print('b =', b)
# print('c =', c)
# print('d =', d)
# print('da =', da)
# print('db =', db)
# print('e =', e)
# print('f =', f)
