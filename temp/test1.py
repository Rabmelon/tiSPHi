import taichi as ti
ti.init(arch=ti.cpu)

# TODO: Test the RK4 procedure!!!

@ti.data_oriented
class test:
    def __init__(self):
        self.N = 5
        self.v = ti.Vector.field(2, dtype=float)
        self.F = ti.Vector.field(4, dtype=float)
        self.F1 = ti.Vector.field(4, dtype=float)
        root_node = ti.root.dense(ti.i, self.N)
        root_node.place(self.v, self.F, self.F1)

        self.initdata()

    @ti.kernel
    def initdata(self):
        for i in range(self.N):
            self.v[i] = ti.Vector([ti.random(float), ti.random(float)])
            for j in ti.static(range(4)):
                self.F[i][j] = i * 100 + j

    @ti.kernel
    def foo(self, m: int):
        for i in ti.static(range(self.N)):
            if m == 0:
                # self.v[None] = self.F[i][m]
                self.F1[i] = self.F[i]
                print('m1 =', m, end='; ')
                print(self.F1[i][m])
                # print(self.v[None])
            elif m < 4:
                # self.v[None] = self.F[i][m-1]
                print('m2 =', m, end='; ')
                print(self.F[i][m-1])
                # print(self.v[None])

    def fooprint(self):
        for m in ti.static(range(4)):
            self.foo(m)

# case = test()
# print(case.F)
# case.fooprint()

N = 5
F = ti.Vector.field(4, dtype=float)
v = ti.Vector.field(2, dtype=float)
ti.root.dense(ti.i, N).place(v, F)

def initF(i):
    for j in range(4):
        F[i][j] = i*100 + j


@ti.kernel
def initdata(N: int):
    for i in range(N):
        v[i] = ti.Vector([ti.random(float), ti.random(float)])
        initF(i)

initdata(N)
print(v)
print(F)

print('Hallo!')