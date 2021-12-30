import taichi as ti
ti.init(arch=ti.cpu)

# TODO: Test the RK4 procedure!!!

@ti.data_oriented
class test:
    def __init__(self):
        self.N = 5
        self.dt = ti.field(dtype=float, shape=())
        self.v = ti.Vector.field(2, dtype=float)
        self.u1234 = ti.Vector.field(2, dtype=float)
        self.F1 = ti.Vector.field(4, dtype=float)
        root_node = ti.root.dense(ti.i, self.N)
        root_node.place(self.v, self.u1234, self.F1)

        self.initdata()

    @ti.kernel
    def initdata(self):
        self.dt[None] = 5e-4
        for i in range(self.N):
            self.v[i] = ti.Vector([ti.random(float), ti.random(float)])
            self.F1[i] = ti.Vector([0.0 for _ in range(4)])

    @ti.kernel
    def compute_F(self, m: int):
        for p_i in range(self.N):
            self.F1[p_i][m] = 1

    @ti.kernel
    def update_u(self, m: int):
        for p_i in range(self.N):
            if m == 0:
                self.u1234[p_i] = self.v[p_i]
            elif m < 4:
                assert m < 1, 'Error: m < 1 here!'
                self.u1234[p_i] = self.v[p_i] + 0.5 * self.dt[None] * self.F1[p_i][m-1]

    def fooprint(self):
        for m in ti.static(range(4)):
            self.update_u(m)

case = test()
print(case.F1)
case.fooprint()



print('Hallo!')