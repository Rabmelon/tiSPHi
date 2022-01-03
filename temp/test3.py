import taichi as ti
ti.init(ti.cpu)

# TODO: test the maximum print

N = 10
x = ti.Vector.field(2, dtype=float, shape=N)

@ti.kernel
def addprint():
    y = ti.Vector([0.0, 0.0])
    for i in range(N):
        x[i] = ti.Vector([ti.random(float), 10+ti.random(float)])
        y = max(abs(x[i]), abs(y))
        print('each x =', x[i], 'and now y =', y)
    print('final y =', y)

addprint()


