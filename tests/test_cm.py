import taichi as ti

ti.init(arch=ti.cuda)

a = ti.field(float, shape=())
a[None] = 0.5
print(a[None])
c = ti.Vector.field(3, ti.f32, shape=1)
pos = ti.Vector.field(2, ti.f32, shape=1)

# color map is copy from: https://forum.taichi.graphics/t/vortex-method-demo/775
@ti.data_oriented
class ColorMap:
    def __init__(self, h, wl, wr, c):
        self.h = h
        self.wl = wl
        self.wr = wr
        self.c = c

    @ti.func
    def clamp(self, x):
        return max(0.0, min(1.0, x))

    @ti.func
    def map(self, x):
        w = 0.0
        if x < self.c:
            w = self.wl
        else:
            w = self.wr
        return self.clamp((w - abs(self.clamp(x) - self.c)) / w * self.h)


jetR = ColorMap(1.5, .37, .37, .75)
jetG = ColorMap(1.5, .37, .37, .5)
jetB = ColorMap(1.5, .37, .37, .25)

bwrR = ColorMap(1.0, .25, 1, .5)
bwrG = ColorMap(1.0, .5, .5, .5)
bwrB = ColorMap(1.0, 1, .25, .5)

myR = ColorMap(1.0, 1, 0.0001, 0.0001)
myG = ColorMap(1.0, 0.0001, 1, 0.0001)
myB = ColorMap(1.0, 0.0001, 0.0001, 1)

@ti.func
def color_map(c):
    # return ti.Vector([R.map(c), G.map(c), B.map(c)])
    # return ti.Vector([jetR.map(c), jetG.map(c), jetB.map(c)])
    return ti.Vector([bwrR.map(c), bwrG.map(c), bwrB.map(c)])
    # return ti.Vector([myR.map(c), myG.map(c), myB.map(c)])

@ti.kernel
def transfer():
    c[0] = color_map(a[None])

if __name__ == "__main__":
    print("hallo test colormap!")

    transfer()
    print(c[0])

    pos[0] = ti.Vector([0.5, 0.5])
    window = ti.ui.Window("Try", (100, 100))
    canvas = window.get_canvas()
    while window.running:
        canvas.circles(pos, radius=0.1, per_vertex_color=c)
        window.show()



