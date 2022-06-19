import taichi as ti

# TODO: (just have a try) add coolwarm colormap
# TODO: try new struct_class of taichi to make some choices of colormap

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


# @ti.struct_class
# class cm:
#     R: ColorMap()
#     G: ColorMap()
#     B: ColorMap()

jetR = ColorMap(1.5, .37, .37, .75)
jetG = ColorMap(1.5, .37, .37, .5)
jetB = ColorMap(1.5, .37, .37, .25)

bwrR = ColorMap(1.0, .25, 1, .5)
bwrG = ColorMap(1.0, .5, .5, .5)
bwrB = ColorMap(1.0, 1, .25, .5)

coolwarmR = ColorMap(0.9, .25, 1, .5)
coolwarmG = ColorMap(0.9, .5, .5, .5)
coolwarmB = ColorMap(0.9, 1, .25, .5)

@ti.func
def color_map(c):
    # return ti.Vector([R.map(c), G.map(c), B.map(c)])
    # return ti.Vector([jetR.map(c), jetG.map(c), jetB.map(c)])
    # return ti.Vector([bwrR.map(c), bwrG.map(c), bwrB.map(c)])
    return ti.Vector([coolwarmR.map(c), coolwarmG.map(c), coolwarmB.map(c)])

