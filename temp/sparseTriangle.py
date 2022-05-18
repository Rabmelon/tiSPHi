import taichi as ti
import time
ti.init(arch = ti.gpu)
c窗口边长 = 1024
c窗口尺寸 = (c窗口边长, c窗口边长)
c网格边长 = 4**4
c网格尺寸 = (c网格边长, c网格边长)
c缩放 = c窗口边长 // c网格边长
c时间间隔 = 1 / 100

@ti.data_oriented
class C三角形:
    def __init__(self):
        self.m点 = ti.Vector.field(2, dtype = float, shape = 3)
    @ti.kernel
    def compute(self, t: float):
        #随便计算三角形的点
        for i in self.m点:
            p = ti.Vector([ti.sin(t), ti.cos(t)])
            p = ti.Matrix.rotation2d(6.2831853 * i / 3) @ p
            p = p * (c网格边长 * 0.4) + (c网格边长 * 0.5)
            self.m点[i] = p
    @ti.func
    def contain_point(self, p)->bool:
        #判断点是否在三角形内，参考 https://blog.nowcoder.net/n/385dfbad3f2a4eaaabc50347c472aa50
        pa = self.m点[0] - p
        pb = self.m点[1] - p
        pc = self.m点[2] - p
        t1 = vector_cross(pa, pb)
        t2 = vector_cross(pb, pc)
        t3 = vector_cross(pc, pa)
        return t1 * t2 >= 0 and t1 * t3 >= 0 and t2 * t3 >= 0
v三角形 = C三角形()

v图像 = ti.Vector.field(3, dtype = float, shape = c窗口尺寸)
v网格 = ti.field(dtype = float)
v大块 = ti.root.pointer(ti.ij, 4)
v中块 = v大块.pointer(ti.ij, 4)
v小块 = v中块.pointer(ti.ij, 4)
v小块.dense(ti.ij, 4).place(v网格)
@ti.func
def vector_cross(a, b):
    #向量叉乘
    return a[0] * b[1] - a[1] * b[0]
@ti.kernel
def activate():
    for i, j in ti.ndrange(c网格边长, c网格边长):
        p = ti.Vector([i, j])
        if v三角形.contain_point(p):
            v网格[i, j] = 1
@ti.kernel
def paint():
    for i, j in ti.ndrange(c窗口边长, c窗口边长):
        t = v网格[i, j]
        block1_index = ti.rescale_index(v网格, v大块, [i, j])
        block2_index = ti.rescale_index(v网格, v中块, [i, j])
        block3_index = ti.rescale_index(v网格, v小块, [i, j])
        t += ti.is_active(v大块, block1_index)
        t += ti.is_active(v中块, block2_index)
        t += ti.is_active(v小块, block3_index)
        v = 1 - t / 4
        c = (v, v, v)
        for x, y in ti.ndrange(c缩放, c缩放):   #缩放
          v图像[i * c缩放 + x, j * c缩放 + y] = c

v窗口 = ti.ui.Window("三角形稀疏", res = c窗口尺寸)
v画布 = v窗口.get_canvas()
v运行时间 = 0
while v窗口.running:
    v运行时间 += c时间间隔
    #计算
    v大块.deactivate_all()
    v三角形.compute(v运行时间)
    activate()
    paint()
    #显示
    v画布.set_image(v图像)
    v窗口.show()
    time.sleep(c时间间隔)