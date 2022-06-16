import taichi as ti
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import reduce    # 整数：累加；字符串、列表、元组：拼接。lambda为使用匿名函数

ti.init(arch=ti.cpu)

N = 10
val = ti.field(float, shape=N)
valmax = ti.field(float, shape=())
valmin = ti.field(float, shape=())
color = ti.Vector.field(3, dtype=float, shape=N)

@ti.kernel
def setval():
    for i in range(N):
        val[i] = ti.random() * 10
        print(val[i])

@ti.kernel
def v_maxmin():
    vmax = -float('Inf')
    vmin = float('Inf')
    for i in range(N):
        ti.atomic_max(vmax, val[i])
        ti.atomic_min(vmin, val[i])
    valmax[None] = vmax
    valmin[None] = vmin

@ti.kernel
def set_color():
    vrange = valmax[None] - valmin[None]
    vrange1 = 1 / vrange
    cmap = mpl.cm.coolwarm
    norm = mpl.colors.Normalize(vmax=valmax[None], vmin=valmin[None])
    for i in range(N):
        color[i] = ti.Vector([0.0, 0.0, 0.0])
        print('color: ', color[i])


if __name__ == "__main__":
    print("hallo tiSPHi!")

    setval()
    v_maxmin()
    set_color()

# distance_list = [-1.1, 0.3, 0.4, 0.5, 1.2, 6, 8.1, 0.9, 5, 0.7]

# min_val, max_val = min(distance_list), max(distance_list)

# # use the coolwarm colormap that is built-in, and goes from blue to red
# cmap = mpl.cm.coolwarm
# norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)

# # convert your distances to color coordinates
# color_list = cmap(distance_list)

# fig, ax = plt.subplots()
# cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, ticks = sorted(distance_list), orientation='horizontal')
# cb.set_label('Distance (least to greatest)')
# ax.tick_params(axis='x', rotation=89)

# plt.show()
