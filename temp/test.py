import taichi as ti
import numpy as np
from show import *

ti.init(arch=ti.cpu)

if __name__ == "__main__":
    print("hallo tiSPHi!")

    # init particle system paras, world unit is cm (BUT not cm actually! maybe still m)
    screen_to_world_ratio = 4   # exp: world = (150, 100), ratio = 4, screen res = (600, 400)
    world = [150, 100]
    write_to_disk = False


    guishow(2, world, screen_to_world_ratio, write_to_disk)
