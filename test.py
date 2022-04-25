import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

# TODO: Test the D-P stress check and update

# basic property
mat_water = 1
mat_soil = 2
mat_elastic = 3
mat_plastic = 4
materials = {
    'water': mat_water,
    'soil': mat_soil,
    'elastic': mat_elastic,
    'plastic': mat_plastic
}

# input
def initPara():
    print('initialising...')


# cal the deviatoric stress tensor s


# check one


# update one


# cal all


# draw


# main
if __name__ == "__main__":
    print('Hallo!')
