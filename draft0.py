import taichi as ti
import numpy as np
from eng.wcsph import *
from eng.soilsph import *

# TODO: make unit testing for basic functions of SPH

# ti.init(arch=ti.cpu, debug=True)
ti.init(arch=ti.gpu, packed=True, device_memory_GB=4)

if __name__ == "__main__":
    print("hallo tiSPHi!")
