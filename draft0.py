import taichi as ti
import numpy as np
from eng.wcsph import *
from eng.soilsph import *

# TODO: make unit testing for basic functions of SPH

# ti.init(arch=ti.cpu, debug=True)
ti.init(arch=ti.cuda, packed=True, device_memory_fraction=0.75)     # MEMORY max 4G in GUT, 6G in Legion

if __name__ == "__main__":
    print("hallo tiSPHi!")
