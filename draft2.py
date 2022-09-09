import taichi as ti
from eng.gguishow import *
from eng.particle_system import *
from eng.choose_solver import *

# ti.init(arch=ti.cpu, debug=True, default_fp=ti.f64, cpu_max_num_threads=1)
ti.init(arch=ti.cuda, packed=True, device_memory_fraction=0.3, default_fp=ti.f64)

if __name__ == "__main__":
    print("hallo tiSPHi! This is for 1d shock wave test!")
