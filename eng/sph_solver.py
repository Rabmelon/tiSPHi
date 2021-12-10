import taichi as ti

# TODO: test the import of engine

ti.require_version(0, 8, 1)

@ti.data_oriented
class SPHSolver:
    def __init__(self):
        print("Hallo, class SPH Solver!")
