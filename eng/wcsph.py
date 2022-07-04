import taichi as ti
from .sph_solver import SPHSolver
from eng.wcsesph import *
from eng.wclfsph import *
from eng.wcrksph import *

class WCSPHSolver(SPHSolver):
    def __init__(self, particleSystem, TDmethod, kernel, visco, stiff, expo, colorPara):
        super().__init__(particleSystem, TDmethod, kernel)
        print("Hallo, class WCSPH Solver starts to serve!")

        # Basic paras
        self.density_0 = 1000.0  # reference density
        self.mass = self.ps.m_V * self.density_0
        self.viscosity = visco  # viscosity

        self.pressure = ti.field(dtype=float)
        self.d_velocity = ti.Vector.field(self.ps.dim, dtype=float)
        self.d_density = ti.field(dtype=float)
        self.v1234 = ti.Vector.field(self.ps.dim, dtype=float)
        self.F = ti.Vector.field(self.ps.dim, dtype=float)
        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.pressure, self.d_velocity, self.d_density, self.v1234)
        particle_node.dense(ti.j, 4).place(self.F)

        # Two paras in taichiWCSPH code
        self.stiffness = stiff   # k1
        self.exponent = expo     # k2

        self.colorPara = colorPara

    @ti.kernel
    def init_value(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] < 10:
                # self.ps.val[p_i] = self.ps.u[p_i].norm()
                # self.ps.val[p_i] = -self.ps.x[p_i][1]
                self.ps.val[p_i] = self.ps.density[p_i]
                # self.ps.val[p_i] = self.pressure[p_i]
                # self.ps.val[p_i] = p_i
