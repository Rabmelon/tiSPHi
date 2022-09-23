import taichi as ti
from eng.gguishow import *
from eng.particle_system import *
from eng.sph_solver import *

ti.init()

class ShockPipeSPHSolver(SPHSolver):
    def __init__(self, particleSystem, kernel):
        super().__init__(particleSystem, kernel)

        self.e = ti.field(dtype=float)
        self.p = ti.field(dtype=float)
        self.m_V = ti.field(float)
        self.mass = ti.field(float)
        self.dd = ti.field(float)
        self.de = ti.field(float)
        self.dv = ti.Vector.field(2, float)

        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.e, self.p, self.m_V, self.mass, self.dd, self.de, self.dv)

    @ti.kernel
    def init_value(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] < 10:
                # self.ps.val[p_i] = p_i
                # self.ps.val[p_i] = self.ps.v[p_i].norm()
                self.ps.val[p_i] = self.ps.density[p_i]
                # self.ps.val[p_i] = self.p[p_i]

    @ti.kernel
    def assign_init_value(self):
        for p_i in range(0, self.ps.particle_num[None]/2):
            if self.ps.material[p_i] < 10:
                self.e[p_i] = 2.5
                self.p[p_i] = 1.0
        for p_i in range(self.ps.particle_num[None]/2, self.ps.particle_num[None]):
            if self.ps.material[p_i] < 10:
                self.e[p_i] = 1.795
                self.p[p_i] = 0.1795
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] < 10:
                self.ps.v[p_i] = 0.0
                self.m_V[p_i] = self.ps.m_V
                self.mass[p_i] = self.m_V[p_i] * self.ps.density[p_i]

    @ti.kernel
    def calc_densities(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] < 10:
                d = 0.0
                for j in range(self.ps.particle_neighbors_num[p_i]):
                    p_j = self.ps.particle_neighbors[p_i, j]
                    tmp = self.kernel(self.ps.x[p_i] - self.ps.x[p_j]) * self.CSPM_f[p_i]
                    d += self.mass[p_j] * tmp
                    if p_i == 10:
                        print(self.m_V[p_j])
                self.ps.density[p_i] = d

    @ti.func
    def calc_pressure(self, density, energy):
        gamma = 1.4
        res = (gamma - 1.0) * density * energy
        return res

    @ti.kernel
    def calc_d(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] < 10:
                self.p[p_i] = self.calc_pressure(self.ps.density[p_i], self.e[p_i])
                dv = ti.math.vec2(0)
                de = 0.0
                for j in range(self.ps.particle_neighbors_num[p_i]):
                    p_j = self.ps.particle_neighbors[p_i, j]
                    self.p[p_j] = self.calc_pressure(self.ps.density[p_j], self.e[p_j])
                    tmp = self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
                    dv -= self.mass[p_j] * (self.p[p_i] / self.ps.density[p_i]**2 + self.p[p_j] / self.ps.density[p_j]**2) * tmp
                    de += 0.5 * self.mass[p_j] * (self.p[p_i] / self.ps.density[p_i]**2 + self.p[p_j] / self.ps.density[p_j]**2) * ((self.ps.v[p_i] - self.ps.v[p_j]).dot(tmp))
                self.dv[p_i] = dv
                self.de[p_i] = de

    @ti.kernel
    def advect(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] < 10:
                self.p[p_i] = self.calc_pressure(self.ps.density[p_i], self.e[p_i])
                self.e[p_i] += self.de[p_i] * self.dt[None]
                self.ps.v[p_i] += self.dv[p_i] * self.dt[None]

    def substep(self):
        self.calc_densities()
        self.calc_d()
        self.advect()


if __name__ == "__main__":
    print("hallo ti test shock pipe!")

    rec_world = [1.2, 1.0]        # a rectangle world start from (0, 0) to this pos
    screen_to_world_ratio = 800 / max(rec_world)   # exp: world = (150m, 100m), ratio = 4, screen res = (600, 400)
    r = 0.005

    case1 = ParticleSystem(world=rec_world, radius=r)
    case1.add_cube(lower_corner=[0.0, 0.5-r], cube_size=[0.6, r*2], density=1, material=1)
    case1.add_cube(lower_corner=[0.6, 0.5-r], cube_size=[0.6, r*2], density=0.25, material=1)
    case1.dim = 1

    solver = ShockPipeSPHSolver(case1, kernel=1)
    gguishow(case1, solver, rec_world, screen_to_world_ratio,
             pause_flag=1, stop_step=1, step_ggui=1, exit_flag=0,
             kradius=1.5, grid_line=0.1, color_title=2,
             given_max=-1, given_min=-1, fix_max=1, fix_min=1)

    '''
    color title:
    1 index
    2 density
        21 d density
    3 velocity norm
        31 x 32 y 33 z
    4 position
        41 x 42 y 43 z
    5 stress
        51 xx 52 yy 53 zz 54 xy 55 yz 56 zx
        57 hydrostatic stress 58 deviatoric stress
    6 strain
        61 equivalent plastic strain
    7 displacement
    8 pressure
    otherwise Null
    '''
