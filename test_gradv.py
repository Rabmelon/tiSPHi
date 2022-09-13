import taichi as ti
from eng.particle_system import *
from eng.sph_solver import *
from eng.gguishow import *

ti.init(arch=ti.cpu, debug=True, default_fp=ti.f64, kernel_profiler=True)

class ChkGradV(SPHSolver):
    def __init__(self, particle_system, kernel):
        super().__init__(particle_system, kernel)
        print("Class check kernel starts to serve!")

        self.mass = self.ps.m_V * 1000

        self.fgv = ti.Matrix.field(2, 2, dtype=float)
        self.v_grad = ti.Matrix.field(2, 2, float)
        self.dv = ti.Matrix.field(2, 2, float)
        particle_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particle_node.place(self.fgv, self.v_grad, self.dv)

    @ti.kernel
    def init_value(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] < 10:
                # self.ps.val[p_i] = 1.0
                # self.ps.val[p_i] = self.ps.v[p_i].x
                # self.ps.val[p_i] = self.ps.v[p_i].y
                # self.ps.val[p_i] = self.fgv[p_i][0,0]
                # self.ps.val[p_i] = self.fgv[p_i][1,1]
                # self.ps.val[p_i] = self.v_grad[p_i][0,0]
                # self.ps.val[p_i] = self.v_grad[p_i][1,1]
                self.ps.val[p_i] = self.dv[p_i][0,0]
                # self.ps.val[p_i] = self.dv[p_i][0,1]
                # self.ps.val[p_i] = self.dv[p_i][1,0]
                # self.ps.val[p_i] = self.dv[p_i][1,1]

    @ti.func
    def calc_v(self, x):
        # res = ti.math.vec2(1.0/(5.0-x.x), -x.y/10.0)
        # res = ti.math.vec2(2*x.x+3*x.y, -3*x.x-x.y)
        res = ti.math.vec2(3*x.x*x.y, 2*x.x-x.y)
        return res

    @ti.func
    def calc_grad_v(self, x):
        # res = ti.math.mat2([[1.0/(5.0-x.x)**2, 0.0], [0.0, -1.0/10.0]])
        # res = ti.math.mat2([[2, 3], [-3, -1]])
        res = ti.math.mat2([[3*x.y, 3*x.x], [2, -1]])
        return res

    @ti.func
    def calc_fixed_v(self, p_j):
        tmp_v = ti.math.vec2(0.0)
        for k in range(self.ps.particle_neighbors_num[p_j]):
            p_k = self.ps.particle_neighbors[p_j, k]
            if self.ps.material[p_k] >= 10:
                continue
            v_k = self.ps.v[p_k]
            xjk = self.ps.x[p_j] - self.ps.x[p_k]
            tmp_v += self.mass / self.ps.density[p_k] * v_k * self.kernel(xjk)
            # tmp *= self.ps.MLS_beta[p_i][0] + self.ps.MLS_beta[p_i][1] * xij[0] + self.ps.MLS_beta[p_i][2] * xij[1]
        tmp_v *= self.CSPM_f[p_j]
        self.ps.v[p_j] = tmp_v


    @ti.kernel
    def calc_fv_aim(self):
        for p_i in range(self.ps.particle_num[None]):
            x_i = self.ps.x[p_i]
            self.ps.v[p_i] = self.calc_v(x_i)

    @ti.kernel
    def calc_fgv_aim(self):
        for p_i in range(self.ps.particle_num[None]):
            x_i = self.ps.x[p_i]
            self.fgv[p_i] = self.calc_grad_v(x_i)

    @ti.kernel
    def cal_v_grad(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            tmp = ti.math.vec2(0.0)
            v_g = ti.Matrix([[0.0 for _ in range(self.ps.dim)] for _ in range(self.ps.dim)])
            # for j in range(self.ps.particle_neighbors_num[p_i]):
            #     p_j = self.ps.particle_neighbors[p_i, j]
            #     if self.ps.material[p_j] == self.ps.material_dummy:
            #         self.ps.density[p_j] = 1000.0
            #         self.ps.v[p_j] = (1.0 - min(1.5, 1.0 + self.calc_d_BA_rec(p_i, p_j))) * self.ps.v[p_i]
            #         self.calc_fixed_v(p_j)
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]

                # if self.ps.material[p_j] == self.ps.material_dummy:
                #     # self.ps.v[p_j] = (1.0 - min(1.5, 1.0 + self.calc_d_BA_rec(p_i, p_j))) * self.ps.v[p_i]
                #     # self.calc_fixed_v(p_j)
                #     self.ps.v[p_j] = ti.math.vec2(0)

                if self.ps.material[p_j] == self.ps.material_dummy:
                #     # tmp = self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
                    continue
                # if self.ps.material[p_j] < 10:
                    # tmp = self.CSPM_L[p_i] @ self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])

                # tmp = self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
                tmp = self.CSPM_L[p_i] @ self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])

                v_g += (self.ps.v[p_j] - self.ps.v[p_i]) @ tmp.transpose()
            self.v_grad[p_i] = v_g * self.ps.m_V
            self.dv[p_i] = self.fgv[p_i] - self.v_grad[p_i]

    def step(self):
        self.ps.initialize_particle_system()
        self.calc_CSPM_L()
        self.calc_CSPM_f()
        self.calc_MLS_beta()

        self.calc_fv_aim()
        self.calc_fgv_aim()
        self.cal_v_grad()


if __name__ == "__main__":
    print("hallo test velocity gradient accuracy!")

    screen_to_world_ratio = 80   # exp: world = (150, 100), ratio = 4, screen res = (600, 400)
    rec_world = [6, 5]   # a rectangle world start from (0, 0) to this pos
    particle_radius = 0.1
    case1 = ParticleSystem(rec_world, particle_radius)
    case1.gen_boundary_dummy()
    case1.add_cube(lower_corner=[0, 0], cube_size=[4, 4], material=1, density=1000)

    solver = ChkGradV(case1, 2)
    gguishow(case1, solver, rec_world, screen_to_world_ratio,
             step_ggui=1, pause_flag=0, stop_step=2,
            #  save_msg=1,
            #  iparticle=[366, 576, 746, 765], # with boundary
            #  iparticle=[0, 210, 380, 399], # without boundary
             kradius=1.05, color_title="dfgv")
