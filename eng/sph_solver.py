import taichi as ti
import numpy as np

# TODO: generate SPH kernel functions and the derivate of kernel functions
# TODO: check boundary treatment
# TODO: add different advection methods

@ti.data_oriented
class SPHSolver:
    def __init__(self, particle_system, TDmethod, kernel):
        print("Class SPH Solver starts to serve!")
        self.ps = particle_system
        self.TDmethod = TDmethod # 1 for Symp Euler, 2 for RK4
        self.flag_kernel = kernel   # 1 for cubic-spline, 2 for Wenland, 3 for
        self.g = -9.81          # gravity, m/s2
        self.usound = 60        # speed of sound, m/s
        self.usound2 = self.usound ** 2
        self.I = ti.Matrix(np.eye(self.ps.dim))
        self.dt = ti.field(float, shape=())
        # self.dt[None] = 2e-5    # "ti video -f125" will be good to make the video 2 times slower than calculation (8s simulation and 16s video, 2000 frames / 8*2s = 125fps)
        self.dt[None] = ti.max(1e-6, 0.2 * self.ps.smoothing_len / self.usound)  # CFL
        self.epsilon = 1e-16

    ###########################################################################
    # Assist
    ###########################################################################
    @ti.kernel
    def init_value(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] < 10:
                self.ps.val[p_i] = 0.0

    @ti.kernel
    def cal_L(self):
        for p_i in range(self.ps.particle_num[None]):
            x_i = self.ps.x[p_i]
            tmpL = ti.Matrix([[0.0 for _ in range(self.ps.dim)] for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                tmp = self.kernel_derivative(x_i - x_j)
                tmpL += self.ps.m_V * (x_j - x_i) @ tmp.transpose()
            self.ps.L[p_i] = tmpL.inverse()

    ###########################################################################
    # Kernel functions
    ###########################################################################
    @ti.func
    def kernel(self, r):
        res = ti.cast(0.0, ti.f32)
        if self.flag_kernel == 1:
            res = self.cubic_kernel(r)
        elif self.flag_kernel == 2:
            res = self.WendlandC2_kernel(r)
        return res

    @ti.func
    def kernel_derivative(self, r):
        res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        if self.flag_kernel == 1:
            res = self.cubic_kernel_derivative(r)
        elif self.flag_kernel == 2:
            res = self.WendlandC2_kernel_derivative(r)
        return res

    # Cubic spline kernel
    @ti.func
    def cubic_kernel(self, r):
        res = ti.cast(0.0, ti.f32)
        h1 = 1 / self.ps.smoothing_len
        k = 1 if self.ps.dim == 1 else 15 / 7 / np.pi if self.ps.dim == 2 else 3 / 2 / np.pi
        k *= h1**self.ps.dim
        r_norm = r.norm()
        q = r_norm * h1
        if r_norm > self.epsilon and q <= 2.0:
            if q <= 1.0:
                q2 = q * q
                q3 = q2 * q
                res = k * (0.5 * q3 - q2 + 2 / 3)
            else:
                res = k / 6 * ti.pow(2 - q, 3.0)
        return res

    @ti.func
    def cubic_kernel_derivative(self, r):
        res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        h1 = 1 / self.ps.smoothing_len
        k = 1 if self.ps.dim == 1 else 15 / 7 / np.pi if self.ps.dim == 2 else 3 / 2 / np.pi
        k *= 6. * h1**self.ps.dim
        r_norm = r.norm()
        q = r_norm * h1
        if r_norm > self.epsilon and q <= 2.0:
            grad_q = r / r_norm * h1
            if q <= 1.0:
                res = k * q * (3.0 / 2.0 * q - 2.0) * grad_q
            else:
                factor = 2.0 - q
                res = k * (-0.5 * factor * factor) * grad_q
        return res

    # Wendland C2 kernel
    @ti.func
    def WendlandC2_kernel(self, r):
        res = ti.cast(0.0, ti.f32)
        h1 = 1 / self.ps.smoothing_len
        k = 7 / (4 * np.pi) if self.ps.dim == 2 else 21 / (2 * np.pi) if self.ps.dim == 3 else 0.0
        k *= h1**self.ps.dim
        r_norm = r.norm()
        q = r_norm * h1
        if r_norm > self.epsilon and q <= 2.0:
            q1 = 1 - 0.5 * q
            res = k * ti.pow(q1, 4) * (1 + 2 * q)
        return res

    @ti.func
    def WendlandC2_kernel_derivative(self, r):
        res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        h1 = 1 / self.ps.smoothing_len
        k = 7 / (4 * np.pi) if self.ps.dim == 2 else 21 / (2 * np.pi) if self.ps.dim == 3 else 0.0
        k *= h1**self.ps.dim
        r_norm = r.norm()
        q = r_norm * h1
        if r_norm > self.epsilon and q <= 2.0:
            q1 = 1 - 0.5 * q
            res = k * ti.pow(q1, 3) * (-5 * q) * h1 * r / r_norm
        return res

    ###########################################################################
    # Boundary treatment
    ###########################################################################
    # Collision factor, assume roughly (1-c_f)*velocity loss after collision
    @ti.func
    def simulate_collisions(self, p_i, vec, d):
        # if self.ps.material[p_i] < 10:
        assert d > self.ps.grid_size, 'My Error 2: particle goes out of the padding! d = %f, vec = [%f, %f], xo[%d] = [%f, %f]' % (d, vec[0], vec[1], p_i, self.ps.x[p_i][0], self.ps.x[p_i][1])
        c_f = 0.7
        self.ps.x[p_i] += (1.0 + c_f) * vec * d
        self.ps.u[p_i] -= (1.0 + c_f) * (self.ps.u[p_i].dot(vec)) * vec

    @ti.kernel
    def enforce_boundary(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.dim == 2:
                if self.ps.material[p_i] < 10:
                    pos = self.ps.x[p_i]
                    if pos[0] < 0:
                        self.simulate_collisions(p_i, ti.Vector([1.0, 0.0]), -pos[0])
                    if pos[0] > self.ps.world[0]:
                        self.simulate_collisions(p_i, ti.Vector([-1.0, 0.0]), pos[0] - self.ps.world[0])
                    if pos[1] > self.ps.world[1]:
                        self.simulate_collisions(p_i, ti.Vector([0.0, -1.0]), pos[1] - self.ps.world[1])
                    if pos[1] < 0:
                        self.simulate_collisions(p_i, ti.Vector([0.0, 1.0]), -pos[1])

    ###########################################################################
    # Time integration
    ###########################################################################
    def substep_SympEuler(self):
        pass

    def RK4_one_step(self, m):
        # compute one step RK4 functions here
        # self.compute_F(m)
        pass

    def advect_RK4(self):
        for m in ti.static(range(4)):
            if m == 0:
                self.update_v_1(m)
            elif m < 4:
                self.update_v_234(m)
            self.RK4_one_step(m)
        # self.update_vel_pos()

    def substep_RK4(self):
        # init F and other paras
        self.advect_RK4()

    def step(self):
        self.ps.initialize_particle_system()
        if self.TDmethod == 1:
            self.substep_SympEuler()
        elif self.TDmethod == 2:
            self.substep_RK4()
        self.enforce_boundary()
