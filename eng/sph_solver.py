import taichi as ti
import numpy as np

# TODO: generate SPH kernel functions and the derivate of kernel functions
# TODO: check boundary treatment
# TODO: add different advection methods

@ti.data_oriented
class SPHSolver:
    def __init__(self, particleSystem, kernel):
        print("Class SPH Solver starts to serve!")
        self.ps = particleSystem
        self.flagKernel = kernel   # 1 for cubic-spline, 2 for Wenland, 3 for
        self.g = -9.81          # gravity, m/s2
        self.I = ti.Matrix(np.eye(self.ps.dim))
        self.dt = ti.field(float, shape=())
        self.dt[None] = 1e-5
        self.dt_min = 1e-6
        self.vsound = 35.0
        self.epsilon = 1e-16
        self.alertratio = 1.25

        self.CSPM_L = ti.Matrix.field(self.ps.dim, self.ps.dim, dtype=float)     # the normalised matrix
        self.CSPM_f = ti.field(dtype=float)
        self.MLS_f = ti.field(dtype=float)
        particles_node = ti.root.dense(ti.i, self.ps.particle_max_num)
        particles_node.place(self.CSPM_L, self.CSPM_f, self.MLS_f)

    ###########################################################################
    # colored value
    ###########################################################################
    @ti.kernel
    def init_value(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] < 10:
                self.ps.val[p_i] = 0.0

    ###########################################################################
    # Assist
    ###########################################################################
    @ti.kernel
    def assign_x0(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[p_i] < 10:
                self.ps.x0[p_i] = self.ps.x[p_i]

    ###########################################################################
    # Kernel correction
    ###########################################################################
    @ti.kernel
    def calc_CSPM_f(self):
        for p_i in range(self.ps.particle_num[None]):
            tmp_CSPM_f = 0.0
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                tmp_CSPM_f += self.ps.m_V * self.kernel(self.ps.x[p_i] - self.ps.x[p_j])
            self.CSPM_f[p_i] = 1.0 / tmp_CSPM_f

    @ti.kernel
    def calc_CSPM_L(self):
        for p_i in range(self.ps.particle_num[None]):
            tmp_CSPM_L = ti.Matrix([[0.0 for _ in range(self.ps.dim)] for _ in range(self.ps.dim)])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                tmp = self.kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
                tmp_CSPM_L += self.ps.m_V * (self.ps.x[p_j] - self.ps.x[p_i]) @ tmp.transpose()
            self.CSPM_L[p_i] = tmp_CSPM_L.inverse()

    @ti.kernel
    def calc_MLS_f(self):
        for p_i in range(self.ps.particle_num[None]):
            x_i = self.ps.x[p_i]
            tmp_MLS_f = 0.0
            multi = ti.Vector([1.0, 0.0, 0.0])
            beta = ti.Vector([0.0, 0.0, 0.0])
            p = ti.Vector([0.0, 0.0, 0.0])
            A = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                p = ti.Vector([1.0, x_i[0]-x_j[0], x_i[1]-x_j[1]])
                A = (p @ p.transpose()).inverse()
                beta += self.ps.m_V * A * self.kernel(x_i - x_j) @ multi
            self.MLS_f[p_i] = tmp_MLS_f


    ###########################################################################
    # Kernel functions
    ###########################################################################
    @ti.func
    def kernel(self, r):
        res = ti.cast(0.0, ti.f32)
        if self.flagKernel == 1:
            res = self.cubic_kernel(r)
        elif self.flagKernel == 2:
            res = self.WendlandC2_kernel(r)
        return res

    @ti.func
    def kernel_derivative(self, r):
        res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        if self.flagKernel == 1:
            res = self.cubic_kernel_derivative(r)
        elif self.flagKernel == 2:
            res = self.WendlandC2_kernel_derivative(r)
        return res

    # Cubic spline kernel
    @ti.func
    def cubic_kernel(self, r):
        res = ti.cast(0.0, ti.f32)
        h1 = 1.0 / self.ps.smoothing_len
        k = 1.0 if self.ps.dim == 1 else 15.0 / 7.0 / np.pi if self.ps.dim == 2 else 3.0 / 2.0 / np.pi
        k *= h1**self.ps.dim
        r_norm = r.norm()
        q = r_norm * h1
        if r_norm > self.epsilon and q <= 2.0:
            if q <= 1.0:
                q2 = q * q
                q3 = q2 * q
                res = k * (0.5 * q3 - q2 + 2.0 / 3.0)
            else:
                res = k / 6.0 * ti.pow(2.0 - q, 3.0)
        return res

    @ti.func
    def cubic_kernel_derivative(self, r):
        res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        h1 = 1.0 / self.ps.smoothing_len
        k = 1.0 if self.ps.dim == 1 else 15.0 / 7.0 / np.pi if self.ps.dim == 2 else 3.0 / 2.0 / np.pi
        k *= 6.0 * h1**self.ps.dim
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
        h1 = 1.0 / self.ps.smoothing_len
        k = 7.0 / (4.0 * np.pi) if self.ps.dim == 2 else 21.0 / (2.0 * np.pi) if self.ps.dim == 3 else 0.0
        k *= h1**self.ps.dim
        r_norm = r.norm()
        q = r_norm * h1
        if r_norm > self.epsilon and q <= 2.0:
            q1 = 1.0 - 0.5 * q
            res = k * ti.pow(q1, 4.0) * (1.0 + 2.0 * q)
        return res

    @ti.func
    def WendlandC2_kernel_derivative(self, r):
        res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        h1 = 1.0 / self.ps.smoothing_len
        k = 7.0 / (4.0 * np.pi) if self.ps.dim == 2 else 21.0 / (2.0 * np.pi) if self.ps.dim == 3 else 0.0
        k *= h1**self.ps.dim
        r_norm = r.norm()
        q = r_norm * h1
        if r_norm > self.epsilon and q <= 2.0:
            q1 = 1.0 - 0.5 * q
            res = k * ti.pow(q1, 3.0) * (-5.0 * q) * h1 * r / r_norm
        return res

    ###########################################################################
    # Boundary treatment
    ###########################################################################
    # Collision factor, assume roughly (1-c_f)*velocity loss after collision
    @ti.func
    def simulate_collisions(self, p_i, vec, d):
        # if self.ps.material[p_i] < 10:
        # assert d > self.ps.grid_size, 'My Error 2: particle goes out of the padding! d = %f, vec = [%f, %f], xo[%d] = [%f, %f]' % (d, vec[0], vec[1], p_i, self.ps.x[p_i][0], self.ps.x[p_i][1])
        c_f = 0.7
        self.ps.x[p_i] += (1.0 + c_f) * vec * d
        self.ps.v[p_i] -= (1.0 + c_f) * (self.ps.v[p_i].dot(vec)) * vec

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

    # Compute the distance between particle and boundary
    @ti.func
    def calc_d_BA(self, p_i, p_j):
        x_i = self.ps.x[p_i]
        x_j = self.ps.x[p_j]
        boundary = ti.Vector([self.ps.world[1], 0.0, self.ps.world[0], 0.0])
        db_i = ti.Vector([x_i[1] - boundary[0], x_i[1] - boundary[1], x_i[0] - boundary[2], x_i[0] - boundary[3]])
        db_j = ti.Vector([x_j[1] - boundary[0], x_j[1] - boundary[1], x_j[0] - boundary[2], x_j[0] - boundary[3]])

        flag_b = db_i * db_j
        flag_dir = flag_b < 0

        if flag_dir.sum() > 1:
            flag_choose = abs(flag_dir * db_i)
            tmp_max = 0.0
            for i in ti.static(range(4)):
                tmp_max = max(tmp_max, flag_choose[i])
            flag_choose -= tmp_max
            flag_choose = flag_choose == 0.0
            flag_dir -= int(flag_choose)     # will cause a warning: Local store may lose precision & Atomic add (i32 to f32) may lose precision

        d_A = abs(db_i.dot(flag_dir))
        d_B = abs(db_j.dot(flag_dir))
        return d_B / (d_A + self.epsilon)

    # repulsive forces
    @ti.func
    def calc_repulsive_force(self, r, vsound):
        r_norm = r.norm()
        chi = 1.0 - r_norm / (1.5 * self.ps.particle_radius) if (r_norm >= 0.0 and r_norm < 1.5 * self.ps.particle_radius) else 0.0
        gamma = r_norm / (0.75 * self.ps.smoothing_len)
        f = 0.0
        if gamma > 0 and gamma <= 2 / 3:
            f = 2 / 3
        elif gamma > 2 / 3 and gamma <= 1:
            f = 2 * gamma - 1.5 * gamma**2
        elif gamma > 1 and gamma < 2:
            f = 0.5 * (2 - gamma)**2
        res = 0.01 * vsound**2 * chi * f / (r_norm**2) * r
        return res

    ###########################################################################
    # Time integration
    ###########################################################################
    def substep_SympEuler(self):
        # one single pipeline
        pass

    def substep_LeapFrog(self):
        # cal rho and u after 0.5dt
        # cal drho du
        # update all in dt
        pass

    def LF_one_step(self):
        pass

    def substep_RK4(self):
        # init F and other paras
        self.advect_RK4()

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

    def substep():
        pass

    def step(self):
        self.ps.initialize_particle_system()
        self.calc_CSPM_L()
        self.calc_CSPM_f()
        self.substep()
        # if self.TDmethod == 1:
        #     self.substep_SympEuler()
        # elif self.TDmethod == 2:
        #     self.substep_LeapFrog()
        # elif self.TDmethod == 4:
        #     self.substep_RK4()
        self.enforce_boundary()   # Needed in WCSPH
