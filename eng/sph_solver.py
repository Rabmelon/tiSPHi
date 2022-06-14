import taichi as ti
import numpy as np

# TODO: generate SPH kernel functions and the derivate of kernel functions
# TODO: check boundary treatment
# TODO: add different advection methods

@ti.data_oriented
class SPHSolver:
    def __init__(self, particle_system, TDmethod):
        print("Class SPH Solver starts to serve!")
        self.ps = particle_system
        self.TDmethod = TDmethod # 1 for Symp Euler; 2 for RK4
        self.g = -9.81
        self.I = ti.Matrix(np.eye(self.ps.dim))
        self.dt = ti.field(float, shape=())
        self.dt[None] = 2e-5    # "ti video -f125" will be good to make the video 2 times slower than calculation (8s simulation and 16s video, 2000 frames / 8*2s = 125fps)
        self.epsilon = 1e-16

    ###########################################################################
    # Kernel functions
    ###########################################################################
    # Cubic spline kernel
    @ti.func
    def cubic_kernel(self, r):
        res = ti.cast(0.0, ti.f32)
        k = 4 / 3 if self.ps.dim == 1 else 40 / 7 / np.pi if self.ps.dim == 2 else 8 / np.pi
        k /= self.ps.support_radius**self.ps.dim
        r_norm = r.norm()
        q = r_norm / self.ps.support_radius
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0*q3-6.0*q2+1)
            else:
                res = k*2*ti.pow(1-q, 3.0)
        return res

    @ti.func
    def cubic_kernel_derivative(self, r):
        res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        k = 4 / 3 if self.ps.dim == 1 else 40 / 7 / np.pi if self.ps.dim == 2 else 8 / np.pi
        k *= 6. / self.ps.support_radius**self.ps.dim
        r_norm = r.norm()
        q = r_norm / self.ps.support_radius
        if r_norm > self.epsilon and q <= 1.0:
            grad_q = r / (r_norm * self.ps.support_radius)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
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
        self.ps.u[p_i] -= (1.0 + c_f) * (self.ps.u[p_i].dot(vec)) * vec

    # Treat the boundary problems
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

    def substep_RK4(self):
        pass

    def step(self):
        self.ps.initialize_particle_system()
        if self.TDmethod == 1:
            self.substep_SympEuler()
        elif self.TDmethod == 2:
            self.substep_RK4()
        self.enforce_boundary()
