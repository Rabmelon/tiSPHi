import taichi as ti
import numpy as np

# TODO: generate SPH kernel functions and the derivate of kernel functions
# TODO: check boundary treatment
# TODO: add different advection methods

epsilon = 1e-5

@ti.data_oriented
class SPHSolver:
    def __init__(self, particle_system, TDmethod):
        print("Class SPH Solver starts to serve!")
        self.ps = particle_system
        self.TDmethod = TDmethod # 1 for Symp Euler; 2 for RK4
        self.g = -9.81
        self.dt = ti.field(float, shape=())
        self.dt[None] = 2e-4

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
        if r_norm > epsilon and q <= 1.0:
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


    ###########################################################################
    # Time integration
    ###########################################################################
