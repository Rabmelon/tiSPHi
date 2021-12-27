import taichi as ti
import numpy as np
from .particle_system import ParticleSystem

@ti.data_oriented
class SPHSolver:
    def __init__(self, particle_system):
        print("Hallo, class SPH Solver starts to serve!")
        self.ps = particle_system
        self.g = -9.81  # Gravity
        self.density_0 = 1000.0  # reference density
        self.mass = self.ps.m_V * self.density_0
        self.dt = ti.field(float, shape=())
        self.dt[None] = 5e-4

    # value of cubic spline smoothing kernel: PPT 10 p52
    @ti.func
    def cubic_kernel(self, r):
        res = ti.cast(0.0, ti.f32)
        h = self.ps.support_radius
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / 7 / np.pi
        elif self.ps.dim == 3:
            k = 8 / np.pi
        k /= h ** self.ps.dim
        r_norm = r.norm()
        q = r_norm / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                res = k * 2 * ti.pow(1 - q, 3.0)
        return res

    # derivative of cubic spline smoothing kernel
    @ti.func
    def cubic_kernel_derivative(self, r):
        h = self.ps.support_radius
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / 7 / np.pi
        elif self.ps.dim == 3:
            k = 8 / np.pi
        k = 6. * k / h ** self.ps.dim
        r_norm = r.norm()
        q = r_norm / h
        res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        if r_norm > 1e-5 and q <= 1.0:
            grad_q = r / (r_norm * h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
        return res

    # Collision factor, assume roughly (1-c_f)*velocity loss after collision
    @ti.func
    def simulate_collisions(self, p_i, vec, d):
        c_f = 0.7
        self.ps.x[p_i] += vec * d
        self.ps.v[p_i] -= (1.0 + c_f) * (self.ps.v[p_i].dot(vec)) * vec

    # Treat the boundary problems
    @ti.kernel
    def enforce_boundary(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.dim == 2:
                if self.ps.material[p_i] == self.ps.material_water:
                    pos = self.ps.x[p_i]
                    if pos[0] < self.ps.padding:
                        self.simulate_collisions(p_i, ti.Vector([1.0, 0.0]),
                                                 self.ps.padding - pos[0])
                    if pos[0] > self.ps.bound[0] - self.ps.padding:
                        self.simulate_collisions(p_i, ti.Vector([-1.0, 0.0]),
                            pos[0] - (self.ps.bound[0] - self.ps.padding))
                    if pos[1] > self.ps.bound[1] - self.ps.padding:
                        self.simulate_collisions(p_i, ti.Vector([0.0, -1.0]),
                            pos[1] - (self.ps.bound[1] - self.ps.padding))
                    if pos[1] < self.ps.padding:
                        self.simulate_collisions(p_i, ti.Vector([0.0, 1.0]),
                                                 self.ps.padding - pos[1])

    def substep(self):
        pass

    def step(self):
        self.ps.initialize_particle_system()
        self.substep()
        self.enforce_boundary()
