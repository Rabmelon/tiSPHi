from eng.configer_builder import SimConfiger
from eng.particle_system import ParticleSystem
from eng.solver_sph_wc import WCSPHSolver
from eng.solver_sph_muI import MUISPHSolver
from eng.solver_sph_dp import DPSPHSolver

class Simulation:
    def __init__(self, config: SimConfiger) -> None:
        self.cfg = config
        self.ps = ParticleSystem(self.cfg)
        self.solver_type = self.cfg.get_cfg("simulationMethod")
        self.solver = self.build_solver()

    def build_solver(self):
        if self.solver_type == 1:
            return WCSPHSolver(self.ps)
        elif self.solver_type == 2:
            return MUISPHSolver(self.ps)
        elif self.solver_type == 3:
            return DPSPHSolver(self.ps)
        else:
            raise NotImplementedError(f"Solver type {self.solver_type} has not been implemented.")
