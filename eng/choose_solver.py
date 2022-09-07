from eng.wcsph import *
from eng.wcsesph import *
from eng.muIsesph import *
from eng.muIlfsph import *
from eng.dpsesph import *
from eng.dplfsph import *
from eng.dprksph import *

def choose_solver(case, material, cmodel, TDmethod, flag_kernel, para1=None, para2=None, para3=None, para4=None, para5=None, para6=None, para7=None, para8=None, para9=None):
    if material == 1 and cmodel == 1:
        rho = para1
        viscosity = para2
        stiffness = para3
        powcomp = para4
        if TDmethod == 1:
            solver = WCSESPHSolver(case, flag_kernel, rho, viscosity, stiffness, powcomp)
        elif TDmethod == 2:
            solver = WCLFSPHSolver(case, flag_kernel, rho, viscosity, stiffness, powcomp)
        elif TDmethod == 4:
            solver = WCRKSPHSolver(case, flag_kernel, rho, viscosity, stiffness, powcomp)
    elif material == 2 and cmodel == 1:
        rho = para1
        coh = para2
        fric = para3
        if TDmethod == 1:
            solver = MCmuISESPHSolver(case, flag_kernel, rho, coh, fric)
        elif TDmethod == 2:
            solver = MCmuILFSPHSolver(case, flag_kernel, rho, coh, fric)
        elif TDmethod == 4:
            raise Exception('Not implemented yet~')
    elif material == 2 and cmodel == 2:
        rho = para1
        coh = para2
        fric = para3
        E = para4
        av_alpha_Pi = para5
        av_beta_Pi = para6
        if TDmethod == 1:
            solver = DPSESPHSolver(case, kernel=flag_kernel, density=rho, cohesion=coh, friction=fric, EYoungMod=E)
        elif TDmethod == 2:
            solver = DPLFSPHSolver(case, kernel=flag_kernel, density=rho, cohesion=coh, friction=fric, EYoungMod=E, alpha_Pi=av_alpha_Pi, beta_Pi=av_beta_Pi)
        elif TDmethod == 4:
            solver = DPRKSPHSolver(case, kernel=flag_kernel, density=rho, cohesion=coh, friction=fric, EYoungMod=E, alpha_Pi=av_alpha_Pi, beta_Pi=av_beta_Pi)
    return solver
