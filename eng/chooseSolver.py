from eng.wcsph import *
from eng.wcsesph import *
from eng.muIsesph import *
from eng.muIlfsph import *
from eng.dpsesph import *
from eng.dplfsph import *

def chooseSolver(case, material, cmodel, TDmethod, flag_kernel, para1=0.0, para2=0.0, para3=0.0, para4=0.0, para5=0.0, para6=0.0, para7=0.0, para8=0.0, para9=0.0, para10=0.0):
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
        if TDmethod == 1:
            solver = DPSESPHSolver(case, flag_kernel, rho, coh, fric, E)
        elif TDmethod == 2:
            solver = DPLFSPHSolver(case, flag_kernel, rho, coh, fric, E)
        elif TDmethod == 4:
            raise Exception('Not implemented yet~')
    return solver
