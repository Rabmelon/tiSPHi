from math import sqrt, tan, pi
import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import copy
plt.switch_backend('TKAgg')

ti.init(arch=ti.cpu)

# TODO: Test the D-P stress check and update

# basic property
par = pi/180
epsilon = 1e-4

# input, list
def initMove(dt, u, x, a):
    dt = dt
    u = u
    x = x
    a = a
    u1 = [iu + ia * dt for iu, ia in zip(u, a)]
    x1 = [ix + iu1 * dt for ix, iu1 in zip(x, u1)]
    du = [iu1 - iu for iu1, iu in zip(u1, u)]
    dx = [ix1 - ix for ix1, ix in zip(x1, x)]
    grad_u = [[idu/jdx for jdx in dx] for idu in du]
    return grad_u

def initProp(coh, fric, youngModulus, nu):
    coh = coh
    fric = fric # rad
    youngModulus = youngModulus
    nu = nu
    alpha_fric = tan(fric)/sqrt(9+12*tan(fric)**2)
    k_c = 3*coh/sqrt(9+12*tan(fric)**2)
    shearModulus = youngModulus/(2*(1+nu))
    bulkModulus = youngModulus/(3*(1-2*nu))
    return alpha_fric, k_c, shearModulus, bulkModulus

def initStress(stress):
    dim = len(stress)
    stress = stress
    stress_m = sum([stress[i][i] for i in range(dim)]) / 3
    stress_kk = [[Kdelta(i, j)*stress_m for i in range(dim)] for j in range(dim)]
    stress_dev = [[stress[i][j]-stress_kk[i][j] for i in range(dim)] for j in range(dim)]
    return stress_m, stress_dev

def calInvariants(stress_m, stress_dev):
    negI1 = -3*stress_m
    sqrtJ2 = sqrt(0.5*dbdot(stress_dev, stress_dev))
    sqrtJ2 = sqrtJ2 if sqrtJ2 > 0 else epsilon
    return negI1, sqrtJ2

def initStrain(grad_u):
    dim = len(grad_u)
    strain = [[0.5*(grad_u[i][j]+grad_u[j][i]) for i in range(dim)] for j in range(dim)]
    strain_m = sum([strain[i][i] for i in range(dim)]) / 3
    strain_kk = [[Kdelta(i, j)*strain_m for i in range(dim)] for j in range(dim)]
    strain_dev = [[strain[i][j]-strain_kk[i][j] for i in range(dim)] for j in range(dim)]
    return strain, strain_m, strain_dev

# Kronecker delta
def Kdelta(i, j):
    r = 1 if i == j else 0
    return r

# double dot product
def dbdot(a, b):
    dim = len(a)
    tmp = [[a[i][j]*b[i][j] for i in range(dim)] for j in range(dim)]
    return sum([sum(tmp[i]) for i in range(dim)])

# calculate strain item g
def checkg(fDP, dfDP):
    if abs(fDP) < epsilon:
        if abs(dfDP) < epsilon:
            flag_g = 1
        elif dfDP > epsilon:
            flag_g = 0
        else:
            flag_g = -1
            print('Wrong with dfDP!')
    elif fDP < -epsilon:
        flag_g = 0
    else:
        flag_g = -1
        print('Wrong with fDP!')
    return flag_g

def calg(flag_g, stress, strain, strain_m, a, shear):
    dim = len(stress)
    if flag_g == 1:
        stress_m, stress_dev = initStress(stress)
        negI1, sqrtJ2 = calInvariants(stress_m, stress_dev)
        tmp1 = dbdot(stress_dev, strain)
        vlambda = (3*a*strain_m+(shear/sqrtJ2)*tmp1)/shear
        tmp2 = vlambda*shear/sqrtJ2
        g = [[tmp2/stress_dev[i][j] if stress_dev[i][j] != 0 else 0.0 for i in range(dim)] for j in range(dim)]
    elif flag_g == 0:
        g = [[0.0 for i in range(dim)] for j in range(dim)]
    else:
        g = None
        print('Wrong with g!')
    return g

# check one
def calfDP(stress, a, k):
    stress_m, stress_dev = initStress(stress)
    negI1, sqrtJ2 = calInvariants(stress_m, stress_dev)
    return sqrtJ2-a*negI1-k

def chkAdapt1(negI1, a, k):
    return True if negI1*a+k<0 else False

def chkAdapt2(negI1, sqrtJ2, a, k):
    return True if negI1*a+k<sqrtJ2 else False

# adapt one
def adapt1(stress, negI1, a, k):
    dim = len(stress)
    r = copy.deepcopy(stress)
    tmp = (-negI1-k/a)/3
    for i in range(dim):
        r[i][i] -= tmp
    return r

def adapt2(stress_dev, negI1, sqrtJ2, a, k):
    dim = len(stress_dev)
    r = copy.deepcopy(stress_dev)
    tmp = (negI1*a+k)/sqrtJ2
    for i in range(dim):
        for j in range(dim):
            r[i][j] = tmp*stress_dev[i][j]-negI1/3 if i == j else tmp*stress_dev[i][j]
    return r

def stressAdapt(stress, a, k):
    # check and update 1
    stress_m, stress_dev = initStress(stress)
    negI1, sqrtJ2 = calInvariants(stress_m, stress_dev)
    r = copy.deepcopy(stress)
    if chkAdapt1(negI1, a, k):
        r = adapt1(r, negI1, a, k)
    # check and update 2
    stress_m, stress_dev = initStress(r)
    negI1, sqrtJ2 = calInvariants(stress_m, stress_dev)
    if chkAdapt2(negI1, sqrtJ2, a, k):
        r = adapt2(stress_dev, negI1, sqrtJ2, a, k)
    return r

# draw
def showDP(stress, stress_new, a, k):
    stress_m, stress_dev = initStress(stress)
    negI1, sqrtJ2 = calInvariants(stress_m, stress_dev)
    stress_m_new, stress_dev_new = initStress(stress_new)
    negI1_new, sqrtJ2_new = calInvariants(stress_m_new, stress_dev_new)
    x_l = -k/a
    x_r = max(-x_l, negI1, negI1_new)
    x = np.linspace(x_l,x_r,100)
    y = a*x+k
    plt.plot(x, y, color='blue')
    plt.plot([x_l, x_l], [0, max(y)], color='black', linestyle='--')
    plt.scatter(negI1, sqrtJ2, color='red', linewidth=2)
    plt.scatter(negI1_new, sqrtJ2_new, color='green', linewidth=2)
    # plt.arrow(negI1, sqrtJ2, negI1_new-negI1, sqrtJ2_new-sqrtJ2+0.5, head_width=0.5)
    plt.grid()
    # plt.xlim(-42.0,35.0)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    plt.show()

# main
if __name__ == "__main__":
    print('Hallo!')

    fDP_old = 0
    dt = 0.01
    u = [3, 2, 1]
    x = [5, 5, 5]
    a = [0.01, 0.02, -0.015]
    coh = 5
    fric = 25*par
    youngMod = 1.80e6
    nu = 0.2
    # stress_ini = [6, 2, 3, -1, 0, 0]  # xx, yy, zz, xy, xz, yz # no adapt elas
    # stress_ini = [6.2992, 4, 5, -2, 0, 0]  # xx, yy, zz, xy, xz, yz # no adapt plas
    # stress_ini = [16, 12, 10, -4, 0, 0]  # xx, yy, zz, xy, xz, yz # adapt 1
    stress_ini = [6, 2, 10, -4, 0, 0]  # xx, yy, zz, xy, xz, yz # adapt 2
    stress0 = [[stress_ini[0], stress_ini[3], stress_ini[4]], [stress_ini[3], stress_ini[1], stress_ini[5]], [stress_ini[4], stress_ini[5], stress_ini[2]]]

    grad_u = initMove(dt, u, x, a)
    alpha_fric, k_c, shearMod, bulkMod = initProp(coh, fric, youngMod, nu)
    strain, strain_m, strain_dev = initStrain(grad_u)

    stress_new = stress0
    fDP_new = calfDP(stress_new, alpha_fric, k_c)
    while fDP_new > epsilon:
        stress_new = stressAdapt(stress_new, alpha_fric, k_c)
        fDP_new = calfDP(stress_new, alpha_fric, k_c)

    dfDP = fDP_new - fDP_old
    flag_g = checkg(fDP_new, dfDP)
    item_g = calg(flag_g, stress_new, strain, strain_m, alpha_fric, shearMod)

    print('g =', item_g)
    showDP(stress0, stress_new, alpha_fric, k_c)

