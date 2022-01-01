from numpy.core.numeric import Inf
import taichi as ti

ti.init(ti.cpu)

#TODO: Calculate dA and dB of boundary treatment.

@ti.func
def cal_d_AB(x_i, x_j, bound, padding):
    print('----cal starts!')
    # dx_ij = x_j - x_i
    # print('dx_ij =', dx_ij)
    # direction: up, down, right, left
    # dir = ti.Vector([[0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0]])
    # dir_num = ti.Vector([0, 1, 2, 3])
    boundary = ti.Vector([bound[1] - padding, padding, bound[0] - padding, padding])

    db_i = ti.Vector([x_i[1] - boundary[0], x_i[1] - boundary[1], x_i[0] - boundary[2], x_i[0] - boundary[3]])
    db_j = ti.Vector([x_j[1] - boundary[0], x_j[1] - boundary[1], x_j[0] - boundary[2], x_j[0] - boundary[3]])
    print('db_i =', db_i)
    print('db_j =', db_j)

    flag_b = db_i * db_j
    print('flag boundary =', flag_b)

    flag_dir = flag_b < 0
    print('flag direction =', flag_dir)

    if sum(flag_dir) > 1:
        flag_choose = abs(flag_dir * db_i)
        print('choose 1:', flag_choose)
        tmp_max = -Inf
        for i in ti.static(range(4)):
            tmp_max = max(tmp_max, flag_choose[i])
        print('max choose:', tmp_max)
        flag_choose -= tmp_max
        print('choose 2:', flag_choose)
        flag_choose = flag_choose == 0.0
        print('choose 2:', flag_choose)
        flag_dir -= flag_choose
        print('choose 3:', flag_dir)

    d_A = abs(db_i.dot(flag_dir))
    d_B = abs(db_j.dot(flag_dir))
    print('Boundary direction: ', d_A, d_B)
    return d_B / d_A

    # flag_dir = dir @ dx_ij
    # print('flag direction =', flag_dir)


@ti.kernel
def cal_d_AB_test():
    bound = ti.Vector([6, 6])
    padding = 1

    x_i = ti.Vector([1.25, 2.0])

    x_j1 = ti.Vector([0.5, 2.75])
    x_j2 = ti.Vector([0.5, 0.5])
    x_j3 = ti.Vector([1.25, 0.5])
    x_j4 = ti.Vector([5.5, 0.5])
    x_j5 = ti.Vector([5.5, 5.5])

    d_BA1 = cal_d_AB(x_i, x_j1, bound, padding)
    d_BA2 = cal_d_AB(x_i, x_j2, bound, padding)
    d_BA3 = cal_d_AB(x_i, x_j3, bound, padding)
    d_BA4 = cal_d_AB(x_i, x_j4, bound, padding)
    d_BA5 = cal_d_AB(x_i, x_j5, bound, padding)



cal_d_AB_test()
