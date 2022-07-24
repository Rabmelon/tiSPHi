import taichi as ti
import numpy as np

# TODO: make a general function to calculate dB/dA

ti.init(debug=True)

epsilon = 1e-16
dim = 2
I = ti.Matrix(np.eye(dim))
# I = ti.Matrix([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
alpha_fric = 0.13685771015527778
k_c = 4.402384598296268

smoothing_len = 0.0024

@ti.func
def cal_d_BA(x_i, x_j, bound):
    print("xi =", x_i, "xj =", x_j)
    boundary = ti.Vector([
        bound[1], 0.0,
        bound[0], 0.0])
    db_i = ti.Vector([x_i[1] - boundary[0], x_i[1] - boundary[1], x_i[0] - boundary[2], x_i[0] - boundary[3]])
    db_j = ti.Vector([x_j[1] - boundary[0], x_j[1] - boundary[1], x_j[0] - boundary[2], x_j[0] - boundary[3]])

    flag_b = db_i * db_j
    flag_dir = flag_b < 0
    print("flag_dir init =", flag_dir)
    flag_choosei = abs(flag_dir * db_i)
    flag_choosej = abs(flag_dir * db_j)
    flag_choose = ti.max(flag_choosei, flag_choosej)
    print("flag_choosei init =", flag_choosei)
    print("flag_choosej init =", flag_choosej)
    print("flag_choose init =", flag_choose)

    if flag_dir.sum() > 1:
        tmp_max = 0.0
        for i in ti.static(range(4)):
            tmp_max = max(tmp_max, flag_choose[i])
        flag_choose -= tmp_max
        print("flag_choose cal1 =", flag_choose)
        flag_choose = flag_choose == 0.0
        print("flag_choose cal2 =", flag_choose)
        flag_dir -= int(flag_choose)
        print("flag_dir final1 =", flag_dir)
        if flag_dir.sum() < 1:
            print("enter the judge!")
            for i in ti.static(range(4)):
                print("i =", i)
                if flag_choose[i] > 0.0:
                    flag_dir[i] = 1
                    break
            print("flag_dir final2 =", flag_dir)

    d_A = abs(db_i.dot(flag_dir))
    d_B = abs(db_j.dot(flag_dir))
    return d_B / d_A


@ti.kernel
def foo():
    res = cal_d_BA(x_i, x_j, bound)
    print("d_BA =", res)

if __name__ == "__main__":
    print("hallo test here!")

    x_i = [0.004, 0.002]
    x_j = [-0.002, -0.004]
    bound = [0.55, 0.2]

    foo()

    pos = ti.Vector.field(2, float, shape=1)
    pos_np = np.array([[0.5, 0.5]])
    pos.from_numpy(pos_np)

    res = (300, 200)
    window = ti.ui.Window('test', res=res)
    canvas = window.get_canvas()
    canvas.set_background_color((1,1,1))
    while window.running:
        canvas.circles(pos, radius=0.2, color=(1.0, 0.0, 0.0))
        window.show()

    flag_end = 1
