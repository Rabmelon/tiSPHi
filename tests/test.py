import taichi as ti
import numpy as np

ti.init(debug=True)

@ti.func
def funtest(pos, t):
    res = ti.Matrix([[1, 2*ti.sin(t), 0.5], [-0.333, 1, -ti.sin(t)], [2*pos[0]*ti.sin(2*t), 0, 1.5]])
    return res


@ti.kernel
def foo():
    pos = ti.Vector([1, 1, 1])
    t = 0.25
    F3 = funtest(pos, t)
    FTF = F3.transpose()@F3
    Feig = ti.sym_eig(FTF)
    FeigV = Feig[0]
    FeigM = Feig[1:]
    FeigMti = ti.Matrix(np.array(FeigM[0]))
    Udash2 = ti.Matrix([[FeigV[0], 0, 0], [0, FeigV[1], 0], [0, 0, FeigV[2]]])
    Udash = Udash2 ** 0.5
    Ucal = FeigMti.transpose()@Udash@FeigMti
    Uinv = Ucal.inverse()
    Rcal = F3@Uinv

    print("F\n",F3)
    print("FT@F\n",FTF)
    print("FeigV\n",FeigV)
    print("FeigM\n",FeigMti)
    print("U'2\n",Udash2)
    print("U'\n",Udash)
    print("U\n",Ucal)
    print("U-1\n",Uinv)
    print("R\n",Rcal)

    # FFT2 = F2@F2.transpose()
    # print("F@FT\n",FFT2)
    # Feig2sym = ti.sym_eig(FFT2)
    # Feig2 = ti.eig(FFT2)
    # print("Feig2sym\n",Feig2sym)
    # print("Feig2\n",Feig2)


if __name__ == "__main__":
    print("hallo test kernel function accuracy!")

    a = ti.Matrix([[1, 2], [3, 4]])
    b = ti.Vector([1, 2])
    a1 = ti.Matrix([[1, 3], [5, 4]])

    D = ti.Matrix([[1,2,0,2],[2,1,0,2],[0,0,3,0],[2,2,0,1]])

    c = a@b
    d = a@a1
    d1 = a1@a

    R = ti.Matrix([[0.866, -0.5], [0.5, 0.866]])
    V = ti.Matrix([[1.313, 0.325], [0.325, 0.938]])
    U = ti.Matrix([[1.5, 0], [0, 0.75]])
    F2 = ti.Matrix([[1.3, -0.375], [0.75, 0.65]])

    V1 = R@U@R.transpose()
    U1 = R.transpose()@V@R
    FU = R@U
    FV = V@R
    FUT1 = (R@U).transpose()
    FUT2 = U.transpose()@R.transpose()
    RTR = R.transpose()@R
    RRT = R@R.transpose()

    print("R[0,1]",R[0,1])
    print("V1\n",V1)
    print("U1\n",U1)
    print("FU\n",FU)
    print("FV\n",FV)
    print("FUT1\n",FUT1)
    print("FUT2\n",FUT2)
    print("RT@R\n",RTR)
    print("R@RT\n",RRT)

    foo()

    flag_end = 1
