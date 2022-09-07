import taichi as ti

ti.init(arch=ti.cpu, debug=True, default_fp=ti.f64, kernel_profiler=True)

N = 2**24   #16777216
# x = ti.field(float, shape=N)
# y = ti.field(float, shape=N)
# z = ti.field(float, shape=N)
x = ti.field(float)
y = ti.field(float)
z = ti.field(float)
node = ti.root.dense(ti.i, N)
# node.place(x, y, z)
node.place(x)
node.place(y)
node.place(z)

@ti.kernel
def fill1():
    for i in range(N):
        x[i] = i
        y[i] = x[i] * 1.2
        z[i] = i * y[i]

@ti.kernel
def fill2():
    for i in range(N):
        x[i] = i
    for i in range(N):
        y[i] = x[i] * 1.2
    for i in range(N):
        z[i] = i * y[i]

print("fill 1")
for i in range(100):
    fill1()
ti.profiler.print_kernel_profiler_info()  # default mode: 'count'
ti.profiler.clear_kernel_profiler_info()  # clear all records

print("fill 2")
for i in range(100):
    fill2()
ti.profiler.print_kernel_profiler_info()  # default mode: 'count'

# @ti.kernel
# def foo():
#     A = An.inverse()
#     print(A)

# def Ap(pi):
#     Ai = pi @ pi.transpose()
#     print("Ai,", Ai)
#     return Ai

# if __name__ == "__main__":
#     print("hallo test here!")

#     p1 = ti.Vector([1.0, 8.0, 0.0])
#     p2 = ti.Vector([1.0, 4.0, 4.0])
#     p3 = ti.Vector([1.0, 0.0, 4.0])
#     p4 = ti.Vector([1.0, 8.0, 8.0])

#     A1 = Ap(p1)
#     A2 = Ap(p2)
#     A3 = Ap(p3)
#     A4 = Ap(p4)
#     An = A1 + A2 + A3 + A4

#     print("An,", An)
#     foo()

#     A1A2 = A1 * A2
#     print(A1A2)

#     flag_end = 1