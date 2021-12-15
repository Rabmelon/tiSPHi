import taichi as ti
ti.init(arch=ti.cpu)

dim = 2

center_cell = ti.Vector.field(2, dtype=ti.i32, shape=2)
center_cell[0] = ti.Vector([0, 1])
center_cell[1] = ti.Vector([2, 2])

@ti.func
def is_valid_cell(cell):
    # Check whether the cell is in the grid
    flag = True
    for d in ti.static(range(dim)):
        flag = flag and (0 <= cell[d] < 3)
    return flag

@ti.kernel
def func_test():
    for i in range(2):
        print(center_cell[i], ';')
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * dim)):
            cell = center_cell[i] + offset
            print(cell, end=', ')
            if not is_valid_cell(cell):
                print('kill', end='! ')
                continue
        print()

func_test()
