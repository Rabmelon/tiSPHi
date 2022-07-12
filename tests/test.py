import taichi as ti
import numpy as np

ti.init()

def gen_grid_line_2d(world, grid_line):
    dim = len(world)
    if not isinstance(grid_line,list):
        grid_line = [grid_line for _ in range(dim)]
    num_grid_point = [int((world[i] - 1e-8) // grid_line[i]) for i in range(dim)]
    num_all_grid_point = sum(num_grid_point)
    num_all2_grid_point = 2 * num_all_grid_point
    np_pos_line = np.array([[0.0 for _ in range(dim)] for _ in range(num_all2_grid_point)])
    np_indices_line = np.array([[i, i + num_all_grid_point] for i in range(num_all_grid_point)])
    for id in range(dim):
        id2 = dim - 1 - id
        for i in range(num_grid_point[id]):
            np_pos_line[i + sum(num_grid_point[0:id])][id] = (i + 1) * grid_line[id]
            np_pos_line[i + sum(num_grid_point[0:id]) + num_all_grid_point][id] = (i + 1) * grid_line[id]
            np_pos_line[i + sum(num_grid_point[0:id]) + num_all_grid_point][id2] = world[id2]
            print(id, i, np_pos_line[i + sum(num_grid_point[0:id])], np_pos_line[i + sum(num_grid_point[0:id]) + num_all_grid_point])


if __name__ == "__main__":
    print("hallo test kernel function accuracy!")

    gen_grid_line_2d([120, 80], [15, 16])
    flag_end = 1
