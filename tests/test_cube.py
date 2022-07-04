import numpy as np
from functools import reduce    # 整数：累加；字符串、列表、元组：拼接。lambda为使用匿名函数

def add_cube(lower_corner, cube_size, offset):
    # lower_corner: corrdinate of left-down corner, exp. [1.0, 2.0, 3.0] or [1.0, 2.0]
    # cube_size: length of each edge, exp. [10, 20, 30] or [10.0, 20.0]
    # offset: offset distance, usually the diameter of particle, exp. 2.0
    dim = len(lower_corner)
    num_dim = []
    range_offset = offset
    for i in range(dim):
        num_dim.append(np.arange(lower_corner[i] + 0.5 * offset, lower_corner[i] + cube_size[i] + 1e-5, range_offset))
    num_new_particles = reduce(lambda x, y: x * y, [len(n) for n in num_dim])

    new_positions = np.array(np.meshgrid(*num_dim, sparse=False, indexing='ij'), dtype=np.float32)
    new_positions = new_positions.reshape(-1, reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()

    return num_new_particles, new_positions

if __name__ == "__main__":
    print("hallo test kernel function accuracy!")

    num_new_particles2, new_positions2 = add_cube([0.0, 0.0], [16.0, 24.0], 8.0)
    num_new_particles3, new_positions3 = add_cube([0.0, 0.0, 0.0], [16.0, 24.0, 32.0], 8.0)

    flag_end = 1
