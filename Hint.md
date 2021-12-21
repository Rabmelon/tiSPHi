
21.12.21.
1. Sometimes the grid(1, 1) will eat particles! (before add boundary particles)
![grid eat particles](temp/Snipaste_2021-12-21_18-00-30.png)
2. But after adding boundary particles, particles will be eaten in the grid(:, 1)!!!!!!!!! Amazing!!!
![bottom eat particles!](temp/211221_bug1.gif)


21.12.20.
1. The sum of m_V of all particles DONOT equal to the area of cube!!!

21.12.17.

1. When the **particle radius > 0.05**, internal error occured:

Exception has occurred: RuntimeError
[llvm_program.cpp:taichi::lang::LlvmProgramImpl::check_runtime_error@436] Assertion failure: (kernel=allocate_particles_to_grid_c20_0) Accessing field (S16place<i32>) of size (256, 256) with indices (-2147483648, -2147483648)
  File "D:\ZLei\Documents\Git\taichiCourse01_tiSPHi\eng\particle_system.py", line 248, in initialize_particle_system
    self.allocate_particles_to_grid()
  File "D:\ZLei\Documents\Git\taichiCourse01_tiSPHi\eng\sph_solver.py", line 122, in step
    self.ps.initialize_particle_system()
  File "D:\ZLei\Documents\Git\taichiCourse01_tiSPHi\test1.py", line 29, in <module>
    wcsph_solver.step()

2. In CPU debug mode, if there are no breakpoints before wcsph solver para, a sentence will occur in the terminal:
IMAGE_REL_AMD64_ADDR32NB relocation requires anordered section layout.
