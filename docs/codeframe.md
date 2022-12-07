# Code frame

## Architecture

<div align="center">
  <img width="800px" src="https://github.com/Rabmelon/tiSPHi/raw/master/docs/img/code_architecture.png">
</div>


## Aim

### Ensure the correctness
1. Generate particles of main domain
2. Initiate particle paras
3. Fix neighbour search method
4. Generate dummy and repulsive particles
5. Make the approximation right
6. Make the kernel functions right
7. Fix the boundary treatment
8. Test for $f=x+y$ etc

### Improve the performance
1. Allocate memory well - more particles and faster speed
2. Promote the calculation function - faster
3. Use correct taichi functions and syntax - more particles and faster

### Balance the convenience
1. Set colorbar of selectable paras while running.
2. Extract the repeated code and functions
3. Promote the convenience of inputing paras