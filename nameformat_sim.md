# e.g.
20220630_dambreak_WC_SE_WL_norm_p_note.mp4

# item
date(ymd)_
case name_
model_
time integration_
kernel_
normalisation or not_
colored property_
comment

# Detail
## Choices:
case name:
db-dambreak, cc-column collapse, st-shear test, is-initial stress

db0: default dambreak case, 0.2\*0.4 in 0.8\*0.8
db01: default dambreak case, 0.4\*0.2 in 0.8\*0.8

db1: water, experiment in xu2021partI, 0.146\*0.292 in 0.584\*0.2 (0.8)

cc0: sand, experiment in Nguyen2017, 0.2\*0.1 in 0.5\*0.2 (0.5)

is0: default initial stress case, 0.2\*0.1 in 0.2\*0.2

# model:
water: WC
soil: muI, DP

# time integration:
SE Symplectic Euler, LF Leap-Frog, RK4 4th order Runge-Kutta

# kernel:
CS cubic spline, WL wendland C2

# normalisation or not:
norm for yes; no for no

# colored property:
p pressure
r density
u norm of velocity, also ux, uy
x position, also y
dr d_density
sm hydrostatic pressure; s norm of stress, also sxx, syy, sxy
ep norm of strain, also epxx, epyy, epxy;





