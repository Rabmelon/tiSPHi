---
html:
  toc: true
---
**Smoothed Particle Hydrodynamics Learning Notes**

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=5 orderedList=false} -->

<!-- code_chunk_output -->

- [Foundation of SPH](#foundation-of-sph)
  - [Basic mathematics](#basic-mathematics)
    - [The spatial derivative operators in 3D](#the-spatial-derivative-operators-in-3d)
    - [Material derivative](#material-derivative)
  - [SPH basic formulations](#sph-basic-formulations)
    - [Kernel estimation](#kernel-estimation)
    - [Spatial derivatives](#spatial-derivatives)
    - [Improving approximations for spatial derivatives](#improving-approximations-for-spatial-derivatives)
    - [Particle approximations](#particle-approximations)
  - [Boundary treatment](#boundary-treatment)
    - [Basic methods](#basic-methods)
    - [Complex boundary treatment](#complex-boundary-treatment)
      - [For straight, stationary walls](#for-straight-stationary-walls)
      - [For free surface problems](#for-free-surface-problems)
  - [Time integration](#time-integration)
    - [Symp Euler - Symplectic Euler](#symp-euler---symplectic-euler)
    - [RK4 - 4th order Runge-Kutta](#rk4---4th-order-runge-kutta)
    - [XSPH](#xsph)
  - [Tensile instability](#tensile-instability)
- [SPH for water](#sph-for-water)
  - [Navier-Stokes equation](#navier-stokes-equation)
    - [Forces for incompressible fluids](#forces-for-incompressible-fluids)
    - [N-S Equations](#n-s-equations)
    - [Temporal discretization](#temporal-discretization)
  - [Full time integration](#full-time-integration)
  - [The weakly compressible assumption](#the-weakly-compressible-assumption)
    - [Fluid dynamics with particles](#fluid-dynamics-with-particles)
    - [RK4 for WCSPH](#rk4-for-wcsph)
- [SPH for soil](#sph-for-soil)
  - [Constitutive model of soil](#constitutive-model-of-soil)
  - [Governing equations](#governing-equations)
    - [Conservation of mass](#conservation-of-mass)
    - [Conservation of momentum](#conservation-of-momentum)
  - [Standard soil SPH](#standard-soil-sph)
    - [Discretization](#discretization)
    - [RK4 for standard soil SPH](#rk4-for-standard-soil-sph)
    - [Steps](#steps)
  - [Stress-Particle SPH](#stress-particle-sph)
- [FEM-SPH](#fem-sph)

<!-- /code_chunk_output -->

For learning how SPH works basically in slope failure and post-failure process, also in the landslide and furthermore debris-flows through obstacles. [@buiLagrangianMeshfreeParticles2008; @chalkNumericalModellingLandslide2019; @chalkStressParticleSmoothedParticle2020; @taichiCourse01; @buiSmoothedParticleHydrodynamics2021]

---
# Foundation of SPH

## Basic mathematics

### The spatial derivative operators in 3D
$\nabla$ 算子的三个语义:
$$\nabla=\boldsymbol{i}\frac{\partial}{\partial x}+\boldsymbol{j}\frac{\partial}{\partial y}+\boldsymbol{k}\frac{\partial}{\partial z}$$

**梯度Gradient**：作用于**标量**$f(x, y, z)$得到**矢量**。$\mathbb{R}^1\rightarrow\mathbb{R}^3, \nabla$
$$grad\ f=\nabla f=(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z})$$

**散度Divergence**：作用于**矢量**$(f_x, f_y, f_z)$得到**标量**。$\mathbb{R}^3\rightarrow\mathbb{R}^1, \nabla\cdot$
$$div\ \boldsymbol{f}=\nabla\cdot \boldsymbol{f}=\frac{\partial f_x}{\partial x} + \frac{\partial f_y}{\partial y} + \frac{\partial f_z}{\partial z}$$

**旋度Curl**：作用于**矢量**$(f_x, f_y, f_z)$得到**矢量**。$\mathbb{R}^3\rightarrow\mathbb{R}^3, \nabla\times$
$$curl\ \boldsymbol{f}=\nabla\times\boldsymbol{f}=\begin{vmatrix} \boldsymbol{i} &\boldsymbol{j} &\boldsymbol{k}\\ \frac{\partial}{\partial x} &\frac{\partial}{\partial y} &\frac{\partial}{\partial z}\\ f_x &f_y &f_z \end{vmatrix}=(\frac{\partial f_z}{\partial y}-\frac{\partial f_y}{\partial z}, \frac{\partial f_x}{\partial z}-\frac{\partial f_z}{\partial x}, \frac{\partial f_y}{\partial x}-\frac{\partial f_x}{\partial y})$$

**拉普拉斯Laplace**: 梯度的散度，作用于任意维度的变量。 $\mathbb{R}^n\rightarrow\mathbb{R}^n, \nabla \cdot \nabla=\nabla^2$
$$laplace\ f=div(grad\ f)=\nabla^2f=\frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} + \frac{\partial^2 f}{\partial z^2}$$

### Material derivative
$$\frac{{\rm D}f}{{\rm D}t}=\frac{\partial f}{\partial t}+\boldsymbol{u}\cdot\nabla f$$

is **material derivative** in fluid mechanics, total derivative in math. 数学上的全导数，流体力学中的物质导数、随体导数，为流体质点在运动时所具有的物理量对时间的全导数。
[Wiki](https://en.wikipedia.org/wiki/Material_derivative): In continuum mechanics, the material derivative describes the time rate of change of some physical quantity (like heat or momentum) of a material element that is subjected to a space-and-time-dependent macroscopic velocity field. The material derivative can serve as a link between Eulerian and Lagrangian descriptions of continuum deformation.
运动的流体微团的物理量随时间的变化率，它等于该物理量由当地时间变化所引起的变化率与由流体对流引起的变化率的和。
从偏导数全微分的摡念出发，密度变化可以认为是密度分布函数（密度场）的时间偏导数项（不定常）和空间偏导数项（空间不均匀）的和。时间偏导项叫局部导数或就地导数。空间偏导项叫位变导数或对流导数。
中科院的李新亮研究员给出了一个更加形象的例子：高铁的电子显示屏上会实时显示车外的温度，如果我们将高铁看作是一个流体微元，它早上从北京出发，中午到达上海，显示屏上记录的室外温度的变化就是物质导数，它包含了两个部分，一是从北京到上海的地理位置的变化所带来的温度变化，即对流导数；二是由于早上到中午由于时间不同而引起的温度变化，即当地导数。)
The final form in Lagrangian method of density: (等号左侧，第一项为微团密度的变化，第二项为微团体积的变化。)
$$\frac{{\rm D}\rho}{{\rm D}t}+\rho\nabla\cdot\boldsymbol{u}=0$$

对于不可压缩流动，质点的密度在运动过程中保持不变，故$\frac{{\rm D}\rho}{{\rm D}t}=0$

> **QUESTIONS**
> 1. Why does Bui use $\frac{{\rm d} \rho}{{\rm d} t}$ while Chalk uses $\frac{{\rm D} \rho}{{\rm D} t}$? Who is right? What's the difference? **GUESS** Maybe Chalk is right? There is no material derivative in Bui's formulations.

## SPH basic formulations

### Kernel estimation

### Spatial derivatives
> taichiCourse01-10 PPT p59 and 72

Approximate a function $f(r)$ using finite probes $f(r_j)$, and the degree of freedom $(r)$ goes inside the kernel functions (**anti-sym** and **sym**).
* SPH discretization:
$$f(r) \approx \sum_j \frac{m_j}{\rho_j}f(r_j)W(r-r_j, h) $$

* SPH spatial derivatives:
$${\color{Salmon} \nabla} f(r) \approx \sum_j \frac{m_j}{\rho_j}f(r_j){\color{Salmon} \nabla}W(r-r_j, h)  $$

$${\color{Salmon} \nabla\cdot} \boldsymbol{F}(r) \approx \sum_j \frac{m_j}{\rho_j}\boldsymbol{F}(r_j){\color{Salmon} \cdot\nabla}W(r-r_j, h)  $$

$${\color{Salmon} \nabla\times} \boldsymbol{F}(r) \approx -\sum_j \frac{m_j}{\rho_j}f(r_j){\color{Salmon} \times\nabla}W(r-r_j, h)  $$

$${\color{Salmon} \nabla^2} f(r) \approx \sum_j \frac{m_j}{\rho_j}f(r_j){\color{Salmon} \nabla^2}W(r-r_j, h)  $$

with $W(r_i-r_j, h) = W_{ij}$ in discrete view.

> **QUESTIONS**
> 1. How to calculate $\nabla W$ and $\nabla^2 W$?

### Improving approximations for spatial derivatives
> taichiCourse01-10 PPT p60-70

* Let $f(r) \equiv 1$, we have:
  * $1 \approx \sum_j \frac{m_j}{\rho_j}W(r-r_j, h)$
  * $0 \approx \sum_j \frac{m_j}{\rho_j}\nabla W(r-r_j, h)$
* Since $f(r)\equiv f(r) * 1$, we have:
  * $\nabla f(r) = \nabla f(r)*1+f(r)*\nabla 1$
  * Or equivalently: $\nabla f(r) = \nabla f(r)*1-f(r)*\nabla 1$
* Then use ${\color{Salmon} \nabla} f(r) \approx \sum_j \frac{m_j}{\rho_j}f(r_j){\color{Salmon} \nabla}W(r-r_j, h)$ to derivate $\nabla f(r)$ and $\nabla 1$, we have:
  * $\nabla f(r) \approx \sum_j \frac{m_j}{\rho_j}f(r_j)\nabla W(r-r_j, h) - f(r)\sum_j \frac{m_j}{\rho_j}\nabla W(r-r_j, h)$
  * $\nabla f(r) \approx \sum_j \frac{m_j}{\rho_j}(f(r_j)-f(r))\nabla W(r-r_j, h)$, we call it the **anti-symmetric form**
* A more general case:
  $$\nabla f(r) \approx \sum_j m_j(\frac{f(r_j)\rho_j^{n-1}}{\rho^n}-\frac{nf(r)}{\rho})\nabla W(r-r_j, h)$$

  * When $n=-1$: $\nabla f(r) \approx \rho\sum_j m_j(\frac{f(r_j)}{\rho_j^2}+\frac{f(r)}{\rho^2})\nabla W(r-r_j, h)$, we call it the **symmetric form**
* 通常会使用一些反对称(**anti-sym**)或对称型(**sym**)来进行一些SPH的空间求导(spatial derivative)，而不直接使用SPH的原型。但两者的选择是个经验性的问题，其中，当$f(r)$是一个力的时候，从动量守恒的角度去推导，使用**sym**更好；当做散度、需要投影的时候，使用**anti-sym**更好。

### Particle approximations


## Boundary treatment

### Basic methods
> taichiCourse01-10 PPT p43 and 79-85

* Mainly two styles: **free surface** and **solid boundary**
* Problems: There are not enough samples within the supporting radius.
* For free surface:
  * Problem: Density $\downarrow$, pressure $\downarrow$; and generate outward pressure.
  * Solution: Clamp the negative pressure (everywhere); assume $p = max(0,k(\rho-\rho_0))$
  * 会导致液面可能会向外膨胀一点
* For solid boundary:
  * Problem: Density $\downarrow$, pressure $\downarrow$; and fluid leakage (due to outbound velocity)
  * Solution: $p = max(0,k(\rho-\rho_0))$;
  * Solution for leakage:
    1. Reflect the outbound velocity when close to boundary. 还可以将垂直边界方向的速度乘上一个衰减值。这样处理大抵应该是不会导致粒子飞出去。
    2. Pad a layer of solid particles (or called ghost particles) underneath the boundaries with $\rho_{solid} = \rho_0$ and $v_{solid} = 0$. 总体来说比方法1稳定，但可能会导致边界附近粒子的数值黏滞。

> **QUESTIONS**
> 1. 多介质的流体混合时，多介质的界面？？？

### Complex boundary treatment
> @Chalk2020

虚拟的边界粒子，本身不具有具体的属性数值。在每一个Step中，在每一个粒子的计算中，先加入一个对Dummy particle对应属性的赋值。

#### For straight, stationary walls

First, choose the method to solve boundary problems. I want to update the behaviour of particles without just invert the operator but with some rules that are suitable for soil dynamics problems.

The dummy particle method is used to represent the wall boundary. For dummy and repulsive particles at the wall boundary, they are spaced apart by $\Delta x/2$. For other dummy particles, are $\Delta x$.
<div align="center">
  <img width="300px" src="./temp/Dummy_particles.png">
</div>

The repulsive particles are set to apply the no-slip effect and always guarantee that the particles do not penetrate the wall boundary. They can apply a soft repulsive force to the particles near the wall boundary, which is incorporated as a body force in the momentum equation. The definition of the repulsive force is introduced that prevents particle penetration without obviously disturbing the interior particels. The force $\hat{\boldsymbol{F}}_{ij}$ is applied to all particles that interact with the repulsive boundary particles, and is included in the SPH momentum equation:
$$\hat{\boldsymbol{F}}_{ij} = \sum_j 0.01c^2\chi\cdot\hat{f}(\gamma)\frac{\boldsymbol{x}_{ij}}{r^2}$$

where:
$$\chi = \left\{
\begin{array}{ll}
  1-\frac{r}{1.5\Delta x}, &0\leq r<1.5\Delta x \\0, &r\geq 1.5\Delta x
\end{array}
\right.$$

$$\gamma = \frac{r}{0.75h_{ij}}$$

$$\hat{f}(\gamma) = \left\{
  \begin{array}{ll}
    \frac{2}{3}, &0<\gamma\leq\frac{2}{3}\\
    2\gamma-1.5\gamma^2, &\frac{2}{3}<\gamma\leq 1\\
    0.5(2-\gamma)^2, &1<\gamma<2\\
    0, &\gamma\geq 2
  \end{array}
\right.$$

And this soft repulsive force has been applied to simulations of water flow and the propagation of a Bingham material.

> **HINT**: Not yet add the repulsive force

For an interior particle A (circle) that contains a dummy particle B (square and triangle) within its neighbourhood, the normal distances $d_A$ and $d_B$ to the wall are calculated. An artificial velocity $\boldsymbol{u}_B$ is then assigned to the dummy particle:
$$\boldsymbol{u}_B = -\frac{d_B}{d_A}\boldsymbol{u}_A$$

To account for extremely large values of the dummy particle velocity when an interior particle approaches the boundary (and $d_A$ approaches 0), a parameter $\beta$ is introduced:
$$\boldsymbol{u}_B = (1-\beta)\boldsymbol{u}_A+\beta\boldsymbol{u}_{wall}\ ,\ \beta = min(\beta_{max}, 1+\frac{d_B}{d_A})$$

$\beta_{max}$ have been found to be between $1.5\rightarrow2$, and here we use $\beta_{max}=1.5$.

And we have $\boldsymbol{\sigma}_B=\boldsymbol{\sigma}_A$. The simple definition ensures that there is a uniform stress distribution for the particles that are near the wall boundaries, and it contributes to smooth stress distributions (through the $\boldsymbol{f}^{\sigma}$ term) on the interior particles in the equation of momentum through the particle-dummy interaction.

> **QUESTIONS**
> 1. How about the mass of repulsive particles? **ANSWER**: maybe the mass of repulsive particel = 0!
> 2. How to add repulsive forces in boundary particles?

#### For free surface problems
The particles that comprise the free surface should satisfy a stress-free condition. When considering large deformations this first requires the detection of free surface particles, followed by a transformation of the stress tensor so that the normal and tangential components are 0.

> **QUESTIONS**
> 1. BUT how does the free surface condition implement?

## Time integration

### Symp Euler - Symplectic Euler
> taichiCourse01-10 PPT p77

$$u_i^* = u_i+\Delta t\frac{{\rm d}u_i}{{\rm d}t},\ \ x_i^* = x_i+\Delta tu_i^*$$

### RK4 - 4th order Runge-Kutta
> Chalk2020 Appendix B.

The RK4 scheme has fourth order accuracy and relatively simple implementation.
Consider a general ordinary differential equation for a variable $\phi$ with an initial condition $\phi^0$ at an initial time $t^0$:
$$\dot{\phi} = f(t, \phi),\ \phi(t^0) = \phi^0$$

where $f$ is a function of $\phi$ and time $t$. The RK4 method is employed to increment $\phi$ by a time step $\Delta t$ to obtain the solution at time $t = t+\Delta t$:
$$\phi^{t+\Delta t}=\phi^t+\frac{\Delta t}{6}(k_1+2k_2+2k_3+k_4)$$

$$k_1=f(\phi_1),\ k_2=f(\phi_2),\ k_3=f(\phi_3),\ k_4=f(\phi_4)$$

$$\phi_1=\phi^t,\ \phi_2=\phi^t+\frac{\Delta t}{2}k_1,\ \phi_3=\phi^t+\frac{\Delta t}{2}k_2,\ \phi_4=\phi^t+\Delta tk_3$$

### XSPH
In addition to the velocity and stress, the position vectors of each particle $\boldsymbol{x}_i$ are updated via the XSPH method at the end of each time step as:
$$\frac{{\rm d} \boldsymbol{x}_i}{{\rm d} t} = \boldsymbol{u}_i + \varepsilon_x\sum_j\frac{m_j}{\rho_j}(\boldsymbol{u}_j - \boldsymbol{u}_i)\nabla W_{ij}$$

Alternatively, the discretised XSPH equation is:
$$\boldsymbol{x}_i^{t+\Delta t} = \boldsymbol{x}_i^t + \Delta t\frac{{\rm d} \boldsymbol{x}_i}{{\rm d} t} = \boldsymbol{x}_i^t + \Delta t(\boldsymbol{u}_i^{t+\Delta t} + \varepsilon_x\sum_j\frac{m_j}{\rho_j}(\boldsymbol{u}_j - \boldsymbol{u}_i)\nabla W_{ij})$$

where $\varepsilon_x$ is a tuning para, $0\leq\varepsilon_x\leq1$.

While, in standard SPH, the simplest way is:
$$\frac{{\rm d} \boldsymbol{x}_i}{{\rm d} t} = \boldsymbol{u}_i$$

And for the particle position update:
$$\boldsymbol{x}_i^{t+\Delta t} = \boldsymbol{x}_i^t + {\Delta t}\boldsymbol{u}_i^{t+\frac{\Delta t}{2}}\ and\ \boldsymbol{u}_i^{t+\frac{\Delta t}{2}} = \frac{1}{2}(\boldsymbol{u}_i^{t+\Delta t}+\boldsymbol{u}_i^t)$$

or just Symplectic Euler:
$$\boldsymbol{x}_i^{t+\Delta t} = \boldsymbol{x}_i^t + {\Delta t}\boldsymbol{u}_i^{t+\Delta t}$$

## Tensile instability

# SPH for water

## Navier-Stokes equation
### Forces for incompressible fluids
> taichiCourse01-10 PPT p8-13

$$f = ma = {\color{Green} f_{ext}} + {\color{RoyalBlue} f_{pres}} + {\color{Orange} f_{visc}}$$

### N-S Equations
> taichiCourse01-10 PPT p16-28

The momentum equation
$$\rho\frac{{\rm D}u}{{\rm D}t}={\color{Green} \rho g} {\color{RoyalBlue} -\nabla p} + {\color{Orange} \mu\nabla^2u}$$

The mass conserving condition
$${\color{RoyalBlue} \nabla\cdot u=0} $$

$\rho\frac{{\rm D}u}{{\rm D}t}$: This is simply "mass" times "acceleration" divided by "volume".
${\color{Green} \rho g}$: External force term, gravitational force divided by "volume".
${\color{Orange} \mu\nabla^2u}$: Viscosity term, how fluids want to move together. 表示扩散有多快，液体尽可能地往相同的方向运动。$\mu$: some fluids are more viscous than others.
${\color{RoyalBlue} -\nabla p}$: Pressure term, fluids do not want to change volume. $p=k(\rho-\rho_0)$ but $\rho$ is unknown.
${\color{RoyalBlue} \nabla\cdot u=0 \Leftrightarrow \frac{{\rm D} \rho}{{\rm D} t} = \rho(\nabla\cdot u) = 0}$: Divergence free condition. Outbound flow equals to inbound flow. The mass conserving condition. 散度归零条件、不可压缩特性，也是质量守恒条件。

### Temporal discretization
> taichiCourse01-10 PPT p32

Integrate the incompressible N-S equation in steps (also reffered as "Operator splitting" or "Advection-Projection" in different contexts):
* Step 1: input $u^t$, output $u^{t+0.5\Delta t}$: $\rho\frac{{\rm D} u}{{\rm D} t}={\color{Green} \rho g} + {\color{Orange} \mu\nabla^2u}$
* Step 2: input $u^{t+0.5\Delta t}$, output $u^{t+\Delta t}$: $\rho\frac{{\rm D} u}{{\rm D} t}={\color{RoyalBlue} -\nabla p}\ and\ {\color{RoyalBlue} \nabla\cdot u=0}$ (构成了$\rho$和$u$的二元非线性方程组)

## Full time integration
> taichiCourse01-10 PPT p33

$$\frac{{\rm D}u}{{\rm D}t}={\color{Green} g} {\color{RoyalBlue} -\frac{1}{\rho}\nabla p} + {\color{Orange} \nu\nabla^2u},\ \nu=\frac{\mu}{\rho_0}$$

* Given $x^t$, $u^t$:
* Step 1: Advection / external and viscosity force integration
  * Solve: ${\color{Purple} {\rm d}u} = {\color{Green} g} + {\color{Orange} \nu\nabla^2u_n}$
  * Update: $u^{t+0.5\Delta t} = u^t+0.5 \Delta t{\color{Purple} {\rm d}u}$
* Step 2: Projection / pressure solver
  * Solve: ${\color{red} {\rm d}u} = {\color{RoyalBlue} -\frac{1}{\rho}\nabla(k(\rho-\rho_0))}$ and ${\color{RoyalBlue} \frac{{\rm D} \rho}{{\rm D} t} = \nabla\cdot(u_{n+0.5}+{\color{red} {\rm d}u})=0}$
  * Update: $u^{t+\Delta t} = u^{t+0.5\Delta t} + 0.5 \Delta t {\color{red} {\rm d}u}$
* Step 3: Update position
  * Update: $x^{t+\Delta t} = x^t+\Delta tu^{t+\Delta t}$
* Return $x^{t+\Delta t}$, $u^{t+\Delta t}$

> **QUESTIONS**
> 1. In step 1 and 2, maybe the $\Delta t$ should also multiple 0.5? **ANSWER**: I think yes!

## The weakly compressible assumption
> taichiCourse01-10 PPT p34-35

Storing the density $\rho$ as an individual variable that advect with the velocity field. Then the $p$ can be assumed as a variable related by time and the mass conserving equation is killed.

* Change in Step 2:
  * Solve: ${\color{red} {\rm d}u} = {\color{RoyalBlue} -\frac{1}{\rho}\nabla(k(\rho-\rho_0))}$
  * Update: $u^{t+\Delta t} = u^{t+0.5\Delta t} + \Delta t {\color{red} {\rm d}u}$
And step 2 and 1 can be merged. This is nothing but Symplectic Euler integration.

### Fluid dynamics with particles
> taichiCourse01-10 PPT p43 and 75-78

Continuous view:
$$\frac{{\rm D}u}{{\rm D}t}={\color{Green} g} {\color{RoyalBlue} -\frac{1}{\rho}\nabla p} + {\color{Orange} \nu\nabla^2u}$$

Discrete view (using particle):
$$\frac{{\rm d}u_i}{{\rm d}t}=a_i={\color{Green} g} {\color{RoyalBlue} -\frac{1}{\rho}\nabla p(x_i)} + {\color{Orange} \nu\nabla^2u(x_i)}$$

Then the problem comes to: how to evaluate a funcion of ${\color{RoyalBlue} \rho(x_i)}$, ${\color{RoyalBlue} \nabla p(x_i)}$, ${\color{Orange} \nabla^2u(x_i)}$

In WCSPH:
* Find a particle of interest ($i$) and its nerghbours ($j$) with its support radius $h$.
* Compute the acceleration for particle $i$:
  * for i in particles:
    * Step 1: Evaluate density
      $$\rho_i = \sum_j \frac{m_j}{\rho_j}\rho_jW(r_i-r_j, h) = \sum_j m_jW_{ij}$$

    * Step 2: Evaluate viscosity (**anti-sym**)
      $$\nu\nabla^2u_i = \nu\sum_j m_j \frac{u_j-u_i}{\rho_j}\nabla^2W_{ij}$$
      in taichiWCSPH code it's a approximation from @monaghan2005 :
      $$\nu\nabla^2u_i = 2\nu(dimension+2)\sum_j \frac{m_j}{\rho_j}(\frac{u_{ij}\cdot r_{ij}}{\|r_{ij}\|^2+0.01h^2})\nabla W_{ij}$$

    * Evaluate pressure gradient (**sym**), where $p = k(\rho-\rho_0)$
      $$-\frac{1}{\rho_i}\nabla p_i = -\frac{\rho_i}{\rho_i}\sum_j m_j(\frac{p_j}{\rho_j^2}+\frac{p_i}{\rho_i^2})\nabla W_{ij} = -\sum_j m_j(\frac{p_j}{\rho_j^2}+\frac{p_i}{\rho_i^2})\nabla W_{ij}$$

      in taichiWCSPH code, $p = k_1((\rho/\rho_0)^{k_2}-1)$, where $k_1$ is a para about stiffness and $k_2$ is just an exponent.
    * Calculate the acceleration
    * Then do time integration using Symplectic Euler method:
      $$u_i^* = u_i+\Delta t\frac{{\rm d}u_i}{{\rm d}t},\ \ x_i^* = x_i+\Delta tu_i^*$$

### RK4 for WCSPH
> By myself

The momentum equation of WCSPH is as:
$$\frac{{\rm D}u_i}{{\rm D}t}={\color{Green} g} {\color{RoyalBlue} -\frac{1}{\rho_i}\nabla p_i} + {\color{Orange} \nu\nabla^2u_i} = F(u_i)$$

and:
$$u_i^{t+\Delta t} = u_i^t+\frac{\Delta t}{6}(F(u_i^1)+2F(u_i^2)+2F(u_i^3)+F(u_i^4))$$

where:
$$\begin{aligned}
    \begin{array}{ll}
      u^1_i = u^t_i\\
      u^2_i = u^t_i+\frac{\Delta t}{2}(F(u^1_i))\\
      u^3_i = u^t_i+\frac{\Delta t}{2}(F(u^2_i))\\
      u^4_i = u^t_i+\Delta t(F(u^3_i))
    \end{array}
\end{aligned}$$


# SPH for soil

## Constitutive model of soil
Constitutive model is to relate the soil stresses to the strain rates in the plane strain condition.
For **Drucker-Prager** yield criteria: $f=\sqrt{J_2}+\alpha_{\varphi}I_1-k_c=0$ and functions of the Coulomb material constants - the soil internal friction $\varphi$ and cohesion $c$:
$$\alpha_{\varphi}=\frac{\tan\varphi}{\sqrt{9+12\tan^2\varphi}}, k_c=\frac{3c}{\sqrt{9+12\tan^2\varphi}}$$

And for the elastoplastic constitutive equation of Drucker-Prager and *non-associated flow rule*, $g=\sqrt{J_2}+3I_1\cdot\sin\psi$, where $\psi$ is dilatancy angle and in Chalk's thesis $\psi=0$. Of *associated flow rule*, $g=\sqrt{J_2}+\alpha_{\varphi}I_1-k_c$. $g$ is the plastic potential function (塑性势函数).
And the **Von Mises** criterion is: $f = \sqrt{3J_2}-f_c$.
The Von Mises and D-P yield criteria are illustrated in two dimensions:
<div align="center">
  <img width="400px" src="./temp/Yield_criterias.png">
</div>

Here we difine the firse invariant of the stress tensor $I_1$ and the second invariant of the deviatoric stress tensor $J_2$:
$$I_1 = \sigma_{xx}+\sigma_{yy}+\sigma_{zz}\ ,\ J_2 = \frac{1}{2}\boldsymbol{s}:\boldsymbol{s}$$

> **QUESTIONS**
> 1. How does the operator : calculated? **Answer**: double dot product of tensors, also a double tensorial contraction. The double dots operator "eats" a tensor and "spits out" a scalar. As for $\boldsymbol{s}:\boldsymbol{s}$, it represents the sum of squares of each element in $\boldsymbol{s}$.

The increment of the yield function after plastic loading or unloading:
$${\rm d}f=\frac{\partial f}{\partial \boldsymbol{\sigma}} {\rm d}\boldsymbol{\sigma}$$

The stress state is not allowed to exceed the yield surface, and the yield function increment cannot be greater than 0. ${\rm d}f=0$ ensures that the stress state remains on the yield surface during plastic loading.

> **QUESTIONS**
> 1. How to calculate ${\rm d}f$???

And in soil mechanics, the soil pressure $p$ is obtained directly from the equation for **hydrostatic pressure**:
$$p = -\frac{1}{3}(\sigma_{xx}+\sigma_{yy}+\sigma_{zz})$$

We define the **elastic strains** according to the **generalised Hooke's law**:
$$\dot{\boldsymbol{\varepsilon}}^e = \frac{\dot{\boldsymbol{s}}}{2G}+\frac{1-2\nu}{3E}\dot{\sigma}_{kk}\boldsymbol{I}$$

where $\dot{\sigma}_{kk} = \dot{\sigma}_{xx}+\dot{\sigma}_{yy}+\dot{\sigma}_{zz}$, $\boldsymbol{s}$ is the **deviatoric stress tensor**: $\boldsymbol{s} = \boldsymbol{\sigma}+p\boldsymbol{I}$ and $\boldsymbol{I}$ is the identity matrix.

> **QUESTIONS**
> 1. the hydrostatic pressure $p$, is positive or negtive? $\boldsymbol{s}$ is only correct when $p$ is positive as Chalk2020's Appendix A, but in the main text of Chalk2020, $p$ is negtive. **Answer**: Generally it's negtive. When it is positive, the meaning is the average normal stress $\sigma_m = -p$.

The fundamental assumption of plasticity is that the total soil strain rate $\boldsymbol{\dot\varepsilon}$ can be divided into an elastic and a plastic component:
$$\boldsymbol{\dot\varepsilon} = \boldsymbol{\dot\varepsilon}^e+\boldsymbol{\dot\varepsilon}^p$$

With an assumption of a kinematic condition between the *total strain rate* and the *velocity gradients*.
$$\dot{\varepsilon}_{\alpha\beta} = \frac{1}{2}(\frac{\partial u_{\alpha}}{\partial x_{\beta}}+\frac{\partial u_{\beta}}{\partial x_{\alpha}})$$

Consider both a **Von Mises** and a **D-P** yield criterion to distinguish between elastic and plastic material behaviour.

In the elastoplastic model, the stress state is not allowed to exceed the yield surface and I should apply a stress adaptation to particles, after every calculation step. And the elastic and plastic behaviour are distinguished via a stress-dependent yield criterion.

> Bui2008 Section 3.3.1 and Chalk2019 Section 4.3.1

But the stress state is not allowed to exceed the yield surfae. The stress must be checked at every step and adapted if it does not lie within a valid range.
<div align="center">
  <img width="800px" src="./temp/Adaptation_stress_states.png">
</div>

First, the stress state must be adapted if it moves outside the apex of the yield surface, which is konwn as **tension cracking**, in the movement of the stress state at point E to point F. Tension cracking occurss when: $-\alpha_{\varphi}I_1+k_c<0$. And in such circumstances, the hydrostatic stress $I_1$ must be shifted back to the apex of the yield surface by adapting the normal stress components:
$$\hat{\sigma}_{\alpha\alpha} = \sigma_{\alpha\alpha}-\frac{1}{3}(I_1-\frac{k_c}{\alpha_{\varphi}})$$

The second corrective stress treatment must be performed when the stress state exceeds the yield surface during plastic loading, as shown by the path A to B. For the D-P yield criterion, this occurs when: $-\alpha_{\varphi}I_1+k_c<\sqrt{J_2}$. And the stress state must be scaleld back appropriately. For this, a scaling factor $r_{\sigma}$ is introduced: $r_{\sigma} = (-\alpha_{\varphi}I_1+k_c) / \sqrt{J_2}$. The deviatoric shear stress is then reduced via this scaling factor for all components of the stress tensor:
$$\hat{\sigma}_{\alpha\alpha} = r_{\sigma}s_{\alpha\alpha}+\frac{1}{3}I_1$$

$$\hat{\sigma}_{\alpha\beta} = r_{\sigma}s_{\alpha\beta}$$

The procedure of applying these two equations is referred to as the stress-scaling back procedure, or stress modification.
In the SPH implementation of the elastoplastic model, the two corrective treatments described above are applied to the particles that have a stress state outside of the valid range.


## Governing equations
Conservation of mass:
$$\frac{{\rm D} \rho}{{\rm D} t}=-\rho \nabla\cdot\boldsymbol{u}$$

Conservation of momentum:
$$\frac{{\rm D} \boldsymbol{u}}{{\rm D} t}=\frac{1}{\rho} \nabla\cdot\boldsymbol{f}^{\sigma}+\boldsymbol{f}^{ext}$$

Constitutive equation:
$$\frac{{\rm D} \boldsymbol{\sigma}}{{\rm D} t}=\boldsymbol{\tilde{\sigma}} +\nabla\cdot\boldsymbol{f}^u-\boldsymbol{g}^{\varepsilon^p}$$

where:
$$\begin{aligned} \boldsymbol{x} = \left (\begin{array}{c}
    x\\ y
\end{array}\right) \end{aligned}
,
\begin{aligned} \boldsymbol{u} = \left (\begin{array}{c}
    u_x\\ u_y
\end{array}\right) \end{aligned}
,
\begin{aligned} \boldsymbol{f}^{\sigma} = \left (\begin{array}{cc}
    \sigma_{xx}    &\sigma_{xy}\\    \sigma_{xy}    &\sigma_{yy}
\end{array}\right) \end{aligned}
,
\begin{aligned} \boldsymbol{f}^{ext} = \left (\begin{array}{c}
    f^{ext}_x\\ f^{ext}_y
\end{array}\right) \end{aligned}$$

$$\begin{aligned} \boldsymbol{\sigma} = \left (\begin{array}{c}
    \sigma_{xx}\\ \sigma_{yy}\\ \sigma_{xy}\\ \sigma_{zz}
\end{array} \right) \end{aligned}
,
\begin{aligned} \boldsymbol{\tilde{\sigma}} = \left(\begin{array}{c}
      2\sigma_{xy}\omega_{xy}\\ 2\sigma_{xy}\omega_{yx}\\
      \sigma_{xx}\omega_{yx}+\sigma_{yy}\omega_{xy}\\ 0
    \end{array} \right)
    = \left(\begin{array}{c}
      2\sigma_{xy}\omega_{xy}\\ -2\sigma_{xy}\omega_{xy}\\
      (\sigma_{yy}-\sigma_{xx})\omega_{xy}\\ 0
\end{array} \right) \end{aligned}$$

$$\dot\omega_{\alpha\beta}=\frac{1}{2}(\frac{\partial u_{\alpha}}{\partial x_{\beta}}-\frac{\partial u_{\beta}}{\partial x_{\alpha}})\ ,\ \omega_{xy} = \frac{1}{2}(\frac{\partial u_x}{\partial x_y}-\frac{\partial u_y}{\partial x_x})$$

$$\begin{aligned} \boldsymbol{f}^u = \left (\begin{array}{cc}
    D^e_{11}u_x    &D^e_{12}u_y\\ D^e_{21}u_x    &D^e_{22}u_y\\
    D^e_{33}u_y    &D^e_{33}u_x\\ D^e_{41}u_x    &D^e_{42}u_y
\end{array}\right)\end{aligned}
,
\begin{aligned} \boldsymbol{g}^{\varepsilon^p} = \left(\begin{array}{c}
      g^{\varepsilon^p}_{xx}(\boldsymbol{\dot \varepsilon}^p)\\
      g^{\varepsilon^p}_{yy}(\boldsymbol{\dot \varepsilon}^p)\\
      g^{\varepsilon^p}_{xy}(\boldsymbol{\dot \varepsilon}^p)\\
      g^{\varepsilon^p}_{zz}(\boldsymbol{\dot \varepsilon}^p)
\end{array} \right) \end{aligned}
,
\begin{aligned} \boldsymbol{\dot \varepsilon}^p = \left(\begin{array}{c}
      \dot \varepsilon^p_{xx}\\ \dot \varepsilon^p_{yy}\\
      \dot \varepsilon^p_{xy}\\ 0
\end{array} \right) \end{aligned}$$

$$\begin{aligned} \boldsymbol{D}^e = D^e_{pq} = \frac{E}{(1+\nu)(1-2\nu)} \left (\begin{array}{cccc}
    1-\nu  &\nu  &0  &\nu\\ \nu  &1-\nu  &0  &\nu\\
    0  &0  &(1-2\nu)/2  &0\\ \nu  &\nu  &0  &1-\nu\\
\end{array}\right) \end{aligned}$$

$D^e_{pq}$ is the **elastic constitutive tensor**, also the ealstic constitutive matrix reduces in plane strain condition.
$\dot{\omega}_{\alpha\beta}$ is the **spin rate tensor**.
And $\boldsymbol{g}^{\varepsilon^p}$ is a vector containing the plastic terms which is the only difference responsible for plastic deformations between the **elastoplastic** and **Perzyna** constitutive models. In both models, the plastic terms are functions of the plastic strain rate, which is dependent on the state of stress and material parameters.

For the elastoplastic model,
$$\boldsymbol{g}^{\varepsilon^p} = \dot{\lambda}\frac{G}{\sqrt{J_2}\boldsymbol{s}}$$

which is non-zero only when $f = \sqrt{J_2}+\alpha_{\varphi}I_1-k_c = 0$ (and ${\rm d}f=0$), according to the D-P yield criterion, where:
$$\dot{\lambda} = \frac{3\alpha_{\varphi}\dot{\varepsilon}_{kk}+(G/\sqrt{J_2})\boldsymbol{s}\dot{\boldsymbol{\varepsilon}}_{ij}}{27\alpha_{\varphi}K\sin{\psi}+G} = \frac{3\alpha_{\varphi}\dot{\varepsilon}_{kk}+(G/\sqrt{J_2})\boldsymbol{s}\dot{\boldsymbol{\varepsilon}}_{ij}}{G}$$

$$\boldsymbol{s}\dot{\boldsymbol{\varepsilon}}_{ij} = \boldsymbol{s}:\dot{\boldsymbol{\varepsilon}},\ \dot{\boldsymbol{\varepsilon}} = \begin{aligned} \left(\begin{array}{c}
      \dot \varepsilon_{xx}\\ \dot \varepsilon_{yy}\\
      2 \dot \varepsilon_{xy}\\ 0
\end{array} \right) \end{aligned}$$

and $G = E/2(1+\nu)$ is the **shear modulus** and $K = E/3(1-2\nu)$ is the **elastic bulk modulus** (although $K$ is not used here).

And for the Perzyna model,
$$\boldsymbol{g}^{\varepsilon^p} = \boldsymbol{D}^e\frac{\partial \sqrt{3J_2}}{\partial \boldsymbol{\sigma}}(\frac{\sqrt{3J_2}-f_c}{f_c})^{\hat{N}}$$

which is non-zero only when $\sqrt{3J_2}>f_c$ (according to the Von mises yield criterion). And $\hat{N}$ is a model parameter.

> **QUESTIONS**
> 1. How does $\boldsymbol{g}^{\varepsilon^p}$ and $\boldsymbol{\dot\varepsilon}^p$ calculated? Maybe it is different in elastoplastic and Perzyna models.
> 2. How does $\dot{\lambda}$ calculated?
> 3. How does $\frac{\partial\sqrt{3J_2}}{\partial\boldsymbol{\sigma}}$ calculated?
> 4. What number should $\hat{N}$ choose?
> 5. What's the difference between $\dot{\boldsymbol{\varepsilon}}$ and $\dot{\boldsymbol{\varepsilon}^p}$???

### Conservation of mass
The loss of mass equals to the net outflow: (控制体内质量的减少=净流出量)
$$-\frac{\partial m}{\partial t} = -\frac{\partial \rho}{\partial t}{\rm d}x{\rm d}y{\rm d}z=[\frac{\partial (\rho u_x)}{\partial x}+\frac{\partial (\rho u_y)}{\partial y}+\frac{\partial (\rho u_z)}{\partial z}]{\rm d}x{\rm d}y{\rm d}z$$

$$\frac{\partial \rho}{\partial t}+\nabla\cdot(\rho \boldsymbol{u})=0$$

$$\frac{\partial \rho}{\partial t}+\boldsymbol{u}\cdot\nabla\rho+\rho\nabla\cdot\boldsymbol{u}=0, ~ \frac{{\rm D}\rho}{{\rm D}t}=\frac{\partial \rho}{\partial t}+\boldsymbol{u}\cdot\nabla\rho$$

$$\frac{{\rm D}\rho}{{\rm D}t}+\rho\nabla\cdot\boldsymbol{u}=0$$

> **QUESTIONS**
> 1. Where do these equations come from?

### Conservation of momentum
根据牛顿流体的本构方程，推导获得流体的动量方程。
无粘流动的动量方程即欧拉方程（惯性力$\frac{{\rm D}\boldsymbol{u}}{{\rm D}t}$，体积力$\boldsymbol{f}$，压差力$-\frac{1}{\rho}\nabla p$，普通动量方程中的粘性力$\frac{\mu}{\rho}\nabla^2\boldsymbol{u}+\frac{1}{3}\frac{\mu}{\rho}\nabla(\nabla\cdot\boldsymbol{u})=0$）：
$$\frac{{\rm D}\boldsymbol{u}}{{\rm D}t}=\boldsymbol{f}-\frac{1}{\rho}\nabla p$$

流体静止时，粘性力项自然为0，惯性力项也为0，即退化为欧拉静平衡方程$\nabla p=\rho\boldsymbol{f}$。

> **QUESTIONS**
> 1. The momentum considered here is not the same as Navier-Stokes equation but what???

## Standard soil SPH

### Discretization
> @chalk2020 Section 3.1

The discrete governing equations of soil motion in the framework of standard SPH are therefore:
$$\frac{{\rm D} \rho_i}{{\rm D} t} = -\sum_j m_j(\boldsymbol{u}_j-\boldsymbol{u}_i)\cdot\nabla W_{ij}$$

$$\frac{{\rm D} \boldsymbol{u}_i}{{\rm D} t} = \sum_j m_j(\frac{\boldsymbol{f}_i^{\sigma}}{\rho_i^2}+\frac{\boldsymbol{f}_j^{\sigma}}{\rho_j^2})\cdot\nabla W_{ij}+\boldsymbol{f}^{ext}_i$$

$$\frac{{\rm D} \boldsymbol{\sigma}_i}{{\rm D} t} = \boldsymbol{\tilde{\sigma}}_i+\sum_j \frac{m_j}{\rho_j}(\boldsymbol{f}_j^u-\boldsymbol{f}_i^u)\cdot\nabla W_{ij}-\boldsymbol{g}_i^{\varepsilon^p}$$

In the current work, each SPH particle is assigned the same, constant density for the duration of the simulation. We treat the soil as incompressible and consequently do not update density through this way.


### RK4 for standard soil SPH
> @Chalk2020, Appendix B.

The considered governing SPH equations are summarised as:
$$\frac{{\rm D} \boldsymbol{u}_i}{{\rm D} t} = \sum_j m_j(\frac{\boldsymbol{f}_i^{\sigma}}{\rho_i^2}+\frac{\boldsymbol{f}_j^{\sigma}}{\rho_j^2})\cdot\nabla W_{ij}+\boldsymbol{f}^{ext}_i = F_1(\boldsymbol{\sigma}_i)$$

$$\frac{{\rm D} \boldsymbol{\sigma}_i}{{\rm D} t} = \boldsymbol{\tilde{\sigma}}_i+\sum_j \frac{m_j}{\rho_j}(\boldsymbol{f}_j^u-\boldsymbol{f}_i^u)\cdot\nabla W_{ij}-\boldsymbol{g}_i^{\varepsilon^p} = F_2(\boldsymbol{u}_i,\boldsymbol{\sigma}_i)$$

Using the fourth order Runge-Kutta (RK4) method:
$$\boldsymbol{u}_i^{t+\Delta t} = \boldsymbol{u}_i^t + \frac{\Delta t}{6}(F_1(\boldsymbol{\sigma}^1_i)+2F_1(\boldsymbol{\sigma}^2_i)+2F_1(\boldsymbol{\sigma}^3_i)+F_1(\boldsymbol{\sigma}^4_i))$$

$$\boldsymbol{\sigma}_i^{t+\Delta t} = \boldsymbol{\sigma}_i^t + \frac{\Delta t}{6}(F_2(\boldsymbol{u}^1_i,\boldsymbol{\sigma}^1_i)+2F_2(\boldsymbol{u}^2_i,\boldsymbol{\sigma}^2_i)+2F_2(\boldsymbol{u}^3_i,\boldsymbol{\sigma}^3_i)+F_2(\boldsymbol{u}^4_i,\boldsymbol{\sigma}^4_i))$$

where:
$$\begin{aligned}
    \begin{array}{ll}
      \boldsymbol{u}^1_i = \boldsymbol{u}^t_i &\boldsymbol{\sigma}^1_i = \boldsymbol{\sigma}^t_i\\
      \boldsymbol{u}^2_i = \boldsymbol{u}^t_i+\frac{\Delta t}{2}(F_1(\boldsymbol{\sigma}^1_i)) &\boldsymbol{\sigma}^2_i = \boldsymbol{\sigma}^t_i+\frac{\Delta t}{2}(F_2(\boldsymbol{u}^1_i, \boldsymbol{\sigma}^1_i))\\
      \boldsymbol{u}^3_i = \boldsymbol{u}^t_i+\frac{\Delta t}{2}(F_1(\boldsymbol{\sigma}^2_i)) &\boldsymbol{\sigma}^3_i = \boldsymbol{\sigma}^t_i+\frac{\Delta t}{2}(F_2(\boldsymbol{u}^2_i, \boldsymbol{\sigma}^2_i))\\
      \boldsymbol{u}^4_i = \boldsymbol{u}^t_i+\Delta t(F_1(\boldsymbol{\sigma}^3_i)) &\boldsymbol{\sigma}^4_i = \boldsymbol{\sigma}^t_i+\Delta t(F_2(\boldsymbol{u}^3_i, \boldsymbol{\sigma}^3_i))
    \end{array}
\end{aligned}$$

In standard SPH, these eight eqs are spatially resolved at each calculation step by calculating $\boldsymbol{u}_i^{t+\Delta t}$ and $\boldsymbol{\sigma}_i^{t+\Delta t}$ at each particle.

### Steps
* Key point and aim: update the position, velocity and stress.
* Known $\Delta x$, $\nu$, $E$, $D_{pq}^e$, $\rho_0$, $\boldsymbol{f}^{ext} = \vec{g}$, and paras for D-P yield criteria $c$, $\varphi$, $\alpha_{\varphi}$ and $k_c$.
* Given $\boldsymbol{x}_i^1$, $\boldsymbol{u}_i^1$, $\boldsymbol{\sigma}_i^1$.
* Step 1: calculate terms $\boldsymbol{f}^{\sigma}$ and $\boldsymbol{f}^u$.
* Step 2: update boundary consitions and adapt the stress.
* Step 3: calculate the gradient terms $(\nabla\cdot\boldsymbol{f}^{\sigma})_i$ and $(\nabla\cdot\boldsymbol{f}^u)_i$.
* Step 4: calculate the additional terms for the momentum equation, mainly the body force $\boldsymbol{f}^{ext}_i$ in which gravity is the only one considered. Also if included, the artificial viscosity is calculated here.
* Step 5: calculate the additional terms for the constitutive equation, mainly the plastic strain function $\boldsymbol{g}^{\varepsilon^p}_i$.
  * When calculating each particle, the stress state is checked to see if the yield criterion has been met. If the stress state lies within the elastic range ($f<0$ or $f=0,\ {\rm d}f<0$), then $\boldsymbol{g}^{\varepsilon^p}_i = 0$. Otherwise ($f=0,\ {\rm d}f=0$), the plastic term is calculated and $\boldsymbol{g}^{\varepsilon^p}_i$ is non-zero.
  * The plastic term is a function of stress $\boldsymbol{\sigma}$ and velocity gradients $\nabla \boldsymbol{u}$.
  * For large deformation problems, the Jaumann stress rate $\tilde{\boldsymbol{\sigma}}_i$ is also updated. This involves gradients of the velocity $\nabla \boldsymbol{u}$.
* Step 6: compute $F_1$ and $F_2$ on particles.
* Step 7: calculate $\boldsymbol{u}_i^2$ and $\boldsymbol{\sigma}_i^2$.
* Step 8: if necessary, the boundary conditions and stress state are again updated.
* Step 9: repeat Steps 1-8 to obtain$\boldsymbol{u}_i^3$, $\boldsymbol{\sigma}_i^3$, $\boldsymbol{u}_i^4$ and $\boldsymbol{\sigma}_i^4$. Then update the velocity $\boldsymbol{u}_i^{t+\Delta t}$ and the stress $\boldsymbol{\sigma}_i^{t+\Delta t}$ at the subsequent time step, also the positions $\boldsymbol{x}_i^{t+\Delta t}$ of the particles.

As for the calculation of strain item:
<div align="center">
  <img width="750px" src="./temp/flowchart_item_strain.svg">
</div>
As for the implementation of RK4:
<div align="center">
  <img width="300px" src="./temp/flowchart_RK4_soil.svg">
</div>




## Stress-Particle SPH

# FEM-SPH

