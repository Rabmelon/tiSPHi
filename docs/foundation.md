# Foundation of SPH

## Basic formulations

### The integral estimation
> @Chalk2019 4.2

The integral approximation involves representing a function $f(x)$ as an integral:
$$f(x)=\int_{\Omega}f(x')\delta(x-x'){\rm d}x'$$

where $\Omega$ donates the integral domain and $\delta(x-x')$ is the Dirac delta function defined as:
$$\delta(x-x')=\begin{cases}
  1, &x=x' \\ 0, &x\neq x'
\end{cases} $$

In the derivation of SPH, the integral approximation is obtained by replacing the Dirac delta with a smoothing function $W$:
$$\langle f(x)\rangle=\int_{\Omega}f(x')W(x-x', h_s){\rm d}x' $$

The smoothing function, or kernel, must satisfy three conditions:
$$\begin{aligned}
    \begin{array}{rl}
    Normalisation\ condition: &\int_{\Omega}W(x-x',h_s){\rm d}x'=1 \\
    Compact\ support: &W(x-x',h_s)=0\ when\ |x-x'|>\kappa h_s \\
    Satisfy\ the\ Dirac\ delta\ function\ condition: &\underset{h\rightarrow0}{\lim}W(x-x',h_s)=\delta(x-x')
    \end{array}
\end{aligned} $$

With these conditions for the smoothing kernel, the integral approximation is of second order accuracy, so that:
$$f(x)=\int_{\Omega}f(x')W(x-x',h_s){\rm d}x'+O(h_s^2) $$

### Particle approximations
The particle approximation is utilised to discretise the integral equation over a set of particles. This involves writing the integral approximation in discrete form using a summation approach:
$$\langle f(x)\rangle\approx\sum_{j=1}^Nf(x_j)W(x-x_j,h_s)V_j $$

where $V_j$ is the discrete volume at each point and $N$ is the total number of particles within the region defined by $W$ and $h_s$. Here, the function $f(x)$ is approximated by summing over all discrete particles $j$ within the domain of influence at the position $x$. So the summation approach can be expressed for a specific particle $i$ as:
$$f(x_i)=\sum_{j=1}^N\frac{m_j}{\rho_j}f(x_j)W(x_i-x_j,h_s) $$

This equation describes the SPH evaluation of a function or variable at a particle $i$.

### Spatial derivatives
> taichiCourse01-10 PPT p59 and 72

Approximate a function $f(x)$ using finite probes $f(x_j)$, and the degree of freedom $(x)$ goes inside the kernel functions (**anti-sym** and **sym**).
* SPH discretization:
$$f(x) \approx \sum_j \frac{m_j}{\rho_j}f(x_j)W(x-x_j, h) $$

* SPH spatial derivatives:
$${\color{Salmon} \nabla} f(x) \approx \sum_j \frac{m_j}{\rho_j}f(x_j){\color{Salmon} \nabla}W(x-x_j, h)  $$

$${\color{Salmon} \nabla\cdot} \boldsymbol{F}(x) \approx \sum_j \frac{m_j}{\rho_j}\boldsymbol{F}(x_j){\color{Salmon} \cdot\nabla}W(x-x_j, h)  $$

$${\color{Salmon} \nabla\times} \boldsymbol{F}(x) \approx -\sum_j \frac{m_j}{\rho_j}f(x_j){\color{Salmon} \times\nabla}W(x-x_j, h)  $$

$${\color{Salmon} \nabla^2} f(x) \approx \sum_j \frac{m_j}{\rho_j}f(x_j){\color{Salmon} \nabla^2}W(x-x_j, h)  $$

with $W(x_i-x_j, h) = W_{ij}$ in discrete view, and:
$$\nabla W_{ij}=\frac{\partial W_{ij}}{\partial x_i} $$

$$\nabla^2W_{ij}=\frac{\partial^2 W_{ij}}{\partial x_i^2} $$

> **QUESTIONS**
> 1. How to calculate $\nabla W$ and $\nabla^2 W$? **ANSWER**: just directly take the partial derivative!

### Kernel functions
> @koschierSmoothedParticleHydrodynamics2019

#### The cubic spline kernel
$$W_{ij}=W(\boldsymbol{r}, h)=k_d\begin{cases}
  6(q^3-q^2)+1, &0\leq q \leq 0.5 \\ 2(1-q)^3, &0.5 < q \leq 1 \\ 0, &otherwise
\end{cases} $$

where $q = \Vert\boldsymbol{r}\Vert/h$, $k_d$ is the kernel normalization factors for respective dimensions $d=1,2,3$ and $k_1=\frac{4}{3h}$, $k_2=\frac{40}{7\pi h^2}$, $k_3=\frac{8}{\pi h^3}$. The kernel is $C^2$ continuous.

The first-order derivation:
$$\nabla W_{ij}=\frac{\partial W}{\partial x_i}=\frac{\partial W}{\partial q}\cdot\frac{\partial q}{\partial r}\cdot\frac{\partial r}{\partial x_i}=\frac{\partial W}{\partial q}\cdot\frac{1}{h}\cdot\frac{x_i-x_j}{\Vert\boldsymbol{r}\Vert},\ \boldsymbol{r}=x_i-x_j $$

$$\frac{\partial W}{\partial q}=k_d\begin{cases}
  6(3q^2-2q), &0\leq q \leq 0.5 \\ -6(1-q)^2, &0.5 < q \leq 1 \\ 0, &otherwise
\end{cases} $$

The second-order derivation:
$$\nabla^2W_{ij}=\frac{\partial^2 W}{\partial x_i^2}=\frac{\partial}{\partial x_i}(\frac{\partial W}{\partial x_i})=\frac{\partial^2 W}{\partial q^2}\cdot(\frac{\partial q}{\partial r}\cdot\frac{\partial r}{\partial x_i})^2=\frac{\partial^2 W}{\partial q^2}\cdot\frac{1}{h^2}\cdot\frac{(x_i-x_j)^2}{\Vert\boldsymbol{r}\Vert^2} $$

$$\frac{\partial^2 W}{\partial q^2}=k_d\begin{cases}
  6(6q-2), &0\leq q \leq 0.5 \\ 12(1-q), &0.5 < q \leq 1 \\ 0, &otherwise
\end{cases} $$

> **QUESTIONS**
> 1. The second-order derivation is wrong!!!!!!!!!!!!!!!!!!

#### The Wenland kernel
$$W_{ij}=W(\boldsymbol{r}, h)=k_d\begin{cases}
  (2-q)^4(2q+1), &0\leq q \leq 2 \\ 0, &otherwise
\end{cases} $$

where $q = \Vert\boldsymbol{r}\Vert/h$, $k_d$ is the kernel normalization factors for respective dimensions $d=2,3$ and $k_2=\frac{7}{32\pi h^2}$, $k_3=\frac{21}{128\pi h^3}$. The kernel is $C^2$ continuous.

The first-order derivation:
$$\nabla W_{ij}=k_d'(2-q)^3\boldsymbol{r} $$
where $$k_d'$ for respective dimensions $d=2,3$: $k_2'=-\frac{35}{32\pi h^4}$, $k_3'=-\frac{105}{128\pi h^5}$. The kernel is $C^2$ continuous.


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
  <img width="300px" src="./img/Dummy_particles.png">
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