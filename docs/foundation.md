# Foundation of SPH

## Basic formulations

### The integral estimation

> @Chalk2019 4.2

The integral approximation involves representing a function $f(\boldsymbol{x})$ as an integral:

$$f(\boldsymbol{x})=\int_{\Omega}f(\boldsymbol{x}')\delta(\boldsymbol{x}-\boldsymbol{x}'){\rm d}\boldsymbol{x}'$$

where $\Omega$ donates the integral domain and $\delta(\boldsymbol{x}-\boldsymbol{x}')$ is the Dirac delta function defined as:

$$\delta(\boldsymbol{x}-\boldsymbol{x}')=\begin{cases}
  1, &\boldsymbol{x}=\boldsymbol{x}' \\ 0, &\boldsymbol{x}\neq\boldsymbol{x}'
\end{cases} $$

In the derivation of SPH, the integral approximation is obtained by replacing the Dirac delta with a smoothing function $W$:

$$\langle f(\boldsymbol{x})\rangle=\int_{\Omega}f(\boldsymbol{x}')W(\boldsymbol{x}-\boldsymbol{x}', h){\rm d}\boldsymbol{x}' $$

The smoothing function, or kernel, must satisfy three conditions:

$$\begin{aligned}
    \begin{array}{rl}
    Normalisation\ condition: &\int_{\Omega}W(\boldsymbol{x}-\boldsymbol{x}',h){\rm d}\boldsymbol{x}'=1 \\
    Compact\ support: &W(\boldsymbol{x}-\boldsymbol{x}',h)=0\ when\ |\boldsymbol{x}-\boldsymbol{x}'|>\kappa h \\
    Satisfy\ the\ \delta\ function\ condition: &\underset{h\rightarrow0}{\lim}W(\boldsymbol{x}-\boldsymbol{x}',h)=\delta(\boldsymbol{x}-\boldsymbol{x}')
    \end{array}
\end{aligned} $$

With these conditions for the smoothing kernel, the integral approximation is of second order accuracy, so that:

$$f(\boldsymbol{x})=\int_{\Omega}f(\boldsymbol{x}')W(\boldsymbol{x}-\boldsymbol{x}',h){\rm d}\boldsymbol{x}'+O(h^2) $$

### Particle approximations

The particle approximation is utilised to discretise the integral equation over a set of particles. This involves writing the integral approximation in discrete form using a summation approach:

$$\langle f(\boldsymbol{x})\rangle=\sum_{j=1}^Nf(\boldsymbol{x}_j)W(\boldsymbol{x}-\boldsymbol{x}_j,h)V_j $$

where $V_j$ is the discrete volume at each point and $N$ is the total number of particles within the region defined by $W$ and $h$. Here, the function $f(\boldsymbol{x})$ is approximated by summing over all discrete particles $j$ within the domain of influence at the position $\boldsymbol{x}$. So the summation approach can be expressed for a specific particle $i$ as:

$$f(\boldsymbol{x}_i)\approx\sum_{j=1}^N V_jf(\boldsymbol{x}_j)W(\boldsymbol{x}_i-\boldsymbol{x}_j,h) $$

This equation describes the SPH evaluation of a function or variable at a particle $i$.

Now, replace $f\equiv\nabla f$, we have:

$$\nabla f(\boldsymbol{x}_i)\approx\sum_{j=1}^N V_j\nabla f(\boldsymbol{x}_j)W(\boldsymbol{x}_i-\boldsymbol{x}_j,h) $$

Applying the Gaussian theorem (in a symmetric and positive weighting function, ${\rm d}V=\vec n\cdot{\rm d}S$, and also $\partial W/\partial \boldsymbol{x}_i=-\partial W/\partial \boldsymbol{x}_j$):

$$\nabla f(\boldsymbol{x}_i)\approx\sum_{j=1}^N V_jf(\boldsymbol{x}_j)\nabla_i W(\boldsymbol{x}_i-\boldsymbol{x}_j,h) $$


### Spatial derivatives

> @taichiCourse01-10 PPT p59 and 72

Approximate a function $f(\boldsymbol{x})$ using finite probes $f(\boldsymbol{x}_j)$, and the degree of freedom $(\boldsymbol{x})$ goes inside the kernel functions (**anti-sym** and **sym**).

* SPH discretization:

$$f(x) \approx \sum_j V_jf(x_j)W(x-x_j, h) $$

* SPH spatial derivatives:

$${\color{Salmon} \nabla} f(x) \approx \sum_j V_jf(x_j){\color{Salmon} \nabla}W(x-x_j, h)  $$

$${\color{Salmon} \nabla\cdot} \boldsymbol{f}(x) \approx \sum_j V_j\boldsymbol{f}(x_j){\color{Salmon} \cdot\nabla}W(x-x_j, h)  $$

$${\color{Salmon} \nabla\times} \boldsymbol{f}(x) \approx -\sum_j V_j\boldsymbol{f}(x_j){\color{Salmon} \times\nabla}W(x-x_j, h)  $$

$${\color{Salmon} \nabla^2} f(x) \approx \sum_j V_jf(x_j){\color{Salmon} \nabla^2}W(x-x_j, h)  $$

with $W(\boldsymbol{x}_i-\boldsymbol{x}_j, h) = W_{ij}$ in discrete view, and:

$$\nabla_i W_{ij}=\frac{\partial W_{ij}}{\partial \boldsymbol{x}_i} $$

$$\nabla^2_i W_{ij}=\frac{\partial^2 W_{ij}}{\partial \boldsymbol{x}_i^2} $$

> **QUESTIONS**
>
> 1. How to calculate $\nabla W$ and $\nabla^2 W$? **ANSWER**: just directly take the partial derivative!

### Improving approximations for spatial derivatives

> @taichiCourse01-10 PPT p60-70

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
    * 或许可以说，当$f$是粒子$i$和$j$的相互作用时，用对称型；当$f$是粒子本身的属性时，用反对称型？

Like to approximate the velocity gradient $\nabla\boldsymbol{v}$, using anti-symmtric form to ensure that the gradients of a constant velocity field vanish:

$$v_{\alpha,\beta}=\frac{\partial v^{\alpha}}{\partial x^{\beta}}=\sum_jV_j(v^{\alpha}_j-v^{\alpha}_i)\cdot\nabla_iW_{ij}^{\beta}$$

## Kernel functions

### Pairing instability

> @bui lecture

A commmon misconception of SPH:

Wrong choice of kernel function lead to "pairing instability" and this was often cited as SPH instability issue!

The source of pairing instability in SPH comes from the gradient term $\nabla_iW_{ij}$:

1. Each kernel function **could only accomodate** a certain number of particles, which means forcing more particles in a kernel approximation cause SPH errors.
2. Each kernel function has **a inflection point** (i.e. zero kernel gradient), which means particles at this point would not gain enough repulsive force due to SPH errors.
3. However, the appropriate choice of kernel function and its parameters would **completely eliminate pairing instability** issues.

**Remarks**:

* Pairing instability issue only occurs in a situation where there are more neighbouring particles in the influence domain that a kernel function can accodomate.
* Kernel functions whose Fourier transformation is negative for some wave vectors will trigger pairing instability at sufficient large number of neighbouring particles. (So that's why Wendland C2 kernel function wins, because all of its Fourier transformations for wave vectors are positive.)
* If we actually use a suitable kernel function with a suitable supporting length, we don't have the problem of pairing instability, and this issue is not because SPH instability.

### The cubic spline kernel

> @bui2021

$$W_{ij}=W(\boldsymbol{r}, h)=k_d\begin{cases}
  \frac{2}{3}-q^2+\frac{1}{2}q^3, &0\leq q \leq 1 \\ \frac{1}{6}(2-q)^3, &1 < q \leq 2 \\ 0, &otherwise
\end{cases} $$

where $q = \Vert\boldsymbol{r}\Vert/h$, $k_d$ is the kernel normalization factors for respective dimensions $d=1,2,3$ and $k_1=\frac{1}{h}$, $k_2=\frac{15}{7\pi h^2}$, $k_3=\frac{3}{2\pi h^3}$.

The first-order derivation:

$$\nabla W_{ij}=\frac{\partial W}{\partial x_i}=\frac{\partial W}{\partial q}\cdot\frac{\partial q}{\partial r}\cdot\frac{\partial r}{\partial x_i}=\frac{\partial W}{\partial q}\cdot\frac{1}{h}\cdot\frac{x_i-x_j}{\Vert\boldsymbol{r}\Vert},\ \boldsymbol{r}=x_i-x_j $$

$$\frac{\partial W}{\partial q}=k_d\begin{cases}
  \frac{3}{2}q^2-2q, &0\leq q \leq 1 \\ -\frac{1}{2}(2-q)^2, &1 < q \leq 2 \\ 0, &otherwise
\end{cases} $$

The second-order derivation:

$$\nabla^2W_{ij}=\frac{\partial^2 W}{\partial x_i^2}=\frac{\partial}{\partial x_i}(\frac{\partial W}{\partial x_i})=\frac{\partial^2 W}{\partial q^2}\cdot(\frac{\partial q}{\partial r}\cdot\frac{\partial r}{\partial x_i})^2=\frac{\partial^2 W}{\partial q^2}\cdot\frac{1}{h^2}\cdot\frac{(x_i-x_j)^2}{\Vert\boldsymbol{r}\Vert^2} $$

$$\frac{\partial^2 W}{\partial q^2}=k_d\begin{cases}
  3q-2, &0\leq q \leq 1 \\ 2-q, &1 < q \leq 2 \\ 0, &otherwise
\end{cases} $$

> **QUESTIONS**
>
> 1. The second-order derivation is wrong!!!!!!!!!!!!!!!!!!
> 2. Why $0<q<2$? The support domain should be $h$, or $2h$? It only depends on the choice and should be $2$ in kernel function but $h$ in neighbor search? **ANSWER**: @peng lecture. $h$ is called "smoothing length" and controls the shape of kernel function, $\kappa h$ is the compact support radius determining the region of support domain, also the neighbour search condition should be $|\boldsymbol{x}_i-\boldsymbol{x}_j|\le\kappa h$. $\kappa$ is usually taking as $2$.

> @koschier2019

$$W_{ij}=W(\boldsymbol{r}, h)=k_d\begin{cases}
  6(q^3-q^2)+1, &0\leq q \leq 0.5 \\ 2(1-q)^3, &0.5 < q \leq 1 \\ 0, &otherwise
\end{cases}$$

where $q = \Vert\boldsymbol{r}\Vert/h$, $k_d$ is the kernel normalization factors for respective dimensions $d=1,2,3$ and $k_1=\frac{4}{3h}$, $k_2=\frac{40}{7\pi h^2}$, $k_3=\frac{8}{\pi h^3}$.

### The Wendland C2 kernel

> @bui2021 2.3

$$W_{ij}=W(\boldsymbol{r}, h)=k_d\begin{cases}
  (1-0.5q)^4(1+2q), &0\leq q \leq 2 \\ 0, &otherwise
\end{cases}$$

where $q = \Vert\boldsymbol{r}\Vert/h$, $k_d$ is the kernel normalization factors for respective dimensions $d=2,3$ and $k_2=\frac{7}{4\pi h^2}$, $k_3=\frac{21}{2\pi h^3}$. For 1d, the formulation is changed. The kernel is $C^2$ continuous.

The first-order derivation:

$$\nabla_i W_{ij}^{\alpha}=k_d(-5q)(1-0.5q)^3\cdot\frac{1}{h}\cdot\frac{x_i^{\alpha}-x_j^{\alpha}}{\Vert\boldsymbol{r}\Vert}$$

The second-order derivation:

???

### CSPM normalisation

> @Bui2021, @Chalk2020, @Chen1999

A corrective term can be multiplied to the smoothing kernel to improve the accuracy of the SPH approximation. The Corrective Smoothed Particle Method (CSPM) increases the accuracy of the kernel via a normalisation procedure, which is based on a Taylor series expansion of the SPH equations.

After doing the Taylor expansion of $f_j$:

$$f_j=f_i+\frac{\partial f_i}{\partial \boldsymbol{x}_i^{\alpha}}(\boldsymbol{x}_j-\boldsymbol{x}_i)^{\alpha}+O(h^2)$$

Neglecting all the derivative terms, we have the corrective kernel estimate:

$$f_i\approx\sum_jV_jf_jW_{ij}\approx f_i{\color{Salmon}\sum_jV_jW_{ij}}+O(h)$$

where the term ${\color{Salmon} 1}$ should be $1$. Then divide the ${\color{Salmon} 1}$ term, leading to:

$$f_i\approx\frac{\sum_jV_jf_jW_{ij}}{\sum_jV_jW_{ij}}=\sum_jV_jf_jW_{ij}^{CSPM}$$

$$W_{ij}^{CSPM}=\frac{W_{ij}}{\sum_kV_kW_{ik}}$$

and for the first derivative:

$$\nabla^{\beta}f_i\approx\sum_jV_jf_j\nabla^{\beta}_iW_{ij}\approx f_i{\color{Salmon} \sum_jV_j\nabla^{\beta}_iW_{ij}}+\frac{\partial f_i}{\partial \boldsymbol{x}_i^{\alpha}}{\color{Green} \sum_jV_j(\boldsymbol{x}_j-\boldsymbol{x}_i)^{\alpha}\nabla^{\beta}_iW_{ij}}+O(h^2)$$

where the term ${\color{Salmon} 1}$ should be $0$ and the term ${\color{Green} 2}$ should be $1$ or $\delta^{\alpha\beta}$.

To completely eliminate these errors, one could subtract the ${\color{Salmon} 1}$ term and then divide the ${\color{Green} 2}$ term, leading to the normalised SPH formulation for the kernel derivative:

$$\nabla f_i\approx\sum_{j=1} V_j(f_j-f_i)\boldsymbol{L_{ij}}\nabla_i W_{ij}=\sum_{j=1} V_j(f_j-f_i)\nabla_iW_{ij}^{CSPM}$$

$$\boldsymbol{L_{ij}}=[\sum_jV_j(\boldsymbol{x}_j-\boldsymbol{x}_i)^{\alpha}\nabla^{\beta}_iW_{ij}]^{-1}$$

$$\nabla_iW_{ij}^{CSPM}=[\sum_kV_k(\boldsymbol{x}_k-\boldsymbol{x}_i)^{\alpha}\nabla^{\beta}_iW_{ik}]^{-1}\nabla_i W_{ij} $$

$\boldsymbol{L}_{ij}$ is the normalised matrix. This formulation has second order accuracy. Additionally, it also removes the boundary effects. But although it is a good operator, it also may become a bad one. Such as in formulations that DO NOT conserve linear momentum like force and stress. So we need an operator to conserve both linear and angular momenta.

### Shepard correction

> @Liu2012, @Shepard1968, @Reinhardt2019

The Shepaard correction of the smoothing kernel $W$ addresses errors introduced by the SPH discretization process due to irregularly distributed particels inside the material domain. Especially near open boundaries (e.g. fluid-air interfaces), the computation of the fluid quantity is error-prone due to the lack of neighboring particles. The corrected kernel function is:

$$W_{ij}^{sh}=\frac{W_{ij}}{\sum_kV_kW_{ik}}$$

And this is a zero-order reinitialization [@pysph].

> **QUESTIONS**
>
> 1. Isn't it the CSPM for $f_i$?

### MLS correction

> @Nguyen2017, @Liu2012, @Dilts1999, @Belytschko1998

The moving least square (MLS) method is adopted to correct the kernel function.

$$W_{ij}^{MLS}=[\beta_0+\beta_x(x_i-x_j)+\beta_y(y_i-y_j)]W_{ij}$$

where

$$[\beta_0,\beta_x,\beta_y]^T=(\sum_jV_j\boldsymbol{A}W_{ij})^{-1}[1,0,0]^T$$

$$\boldsymbol{A}=\left[\begin{matrix}
  1 &x_i-x_j &y_i-y_j \\ x_i-x_j &(x_i-x_j)^2 &(x_i-x_j)(y_i-y_j) \\ y_i-y_j &(x_i-x_j)(y_i-y_j) &(y_i-y_j)^2
\end{matrix}\right]$$

or

$$\boldsymbol{A}=pp^T\ and\ p=[1,\ x_i-x_j,\ y_i-y_j]^T$$

And this is the first order correction that reproduces exactly the linear variation of quantity.

> **QUESTIONS**
>
> 1. Only suitable for 2D problems?
> 2. How to compare with CSPM?

## Neighbour search

### Grid method

### Hash grid method


## Boundary treatment

### Types of boundary conditions

> @Bui2021, @Bui lecture

Like any other numerical methods, the treatment of boundary condditions in SPH is required to facilitate its applications to a wide range of engineering problems.

1. Solid boundary conditions such as **fully-fixed**, **free-roller** (or **no-slip** and **free-slip**) or **symmetric**.
2. Flexible **confining stress** boundary conditions such as prescribed pressure of confining stress in triaxial tests.
3. **Free-surface** condition.

<div align="center">
  <img width="500px" src="https://github.com/Rabmelon/tiSPHi/raw/master/docs/img/Boundary_conditions_basic.png">
</div>


### Simplest treatments for water

> @taichiCourse01-10 PPT p43 and 79-85

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
        2. Pad a layer of solid particles (or called ghost particles, dummy particles) underneath the boundaries with $\rho_{solid} = \rho_0$ and $v_{solid} = 0$. 总体来说比方法1稳定，但可能会导致边界附近粒子的数值黏滞。

> **QUESTIONS**
>
> 1. 多介质的流体混合时，多介质的界面？？？

### Free surface problems

The particles that comprise the free surface should satisfy a stress-free condition. When considering large deformations this first requires the detection of free surface particles, followed by a transformation of the stress tensor so that the normal and tangential components are 0.

> **QUESTIONS**
>
> 1. BUT how does the free surface condition implement?

### Dummy particles (or fixed-boundary particles)

> @Chalk2020, @Bui2021, @Zhao2019

虚拟的边界粒子，本身不具有具体的属性数值。在每一个Step中，在每一个粒子的计算中，先加入一个对Dummy particle对应属性的赋值。

The dummy particle (or ghost particle) method is used to represent the wall boundary. For dummy particles outside the wall boundary, they are spaced apart by $\Delta x$. For repulsive particles at the wall boundary, are $\Delta x/2$.

<div align="center">
  <img width="300px" src="https://github.com/Rabmelon/tiSPHi/raw/master/docs/img/Dummy_particles.png">
</div>

For an interior particle A (circle) that contains a dummy particle B (square) within its neighbourhood, the normal distances $d_A$ and $d_B$ to the wall are calculated. An artificial velocity $\boldsymbol{v}_B$ is then assigned to the dummy particle:

$$\boldsymbol{v}_B = -\frac{d_B}{d_A}\boldsymbol{v}_A$$

To account for extremely large values of the dummy particle velocity when an interior particle approaches the boundary (and $d_A$ approaches 0), a parameter $\beta$ is introduced:

$$\boldsymbol{v}_B = (1-\beta)\boldsymbol{v}_A+\beta\boldsymbol{v}_{wall}\ ,\ \beta = min(\beta_{max}, 1+\frac{d_B}{d_A})$$

$\beta_{max}$ have been found to be between $1.5\rightarrow2$, and here we use $\beta_{max}=1.5$.

And we have $\boldsymbol{\sigma}_B=\boldsymbol{\sigma}_A$ and $p_B=p_A$, etc. The simple definition ensures that there is a uniform stress distribution for the particles that are near the wall boundaries, and it contributes to smooth stress distributions (through the $\boldsymbol{f}^{\sigma}$ term) on the interior particles in the equation of momentum through the particle-dummy interaction.

From *Bui's lecture*, the stress and velocity of fixed boundary particles ($a$) can also be interpolated from real particles ($b$). For the fully-fixed boundary:

$$\boldsymbol{v}_i^a=-\sum_jV_i^b\boldsymbol{v}_i^b\widetilde{W}_{ij},\ \boldsymbol{\sigma}_i^a=\sum_jV_i^b\boldsymbol{\sigma}_i^b\widetilde{W}_{ij}$$

While for the free-slip boundary:

$$\boldsymbol{v}_i^{a,n}=\sum_jV_j^b(\boldsymbol{v}_i^{a,n}-2\boldsymbol{v}_j^{b,n})\widetilde{W}_{ij}\ or\ \boldsymbol{v}_i^{a,n}=-\sum_jV_j^b\boldsymbol{v}_j^{b,n}\widetilde{W}_{ij},\ \boldsymbol{v}_i^{a,t}=\sum_jV_j^b\boldsymbol{v}_j^{b,t}\widetilde{W}_{ij}$$

$$\sigma_i^{a, \alpha\beta}=\begin{cases}
  \sum_jV_j^b\sigma_j^{b, \alpha\beta}\widetilde{W}_{ij}, &\alpha=\beta \\ -\sum_jV_j^b\sigma_j^{b, \alpha\beta}\widetilde{W}_{ij}, & \alpha \neq \beta
\end{cases}$$

where $\boldsymbol{v}^{a,n}$ and $\boldsymbol{v}^{a,t}$ are the normal and shear velocity components of particle $a$ with respect to the solid boundary surface. To calculate the normal vector for each particle on the open boundary, refer to *@Zhao2019 Sec 4.1*.

<div align="center">
  <img width="160px" src="https://github.com/Rabmelon/tiSPHi/raw/master/docs/img/Boundary_normal.png">
</div>

### A "soft" repulsive force

> @Chalk2020, Liu2012

This is a coupled dynamic solid boundary treatment. The repulsive particles (triangle) are set to apply the no-slip effect and always guarantee that the particles do not penetrate the wall boundary. They can apply a soft repulsive force to the particles near the wall boundary, which is incorporated as a body force in the momentum equation. The definition of the repulsive force is introduced that prevents particle penetration without obviously disturbing the interior particels. The force $\hat{\boldsymbol{F}}_{ij}$ is applied to all particles that **interact** with the repulsive boundary particles, and is included in the SPH momentum equation:

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

And this soft repulsive force was combined with dummy particles and applied to simulations of water flow and the propagation of a Bingham material.

## Time integration and advection

### Courant-Friedrichs-Lewy (CFL)

> @Bui2021 3.6, Yang2021 2.5 and Koschier2019 2.9

The CFL condition is a necessary condition for the convergence of numerical solvers for differential equations and, as a result, provides an upper bound for the time step width.

The size of $\Delta t$ is determined using the Courant-Friedrichs-Lewy (CFL) stability condition, which, for SPH states that:

$$\Delta t=C_{CFL}\frac{h}{\Vert \boldsymbol{v}^{max}\Vert}$$

where a suitable constant value for $C_{CFL}$ was found to be 0.2 from Yang2021, 0.1 from Bui2021, 0.4 from Koschier2019.
$h$ is the smoothing length and $\boldsymbol{v}^{max}$ is the velocity at which the fastest particle travels, which can be the speed of sound $c$ of the material with $c=\sqrt{E/\rho}$.

### Symp Euler - Symplectic Euler (SE)

> @taichiCourse01-10 PPT p77

Also referred to as semi-implicit Euler or Euler-Cromer scheme.

$$v_i^* = v_i+\Delta t\frac{{\rm d}v_i}{{\rm d}t},\ \ x_i^* = x_i+\Delta tv_i^*$$

### Leap-Frog (LF)

> @yang2021

Leap-Frog(LF) time-integration scheme is sufficiently stable, accurate, and fast due to only requiring one calculation of forces for each timestep. For a given time-step, the density and velocity are brought forward to the mid-increment using material derivatives from the previous timestep(if available), and the position is updated at full-increments:

$$f_{n+\frac{1}{2}}=f_n+\frac{\Delta t}{2}(\frac{{\rm D}f}{{\rm D}t})_{n-1}$$

$f$ is density or velocity or stress. Then calculate $(\frac{{\rm D}f}{{\rm D}t})_{n}$

$$f_{n+1}=f_n+\Delta t(\frac{{\rm D}f}{{\rm D}t})_{n}$$

$$\boldsymbol{x}_{n+1}=\boldsymbol{x}_n+\Delta t\times\boldsymbol{u}_{n+1}$$



### RK4 - 4th order Runge-Kutta (RK4)

> @Chalk2020 Appendix B.

The RK4 scheme has fourth order accuracy and relatively simple implementation.
Consider a general ordinary differential equation for a variable $\phi$ with an initial condition $\phi^0$ at an initial time $t^0$:

$$\dot{\phi} = f(t, \phi),\ \phi(t^0) = \phi^0$$

where $f$ is a function of $\phi$ and time $t$. The RK4 method is employed to increment $\phi$ by a time step $\Delta t$ to obtain the solution at time $t = t+\Delta t$:

$$\phi^{t+\Delta t}=\phi^t+\frac{\Delta t}{6}(k_1+2k_2+2k_3+k_4)$$

$$k_1=f(\phi_1),\ k_2=f(\phi_2),\ k_3=f(\phi_3),\ k_4=f(\phi_4)$$

$$\phi_1=\phi^t,\ \phi_2=\phi^t+\frac{\Delta t}{2}k_1,\ \phi_3=\phi^t+\frac{\Delta t}{2}k_2,\ \phi_4=\phi^t+\Delta tk_3$$

### XSPH

In addition to the velocity and stress, the position vectors of each particle $\boldsymbol{x}_i$ are updated via the XSPH method at the end of each time step as:

$$\frac{{\rm d} \boldsymbol{x}_i}{{\rm d} t} = \boldsymbol{v}_i + \epsilon_x\sum_j\frac{m_j}{\rho_j}(\boldsymbol{v}_j - \boldsymbol{v}_i)\nabla W_{ij}$$

Alternatively, the discretised XSPH equation is:

$$\boldsymbol{x}_i^{t+\Delta t} = \boldsymbol{x}_i^t + \Delta t\frac{{\rm d} \boldsymbol{x}_i}{{\rm d} t} = \boldsymbol{x}_i^t + \Delta t(\boldsymbol{v}_i^{t+\Delta t} + \epsilon_x\sum_j\frac{m_j}{\rho_j}(\boldsymbol{v}_j - \boldsymbol{v}_i)\nabla W_{ij})$$

where $\epsilon_x$ is a tuning para, $0\leq\epsilon_x\leq1$.

While, in standard SPH, the simplest way is:

$$\frac{{\rm d} \boldsymbol{x}_i}{{\rm d} t} = \boldsymbol{v}_i$$

And for the particle position update (Leap-Frog):

$$\boldsymbol{x}_i^{t+\Delta t} = \boldsymbol{x}_i^t + {\Delta t}\boldsymbol{v}_i^{t+\frac{\Delta t}{2}}\ and\ \boldsymbol{v}_i^{t+\frac{\Delta t}{2}} = \frac{1}{2}(\boldsymbol{v}_i^{t+\Delta t}+\boldsymbol{v}_i^t)$$

or just Symplectic Euler:

$$\boldsymbol{x}_i^{t+\Delta t} = \boldsymbol{x}_i^t + {\Delta t}\boldsymbol{v}_i^{t+\Delta t}$$

## Numerical oscillations and dissipations in SPH

### Artificial viscosity - standard approach

> @bui2021 3.3, @chalk2020 4.5.1, @nguyen2017, @Adami2012, from @Monaghan1983

The fully dynamic equation would cause SPH particles to freely oscillate due to even small unbalanced forces, most of which is attributed to the zero-energy mode produced by the anti-symmetric kernel function with zero kernel gradient at the inflection point. However, this oscillation of SPH particles or material points is a common issue associated with any numerical method used to solve the fully dynamic motion equation.

An adapted artificial viscosity was implemented with SPH to dampen the irregular particle motion and pressure fluctuations, and to prevent the non-physical collisions of two approaching particles. The artificial viscosity term $\Pi_{ij}$ is included in the SPH momentum equation as:

$$\frac{{\rm D}\boldsymbol{v}_i}{{\rm D}t}=\sum_jm_j(\frac{\boldsymbol{\sigma}_j}{\rho_j^2}+\frac{\boldsymbol{\sigma}_i}{\rho_i^2}+\Pi_{ij}\boldsymbol{I})\cdot\nabla_iW_{ij}+\boldsymbol{f}^{ext}_i$$

And the most widely used form of artificial viscosity is:

$$\Pi_{ij}=\begin{cases} \frac{-\alpha_{\Pi}c_{ij}\phi_{ij}+\beta_{\Pi}\phi_{ij}^2}{\rho_{ij}},&\boldsymbol{v}_{ij}\cdot\boldsymbol{x}_{ij}<0\\ 0,&\boldsymbol{v}_{ij}\cdot\boldsymbol{x}_{ij}\ge0\\ \end{cases}$$

$$\phi_{ij}=\frac{h_{ij}\boldsymbol{v}_{ij}\cdot\boldsymbol{x}_{ij}}{\Vert\boldsymbol{x}_{ij}\Vert^2+\varepsilon h_{ij}^2}$$

$$c_{ij}=\frac{c_i+c_j}{2},\ \rho_{ij}=\frac{\rho_i+\rho_j}{2},\ h_{ij}=\frac{h_i+h_j}{2},\ \boldsymbol{x}_{ij}=\boldsymbol{x}_i-\boldsymbol{x}_j,\ \boldsymbol{v}_{ij}=\boldsymbol{v}_i-\boldsymbol{v}_j$$

where $\alpha_{\Pi}$ and $\beta_{\Pi}$ are problem dependent tuning parameters, $c$ is the speed of sound. $\alpha_{\Pi}$ is associated with the speed of sound and is related to the linear term, while $\beta_{\Pi}$ is associated with the square of the velocity and has little effect in problems where the flow velocity is not comparable to the speed of sound. $\varepsilon=0.01$ is a numerical parameter introduced to prevent numerical divergences, only to ensure a non-zero denominator.

This artificial viscosity is applied only for interactions between material particles, i.e. no artificial dissipation is introduced for the interaction of dummy particles and real particles.

A disadvantage of using the artificial viscosity is that parameter tuning may be required to obtain the optimal values which are not directly associated with any physical properties. The use of the artificial viscosity in SPH simulations is purely for the purposes of numerical stabilisation.

### Alternative viscous damping term

> @bui2021 3.3, @chalk2020 4.5.1, @nguyen2017

Alternative damping terms can be used instead of the artificial viscosity that have more physical relevance to the problem, or require less calibration. The following velocity-dependent damping term can be included as a body force in the equation of the momentum:

$$\boldsymbol{F}_d=-\mu_d\boldsymbol{v}$$

$\mu_d$ is the damping factor which can be computed by $\mu_d=\xi\sqrt{E/\rho h^2}$ with $\xi$ being a non-dimensional damping coefficient that requires calibrations for different applications. For the simulation of granular flows, such as the flow of granular column collapse experiments in *Nguyen2017*, a constant value of $\xi=5\times10^{-5}$ is recommended.

### Stress/strain regularisation

> @bui2021 3.3, @nguyen2017

While the kinematics of SPH simulation is generally realistic, the stress-pressure fields of SPH particles undergoing large deformation can exhibit large oscillations. This problem is known as the sort-length-scale-noise and is identified as one of the key challenges of the standard SPH method tha needs to be addressed in order to improve the accuracy of SPH simulations.

The problem becomes worse when the artificial viscosity is not adopted in SPH simulations, although the viscous damping force could slow down the numerical instability process.

*Nguyen2017* suggests regularising the stresses and strains of each SPH particle over its kernel integral domain after a certain number of computational cycles and uses MLS method:

$$\langle\boldsymbol{\sigma}_{i}\rangle=\sum_jV_j\boldsymbol{\sigma}_{j}W^{MLS}_{ij}$$

$$\langle\boldsymbol{\epsilon}_{i}\rangle=\sum_jV_j\boldsymbol{\epsilon}_{j}W^{MLS}_{ij}$$

And *Nguyen2017* suggestes applying the above MLS correction every 5 steps.

## Tensile instability
