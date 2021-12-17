**Smoothed Particle Hydrodynamics Learning Notes**

<!-- @import "[TOC]" {cmd="toc" depthFrom=2 depthTo=5 orderedList=false} -->

<!-- code_chunk_output -->

- [Basic mathematics](#basic-mathematics)
  - [The spatial derivative operators in 3D](#the-spatial-derivative-operators-in-3d)
  - [Material derivative](#material-derivative)
- [Governing equations and constitutive model of soil](#governing-equations-and-constitutive-model-of-soil)
  - [Conservation of mass](#conservation-of-mass)
  - [Conservation of momentum](#conservation-of-momentum)
  - [Constitutive model](#constitutive-model)
- [Standard SPH](#standard-sph)
  - [SPH basic fomulations](#sph-basic-fomulations)
  - [Improving approximations for spatial derivatives:](#improving-approximations-for-spatial-derivatives)
  - [Forces for incompressible fluids](#forces-for-incompressible-fluids)
  - [Incompressible Navier-Stokes equation](#incompressible-navier-stokes-equation)
  - [Temporal discretization](#temporal-discretization)
  - [Full time integration](#full-time-integration)
  - [The weakly compressible assumption](#the-weakly-compressible-assumption)
  - [Fluid dynamics with particles (weakly compressible)](#fluid-dynamics-with-particles-weakly-compressible)
  - [Boundary conditions](#boundary-conditions)
- [Complex boundary treatment](#complex-boundary-treatment)
- [SPH improvement techniques](#sph-improvement-techniques)
- [Nearest neighbouring search](#nearest-neighbouring-search)
- [Time discretisation](#time-discretisation)
- [Stress-Particle SPH](#stress-particle-sph)

<!-- /code_chunk_output -->

For learning how SPH works in slope failure and post-failure process, also in the landslide and furthermore debris-flows through obstacles. [@buiLagrangianMeshfreeParticles2008; @chalkNumericalModellingLandslide2019; @chalkStressParticleSmoothedParticle2020; @taichiCourse01; ]


## Basic mathematics

### The spatial derivative operators in 3D
$\nabla$ 算子的三个语义:
**梯度Gradient**：作用于**标量**$f(x, y, z)$得到**矢量**。$\mathbb{R}^1\rightarrow\mathbb{R}^3, \nabla$
$$grad\ f=\nabla f=(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z})$$
**散度Divergence**：作用于**矢量**$(f_x, f_y, f_z)$得到**标量**。$\mathbb{R}^3\rightarrow\mathbb{R}^1, \nabla\cdot$
$$div\ \boldsymbol{f}=\nabla\cdot \boldsymbol{f}=\frac{\partial f_x}{\partial x} + \frac{\partial f_y}{\partial y} + \frac{\partial f_z}{\partial z}$$
**旋度Curl**：作用于**矢量**$(f_x, f_y, f_z)$得到**矢量**。$\mathbb{R}^3\rightarrow\mathbb{R}^3, \nabla\times$
$$curl\ \boldsymbol{f}=\nabla\times\boldsymbol{f}=\begin{vmatrix} \boldsymbol{i} &\boldsymbol{j} &\boldsymbol{k}\\ \frac{\partial}{\partial x} &\frac{\partial}{\partial y} &\frac{\partial}{\partial z}\\ f_x &f_y &f_z \end{vmatrix}=(\frac{\partial f_z}{\partial y}-\frac{\partial f_y}{\partial z}, \frac{\partial f_x}{\partial z}-\frac{\partial f_z}{\partial x}, \frac{\partial f_y}{\partial x}-\frac{\partial f_x}{\partial y})$$
**拉普拉斯Laplace**: 梯度的散度，作用于任意维度的变量。 $\mathbb{R}^n\rightarrow\mathbb{R}^n, \nabla \cdot \nabla=\nabla^2$
$$laplace\ f=div(grad\ f)=\nabla^2f=\frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} + \frac{\partial^2 f}{\partial z^2}$$

### Material derivative
$\frac{{\rm D}f}{{\rm D}t}=\frac{\partial f}{\partial t}+\boldsymbol{u}\cdot\nabla f$ is **material derivative** in fluid mechanics, total derivative in math. (数学上的全导数，流体力学中的物质导数、随体导数，为流体质点在运动时所具有的物理量对时间的全导数。
运动的流体微团的物理量随时间的变化率，它等于该物理量由当地时间变化所引起的变化率与由流体对流引起的变化率的和。
从偏导数全微分的摡念出发，密度变化可以认为是密度分布函数（密度场）的时间偏导数项（不定常）和空间偏导数项（空间不均匀）的和。时间偏导项叫局部导数或就地导数。空间偏导项叫位变导数或对流导数。
中科院的李新亮研究员给出了一个更加形象的例子：高铁的电子显示屏上会实时显示车外的温度，如果我们将高铁看作是一个流体微元，它早上从北京出发，中午到达上海，显示屏上记录的室外温度的变化就是物质导数，它包含了两个部分，一是从北京到上海的地理位置的变化所带来的温度变化，即对流导数；二是由于早上到中午由于时间不同而引起的温度变化，即当地导数。)
The final form in Lagrangian method: (等号左侧，第一项为微团密度的变化，第二项为微团体积的变化。)
$$\frac{{\rm D}\rho}{{\rm D}t}+\rho\nabla\cdot\boldsymbol{u}=0$$
对于不可压缩流动，质点的密度在运动过程中保持不变，故$\frac{{\rm D}\rho}{{\rm D}t}=0$

## Governing equations and constitutive model of soil
Conservation of mass:
$$\frac{{\rm D} \rho}{{\rm D} t}=-\rho \nabla\cdot\boldsymbol{u}$$
Conservation of momentum:
$$\frac{{\rm D} \boldsymbol{u}}{{\rm D} t}=\frac{1}{\rho} \nabla\cdot\boldsymbol{f}^{\sigma}+\boldsymbol{b}$$
Constitutive equation:
$$\frac{{\rm D} \boldsymbol{\sigma}}{{\rm D} t}=\boldsymbol{\widetilde{\sigma}} +\nabla\cdot\boldsymbol{f}^u-\boldsymbol{g}^{\varepsilon^p}$$
where:
$$\begin{aligned}
\boldsymbol{x}=
  \left (\begin{array}{c}
    x\\y
  \end{array}\right)
\end{aligned}
,
\begin{aligned}
\boldsymbol{u}=
  \left (\begin{array}{c}
    u_x\\u_y
  \end{array}\right)
\end{aligned}
,
\begin{aligned}
\boldsymbol{f}^{\sigma}=
  \left (\begin{array}{cc}
    \sigma_{xx}    &\sigma_{xy}\\    \sigma_{xy}    &\sigma_{yy}
  \end{array}\right)
\end{aligned}
,
\begin{aligned}
\boldsymbol{b}=
  \left (\begin{array}{c}
    b_x\\b_y
  \end{array}\right)
\end{aligned}$$

$$\begin{aligned}
\boldsymbol{\sigma}=
  \left (\begin{array}{c}
    \sigma_{xx}\\
    \sigma_{yy}\\
    \sigma_{xy}\\
    \sigma_{zz}
  \end{array} \right)
\end{aligned}
,
\begin{aligned}
  \boldsymbol{\widetilde{\sigma}}=
    \left(\begin{array}{c}
      2\sigma_{xy}\omega_{xy}\\
      2\sigma_{xy}\omega_{yx}\\
      \sigma_{xx}\omega_{yx}+\sigma_{yy}\omega_{xy}\\
      0
    \end{array} \right)
\end{aligned}$$

$$\dot\omega_{\alpha\beta}=\frac{1}{2}(\frac{\partial u_{\alpha}}{\partial x_{\beta}}-\frac{\partial u_{\beta}}{\partial x_{\alpha}})$$

$$\begin{aligned}
\boldsymbol{f}^u=
  \left (\begin{array}{cc}
    D^e_{11}u_x    &D^e_{12}u_y\\
    D^e_{12}u_x    &D^e_{22}u_y\\
    D^e_{33}u_y    &D^e_{33}u_x\\
    D^e_{41}u_x    &D^e_{42}u_y
  \end{array}\right)
\end{aligned}
,
\begin{aligned}
  \boldsymbol{g}^{\varepsilon^p}=
    \left(\begin{array}{c}
      g^{\varepsilon^p}_{xx}(\boldsymbol{\dot \varepsilon}^p)\\
      g^{\varepsilon^p}_{yy}(\boldsymbol{\dot \varepsilon}^p)\\
      g^{\varepsilon^p}_{xy}(\boldsymbol{\dot \varepsilon}^p)\\
      g^{\varepsilon^p}_{zz}(\boldsymbol{\dot \varepsilon}^p)
    \end{array} \right)
\end{aligned}
,
\begin{aligned}
  \boldsymbol{\dot \varepsilon}^p=
    \left(\begin{array}{c}
      \dot \varepsilon^p_{xx}\\
      \dot \varepsilon^p_{yy}\\
      \dot \varepsilon^p_{xy}\\
      0
    \end{array} \right)
\end{aligned}$$

$$\begin{aligned}
D^e_{pq}=\frac{E}{(1+\nu)(1-\nu)}
  \left (\begin{array}{cccc}
    1-\nu  &\nu  &0  &\nu\\
    \nu  &1-\nu  &0  &\nu\\
    0  &0  &(1-2\nu)/2  &0\\
    \nu  &\nu  &0  &1-\nu\\
  \end{array}\right)
\end{aligned}$$

### Conservation of mass
The loss of mass equals to the net outflow: (控制体内质量的减少=净流出量)
$$-\frac{\partial m}{\partial t} = -\frac{\partial \rho}{\partial t}{\rm d}x{\rm d}y{\rm d}z=[\frac{\partial (\rho u_x)}{\partial x}+\frac{\partial (\rho u_y)}{\partial y}+\frac{\partial (\rho u_z)}{\partial z}]{\rm d}x{\rm d}y{\rm d}z$$
$$\frac{\partial \rho}{\partial t}+\nabla\cdot(\rho \boldsymbol{u})=0, ~ \nabla=\boldsymbol{i}\frac{\partial}{\partial x}+\boldsymbol{j}\frac{\partial}{\partial y}+\boldsymbol{k}\frac{\partial}{\partial z}$$
$$\frac{\partial \rho}{\partial t}+\boldsymbol{u}\cdot\nabla\rho+\rho\nabla\cdot\boldsymbol{u}=0, ~ \frac{{\rm D}\rho}{{\rm D}t}=\frac{\partial \rho}{\partial t}+\boldsymbol{u}\cdot\nabla\rho$$


### Conservation of momentum
根据牛顿流体的本构方程，推导获得流体的动量方程。
无粘流动的动量方程即欧拉方程（惯性力$\frac{{\rm D}\boldsymbol{u}}{{\rm D}t}$，体积力$\boldsymbol{f}$，压差力$-\frac{1}{\rho}\nabla p$，普通动量方程中的粘性力$\frac{\mu}{\rho}\nabla^2\boldsymbol{u}+\frac{1}{3}\frac{\mu}{\rho}\nabla(\nabla\cdot\boldsymbol{u})=0$）：
$$\frac{{\rm D}\boldsymbol{u}}{{\rm D}t}=\boldsymbol{f}-\frac{1}{\rho}\nabla p$$
流体静止时，粘性力项自然为0，惯性力项也为0，即退化为欧拉静平衡方程$\nabla p=\rho\boldsymbol{f}$。

> **QUESTIONS**
> 1. The momentum considered here is not the same as Navier-Stokes equation but what???


### Constitutive model
Constitutive model is to relate the soil stresses to the strain rates in the plane strain condition.
For Drucker-Prager yield criteria: $f=\sqrt{J_2}+\alpha_{\varphi}I_1-k_c=0$ and functions of the Coulomb material constants - the soil internal friction $\varphi$ and cohesion $c$:
$$\alpha_{\varphi}=\frac{\tan\varphi}{\sqrt{9+12\tan^2\varphi}}, k_c=\frac{3c}{\sqrt{9+12\tan^2\varphi}}$$
And for the elastoplastic constitutive equation of Drucker-Prager and *non-associated flow rule*, $g=\sqrt{J_2}+3I_1\cdot\sin\psi$, where $\psi$ is dilatancy angle and in Chalk's thesis$\psi=0$. Of *associated flow rule*, $g=\sqrt{J_2}+\alpha_{\varphi}I_1-k_c$.

> **QUESTIONS**
> 1. How does $\boldsymbol{g}^{\varepsilon^p}$ and $\boldsymbol{\dot\varepsilon}^p$ calculated? Maybe it is different in elastoplastic and Perzyna models.
> 2. How does $\dot\omega_{\alpha\beta}$ calculated? Is it equal to $\omega_{\alpha\beta}$ in $\boldsymbol{\widetilde{\sigma}}$ ?

## Standard SPH

### SPH basic fomulations
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

### Improving approximations for spatial derivatives:
> taichiCourse01-10 PPT p60-70

* Let $f(r) \equiv 1$, we have:
  * $1 \approx \sum_j \frac{m_j}{\rho_j}W(r-r_j, h)$
  * $0 \approx \sum_j \frac{m_j}{\rho_j}\nabla W(r-r_j, h)$
* Since $f(r)\equiv f(r) * 1$, we have:
  * $\nabla f(r) = \nabla f(r)*1+f(r)*\nabla 1$
  * Or equivalently: $\nabla f(r) = \nabla f(r)*1-f(r)*\nabla 1$
* Then use ${\color{Salmon} \nabla} f(r) \approx \sum_j \frac{m_j}{\rho_j}f(r_j){\color{Salmon} \nabla}W(r-r_j, h)$ to derivate $\nabla f(r)$ and $\nabla 1$, we have:
  * $\nabla f(r) \approx \sum_j \frac{m_j}{\rho_j}f(r_j)\nabla W(r-r_j, h) - f(r)\sum_j \frac{m_j}{\rho_j}\nabla W(r-r_j, h)$
  * $\nabla f(r) \approx \sum_j m_j\frac{f(r_j)-f(r)}{\rho_j}\nabla W(r-r_j, h)$, we call it the **anti-symmetric form**
* A more general case:
  $$\nabla f(r) \approx \sum_j m_j(\frac{f(r_j)\rho_j^{n-1}}{\rho^n}-\frac{nf(r)}{\rho})\nabla W(r-r_j, h)$$
  * When $n=-1$: $\nabla f(r) \approx \rho\sum_j m_j(\frac{f(r_j)}{\rho_j^2}+\frac{f(r)}{\rho^2})\nabla W(r-r_j, h)$, we call it the **symmetric form**
* 通常会使用一些反对称(**anti-sym**)或对称型(**sym**)来进行一些SPH的空间求导(spatial derivative)，而不直接使用SPH的原型。但两者的选择是个经验性的问题，其中，当$f(r)$是一个力的时候，从动量守恒的角度去推导，使用**sym**更好；当做散度、需要投影的时候，使用**anti-sym**更好。


### Forces for incompressible fluids
> taichiCourse01-10 PPT p8-13

$$f = ma = {\color{Green} f_{ext}} + {\color{RoyalBlue} f_{pres}} + {\color{Orange} f_{visc}}$$
### Incompressible Navier-Stokes equation
> taichiCourse01-10 PPT p16-28

The momentum equation
$$\rho\frac{{\rm D}v}{{\rm D}t}={\color{Green} \rho g} {\color{RoyalBlue} -\nabla p} + {\color{Orange} \mu\nabla^2v}$$
The mass conserving condition
$${\color{RoyalBlue} \nabla\cdot v=0} $$
$\rho\frac{{\rm D}v}{{\rm D}t}$: This is simply "mass" times "acceleration" divided by "volume".
${\color{Green} \rho g}$: External force term, gravitational force divided by "volume".
${\color{Orange} \mu\nabla^2v}$: Viscosity term, how fluids want to move together. 表示扩散有多快，液体尽可能地往相同的方向运动。$\mu$: some fluids are more viscous than others.
${\color{RoyalBlue} -\nabla p}$: Pressure term, fluids do not want to change volume. $p=k(\rho-\rho_0)$ but $\rho$ is unknown.
${\color{RoyalBlue} \nabla\cdot v=0 \Leftrightarrow \frac{{\rm D} \rho}{{\rm D} t} = \rho(\nabla\cdot v) = 0}$: Divergence free condition. Outbound flow equals to inbound flow. The mass conserving condition. 散度归零条件、不可压缩特性，也是质量守恒条件。

### Temporal discretization
> taichiCourse01-10 PPT p32

Integrate the incompressible N-S equation in steps (also reffered as "Operator splitting" or "Advection-Projection" in different contexts):
* Step 1: input $v_n$, output $v_{n+0.5}$: $\rho\frac{{\rm D} v}{{\rm D} t}={\color{Green} \rho g} + {\color{Orange} \mu\nabla^2v}$
* Step 2: input $v_{n+0.5}$, output$v_{n+1}$: $\rho\frac{{\rm D} v}{{\rm D} t}={\color{RoyalBlue} -\nabla p}\ and\ {\color{RoyalBlue} \nabla\cdot v=0}$ (构成了$\rho$和$v$的二元非线性方程组)

### Full time integration
> taichiCourse01-10 PPT p33

$$\frac{{\rm D}v}{{\rm D}t}={\color{Green} g} {\color{RoyalBlue} -\frac{1}{\rho}\nabla p} + {\color{Orange} \nu\nabla^2v},\ \nu=\frac{\mu}{\rho_0}$$
* Given $x_n$, $v_n$:
* Step 1: Advection / external and viscosity force integration
  * Solve: ${\color{Purple} dv} = {\color{Green} g} + {\color{Orange} \nu\nabla^2v_n}$
  * Update: $v_{n+0.5} = v_n+\Delta t{\color{Purple} dv}$
* Step 2: Projection / pressure solver
  * Solve: ${\color{red} dv} = {\color{RoyalBlue} -\frac{1}{\rho}\nabla(k(\rho-\rho_0))}$ and ${\color{RoyalBlue} \frac{{\rm D} \rho}{{\rm D} t} = \nabla\cdot(v_{n+0.5}+{\color{red} dv})=0}$
  * Update: $v_{n+1} = v_{n+0.5} + \Delta t {\color{red} dv}$
* Step 3: Update position
  * Update: $x_{n+1} = x_n+\Delta tv_{n+1}$
* Return $x_{n+1}$, $v_{n+1}$
> **QUESTIONS**
> 1. In step 1 and 2, maybe the $\Delta t$ should also multiple 0.5?

### The weakly compressible assumption
> taichiCourse01-10 PPT p34-35

Storing the density $\rho$ as an individual variable that advect with the velocity field. Then the $p$ can be assumed as a variable related by time and the mass conserving equation is killed.

* Change in Step 2:
  * Solve: ${\color{red} dv} = {\color{RoyalBlue} -\frac{1}{\rho}\nabla(k(\rho-\rho_0))}$
  * Update: $v_{n+1} = v_{n+0.5} + \Delta t {\color{red} dv}$
And step 2 and 1 can be merged. This is nothing but Symplectic Euler integration.

### Fluid dynamics with particles (weakly compressible)
> taichiCourse01-10 PPT p43 and 75-78

Continuous view:
$$\frac{{\rm D}v}{{\rm D}t}={\color{Green} g} {\color{RoyalBlue} -\frac{1}{\rho}\nabla p} + {\color{Orange} \nu\nabla^2v}$$
Discrete view (using particle):
$$\frac{{\rm d}v_i}{{\rm d}t}=a_i={\color{Green} g} {\color{RoyalBlue} -\frac{1}{\rho}\nabla p(x_i)} + {\color{Orange} \nu\nabla^2v(x_i)}$$

Then the problem comes to: how to evaluate a funcion of ${\color{RoyalBlue} \rho(x_i)}$, ${\color{RoyalBlue} \nabla p(x_i)}$, ${\color{Orange} \nabla^2v(x_i)}$

In WCSPH:
* Find a particle of interest ($i$) and its nerghbours ($j$) with its support radius $h$.
* Compute the acceleration for particle $i$:
  * for i in particles:
    * Step 1: Evaluate density
      $$\rho_i = \sum_j \frac{m_j}{\rho_j}\rho_jW(r_i-r_j, h) = \sum_j m_jW_{ij}$$
    * Step 2: Evaluate viscosity (**anti-sym**)
      $$\nu\nabla^2v_i = \nu\sum_j m_j \frac{v_j-v_i}{\rho_j}\nabla^2W_{ij}$$
      in taichiWCSPH code it's a approximation from @monaghan2005 :
      $$\nu\nabla^2v_i = 2\nu(dimension+2)\sum_j \frac{m_j}{\rho_j}(\frac{v_{ij}\cdot r_{ij}}{\|r_{ij}\|^2+0.01h^2})\nabla W_{ij}$$
    * Evaluate pressure gradient (**sym**), where $p = k(\rho-\rho_0)$
      $$-\frac{1}{\rho_i}\nabla p_i = -\frac{\rho_i}{\rho_i}\sum_j m_j(\frac{p_j}{\rho_j^2}+\frac{p_i}{\rho_i^2})\nabla W_{ij} = -\sum_j m_j(\frac{p_j}{\rho_j^2}+\frac{p_i}{\rho_i^2})\nabla W_{ij}$$
      in taichiWCSPH code, $p = k_1((\rho/\rho_0)^{k_2}-1)$, where $k_1$ is a para about stiffness and $k_2$ is just an exponent.
    * Calculate the acceleration
    * Then do time integration using Symplectic Euler method:
      $$v_{i+1} = v_i+\Delta t*\frac{{\rm d}v_i}{{\rm d}t},\ \ x_{i+1} = x_i+\Delta t*v_{i+1}$$

### Boundary conditions
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


## Complex boundary treatment

## SPH improvement techniques

## Nearest neighbouring search

## Time discretisation

## Stress-Particle SPH



# Test MD
这是一段测试文字

**行内公式**
这是$\lambda=\frac{\alpha}{\beta}$一段测试文字

**单行公式**
这是一段测试文字
$\lambda=\frac{\alpha\times\beta}{\Gamma}$
$$\Gamma = \sqrt{\int{x^{\varphi}\,{\rm D}x}}$$
$$\boldsymbol{a}\cdot\boldsymbol{b}=0$$

$$ a^2+b^2=c_0^2 \tag{1.2}$$
这是一段$(1.2)$测试(1.2)文字


$$\begin{align}
\sqrt{37} & = \sqrt{\frac{73^2-1}{12^2}} \\
 & = \sqrt{\frac{73^2}{12^2}\cdot\frac{73^2-1}{73^2}} \\
 & = \sqrt{\frac{73^2}{12^2}}\sqrt{\frac{73^2-1}{73^2}} \\
 & = \frac{73}{12}\sqrt{1 - \frac{1}{73^2}} \\
 & \approx \frac{73}{12}\left(1 - \frac{1}{2\cdot73^2}\right)
\end{align}$$

这是一行测试文字
$$\begin{equation}
\lambda = 1\\
\end{equation}$$

**图片**
<div align="center">
  <img width="200px"  src="../img/tiSPHi_logo.jpg">
</div>

**链接**
这是一段测试文字<zhibinlei@outlook.com>
[baidu](www.baidu.com "链接测试显示文字")

**表格**
这是一段测试文字


**代码段**
这是一段测试文字


**行内代码**
这是一段测试文字
