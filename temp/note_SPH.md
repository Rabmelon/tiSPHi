---
html:
  toc: true
---
**Smoothed Particle Hydrodynamics Learning Notes**

<!-- @import "[TOC]" {cmd="toc" depthFrom=2 depthTo=5 orderedList=false} -->

<!-- code_chunk_output -->

- [Basic mathematics](#basic-mathematics)
  - [The spatial derivative operators in 3D](#the-spatial-derivative-operators-in-3d)
  - [Material derivative](#material-derivative)
- [Governing equations and constitutive model of soil](#governing-equations-and-constitutive-model-of-soil)
  - [Constitutive model](#constitutive-model)
  - [Conservation of mass](#conservation-of-mass)
  - [Conservation of momentum](#conservation-of-momentum)
  - [Discretization](#discretization)
- [Standard SPH](#standard-sph)
  - [SPH basic fomulations](#sph-basic-fomulations)
  - [Improving approximations for spatial derivatives:](#improving-approximations-for-spatial-derivatives)
- [WCSPH](#wcsph)
  - [Forces for incompressible fluids](#forces-for-incompressible-fluids)
  - [Incompressible Navier-Stokes equation](#incompressible-navier-stokes-equation)
  - [Temporal discretization](#temporal-discretization)
  - [Full time integration](#full-time-integration)
  - [The weakly compressible assumption](#the-weakly-compressible-assumption)
  - [Fluid dynamics with particles (weakly compressible)](#fluid-dynamics-with-particles-weakly-compressible)
  - [Boundary conditions](#boundary-conditions)
- [Complex boundary treatment](#complex-boundary-treatment)
  - [For straight, stationary walls](#for-straight-stationary-walls)
  - [For free surface problems](#for-free-surface-problems)
- [SPH improvement techniques](#sph-improvement-techniques)
- [Nearest neighbouring search](#nearest-neighbouring-search)
- [Time discretisation](#time-discretisation)
  - [RK4 time integration](#rk4-time-integration)
  - [XSPH and simplest position update](#xsph-and-simplest-position-update)
  - [Steps](#steps)
- [Stress-Particle SPH](#stress-particle-sph)

<!-- /code_chunk_output -->

For learning how SPH works in slope failure and post-failure process, also in the landslide and furthermore debris-flows through obstacles. [@buiLagrangianMeshfreeParticles2008; @chalkNumericalModellingLandslide2019; @chalkStressParticleSmoothedParticle2020; @taichiCourse01; ]


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
$$\frac{{\rm D} \boldsymbol{\sigma}}{{\rm D} t}=\boldsymbol{\tilde{\sigma}} +\nabla\cdot\boldsymbol{f}^u-\boldsymbol{g}^{\varepsilon^p}$$

where:
$$\begin{aligned}
\boldsymbol{x}=
  \left (\begin{array}{c}
    x\\ y
  \end{array}\right)
\end{aligned}
,
\begin{aligned}
\boldsymbol{u}=
  \left (\begin{array}{c}
    u_x\\ u_y
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
    b_x\\ b_y
  \end{array}\right)
\end{aligned}$$

$$\begin{aligned}
\boldsymbol{\sigma}=
  \left (\begin{array}{c}
    \sigma_{xx}\\ \sigma_{yy}\\ \sigma_{xy}\\ \sigma_{zz}
  \end{array} \right)
\end{aligned}
,
\begin{aligned}
  \boldsymbol{\tilde{\sigma}}=
    \left(\begin{array}{c}
      2\sigma_{xy}\omega_{xy}\\ 2\sigma_{xy}\omega_{yx}\\
      \sigma_{xx}\omega_{yx}+\sigma_{yy}\omega_{xy}\\ 0
    \end{array} \right)
    =\left(\begin{array}{c}
      2\sigma_{xy}\omega_{xy}\\ -2\sigma_{xy}\omega_{xy}\\
      (\sigma_{yy}-\sigma_{xx})\omega_{xy}\\ 0
    \end{array} \right)
\end{aligned}$$

$$\dot\omega_{\alpha\beta}=\frac{1}{2}(\frac{\partial u_{\alpha}}{\partial x_{\beta}}-\frac{\partial u_{\beta}}{\partial x_{\alpha}})\ ,\ \omega_{xy} = \frac{1}{2}(\frac{\partial u_y}{\partial x_x}-\frac{\partial u_x}{\partial x_y})$$

$$\begin{aligned}
\boldsymbol{f}^u=
  \left (\begin{array}{cc}
    D^e_{11}u_x    &D^e_{12}u_y\\ D^e_{21}u_x    &D^e_{22}u_y\\
    D^e_{33}u_y    &D^e_{33}u_x\\ D^e_{41}u_x    &D^e_{42}u_y
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
      \dot \varepsilon^p_{xx}\\ \dot \varepsilon^p_{yy}\\
      \dot \varepsilon^p_{xy}\\ 0
    \end{array} \right)
\end{aligned}$$

$$\begin{aligned}
D^e_{pq}=\frac{E}{(1+\nu)(1-\nu)}
  \left (\begin{array}{cccc}
    1-\nu  &\nu  &0  &\nu\\ \nu  &1-\nu  &0  &\nu\\
    0  &0  &(1-2\nu)/2  &0\\ \nu  &\nu  &0  &1-\nu\\
  \end{array}\right)
\end{aligned}$$

$D^e_{pq}$ is the **elastic constitutive tensor**, $\dot{\omega}_{\alpha\beta}$ is the **spin rate tensor**.

And in soil mechanics, the soil pressure $p$ is obtained directly from the equation for **hydrostatic pressure**:
$$p = -\frac{1}{3}(\sigma_{xx}+\sigma_{yy}+\sigma_{zz})$$

We define the **elastic strains** according to the **generalised Hooke's law**:
$$\dot{\boldsymbol{\varepsilon}}^e = \frac{\dot{\boldsymbol{s}}}{2G}+\frac{1-2\nu}{3E}\dot{\sigma}_{kk}\boldsymbol{I}$$

where $\dot{\sigma}_{kk} = \dot{\sigma}_{xx}+\dot{\sigma}_{yy}+\dot{\sigma}_{zz}$, $\boldsymbol{s}$ is the **deviatoric stress tensor**: $\boldsymbol{s} = \boldsymbol{\sigma}-p\boldsymbol{I}$ and $\boldsymbol{I}$ is the identity matrix.

> **QUESTIONS**
> 1. the hydrostatic pressure $p$, is positive or negtive? $\boldsymbol{s}$ is only correct when $p$ is positive as Chalk2020's Appendix A, but in the main text of Chalk2020, $p$ is negtive.

### Constitutive model
Constitutive model is to relate the soil stresses to the strain rates in the plane strain condition.
For **Drucker-Prager** yield criteria: $f=\sqrt{J_2}+\alpha_{\varphi}I_1-k_c=0$ and functions of the Coulomb material constants - the soil internal friction $\varphi$ and cohesion $c$:
$$\alpha_{\varphi}=\frac{\tan\varphi}{\sqrt{9+12\tan^2\varphi}}, k_c=\frac{3c}{\sqrt{9+12\tan^2\varphi}}$$

And for the elastoplastic constitutive equation of Drucker-Prager and *non-associated flow rule*, $g=\sqrt{J_2}+3I_1\cdot\sin\psi$, where $\psi$ is dilatancy angle and in Chalk's thesis$\psi=0$. Of *associated flow rule*, $g=\sqrt{J_2}+\alpha_{\varphi}I_1-k_c$.
And the **Von Mises** criterion is: $f = \sqrt{3J_2}-f_c$.
The Von Mises and D-P yield criteria are illustrated in two dimensions:
<div align="center">
  <img width="400px" src=".\Yield_criterias.png">
</div>

Here we difine the firse invariant of the stress tensor $I_1$ and the second invariant of the deviatoric stress tensor $J_2$:
$$I_1 = \sigma_{xx}+\sigma_{yy}+\sigma_{zz}\ ,\ J_2 = \frac{1}{2}\boldsymbol{s}:\boldsymbol{s}$$

> **QUESTIONS**
> 1. How does $\boldsymbol{g}^{\varepsilon^p}$ and $\boldsymbol{\dot\varepsilon}^p$ calculated? Maybe it is different in elastoplastic and Perzyna models.
> 2. How does the operator : calculated? ----**Answer**: double dot product of tensors, also a double tensorial contraction. The double dots operator "eats" a tensor and "spits out" a scalar.


The fundamental assumption of plasticity is that the total soil strain rate $\boldsymbol{\dot\varepsilon}$ can be divided into an elastic and a plastic component:
$$\boldsymbol{\dot\varepsilon} = \boldsymbol{\dot\varepsilon}^e+\boldsymbol{\dot\varepsilon}^p$$

With an assumption of a kinematic condition between the *total strain rate* and the *velocity gradients*.
$$\dot{\varepsilon}_{\alpha\beta} = \frac{1}{2}(\frac{\partial u_{\alpha}}{\partial x_{\beta}}+\frac{\partial u_{\beta}}{\partial x_{\alpha}})$$

Consider both a **Von Mises** and a **D-P** yield criterion to distinguish between elastic and plastic material behaviour.

In the elastoplastic model, the stress state is not allowed to exceed the yield surface and I should apply a stress adaptation to particles, after every calculation step. And the elastic and plastic behaviour are distinguished via a stress-dependent yield criterion.


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

### Discretization
> @chalk2020 Section 3.1

The discrete governing equations of soil motion in the framework of standard SPH are therefore:
$$\frac{{\rm D} \rho_i}{{\rm D} t} = -\sum_j m_j(\boldsymbol{u}_j-\boldsymbol{u}_i)\nabla W_{ij}$$

$$\frac{{\rm D} \boldsymbol{u}_i}{{\rm D} t} = \sum_j m_j(\frac{\boldsymbol{f}_i^{\sigma}}{\rho_i^2}+\frac{\boldsymbol{f}_j^{\sigma}}{\rho_j^2})\nabla W_{ij}+\boldsymbol{b}_i$$

$$\frac{{\rm D} \boldsymbol{\sigma}_i}{{\rm D} t} = \boldsymbol{\tilde{\sigma}}_i+\sum_j \frac{m_j}{\rho_j}(\boldsymbol{f}_j^u-\boldsymbol{f}_i^u)\nabla W_{ij}-\boldsymbol{g}_i^{\varepsilon^p}$$

In the current work, each SPH particle is assigned the same, constant density for the duration of the simulation. We treat the soil as incompressible and consequently do not update density through this way.


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


## WCSPH
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
> @Chalk2020

虚拟的边界粒子，本身不具有具体的属性数值。在每一个Step中，在每一个粒子的计算中，先加入一个对Dummy particle对应属性的赋值。

### For straight, stationary walls
First, choose the method to solve boundary problems. I want to update the behaviour of particles without just invert the operator but with some rules that are suitable for soil dynamics problems.

The dummy particle method is used to represent the wall boundary. For dummy and repulsive particles at the wall boundary, they are spaced apart by $\Delta x/2$. For other dummy particles, are $\Delta x$.
<div align="center">
  <img width="300px" src=".\Dummy_particles.png">
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

For an interior particle A (circle) that contains a dummy particle B (square and triangle) within its neighbourhood, the normal distances $d_A$ and $d_B$ to the wall are calculated. An artificial velocity $\boldsymbol{u}_B$ is then assigned to the dummy particle:
$$\boldsymbol{u}_B = -\frac{d_B}{d_A}\boldsymbol{u}_A$$

To account for extremely large values of the dummy particle velocity when an interior particle approaches the boundary (and $d_A$ approaches 0), a parameter $\beta$ is introduced:
$$\boldsymbol{u}_B = (1-\beta)\boldsymbol{u}_A+\beta\boldsymbol{u}_{wall}\ ,\ \beta = min(\beta_{max}, 1+\frac{d_B}{d_A})$$

$\beta_{max}$ have been found to be between $1.5\rightarrow2$, and here we use $\beta_{max}=1.5$.

And we have $\boldsymbol{\sigma}_B=\boldsymbol{\sigma}_A$. The simple definition ensures that there is a uniform stress distribution for the particles that are near the wall boundaries, and it contributes to smooth stress distributions (through the $\boldsymbol{f}^{\sigma}$ term) on the interior particles in the equation of momentum through the particle-dummy interaction.

> **QUESTIONS**
> 1. How about the mass of repulsive particles?

### For free surface problems
The particles that comprise the free surface should satisfy a stress-free condition. When considering large deformations this first requires the detection of free surface particles, followed by a transformation of the stress tensor so that the normal and tangential components are 0.

> **QUESTIONS**
> BUT how does the free surface condition implement?


## SPH improvement techniques

## Nearest neighbouring search

## Time discretisation
> @Chalk2020, Appendix B.

### RK4 time integration
The considered governing SPH equations are summarised as:
$$\frac{{\rm D} \boldsymbol{u}_i}{{\rm D} t} = \sum_j m_j(\frac{\boldsymbol{f}_i^{\sigma}}{\rho_i^2}+\frac{\boldsymbol{f}_j^{\sigma}}{\rho_j^2})\cdot\nabla W_{ij}+\boldsymbol{b}_i = F_1(\boldsymbol{\sigma}_i)$$

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
      \boldsymbol{u}^4_i = \boldsymbol{u}^t_i+\frac{\Delta t}{2}(F_1(\boldsymbol{\sigma}^3_i)) &\boldsymbol{\sigma}^4_i = \boldsymbol{\sigma}^t_i+\frac{\Delta t}{2}(F_2(\boldsymbol{u}^3_i, \boldsymbol{\sigma}^3_i))
    \end{array}
\end{aligned}$$

In standard SPH, these eight eqs are spatially resolved at each calculation step by calculating $\boldsymbol{u}_i^{t+\Delta t}$ and $\boldsymbol{\sigma}_i^{t+\Delta t}$ at each particle.

### XSPH and simplest position update
In addition to the velocity and stress, the position vectors of each particle $\boldsymbol{x}_i$ are updated via the XSPH method at the end of each time step as:
$$\frac{{\rm d} \boldsymbol{x}_i}{{\rm d} t} = \boldsymbol{u}_i + \varepsilon_x\sum_j\frac{m_j}{\rho_j}(\boldsymbol{u}_j - \boldsymbol{u}_i)\nabla W_{ij}$$

Alternatively, the discretised XSPH equation is:
$$\boldsymbol{x}_i^{t+\Delta t} = \boldsymbol{x}_i^t + \Delta t\frac{{\rm d} \boldsymbol{x}_i}{{\rm d} t} = \boldsymbol{x}_i^t + \Delta t(\boldsymbol{u}_i^{t+\Delta t} + \varepsilon_x\sum_j\frac{m_j}{\rho_j}(\boldsymbol{u}_j - \boldsymbol{u}_i)\nabla W_{ij})$$

where $\varepsilon_x$ is a tuning para, $0\leq\varepsilon_x\leq1$.

While, in standard SPH, the simplest way is:
$$\frac{{\rm d} \boldsymbol{x}_i}{{\rm d} t} = \boldsymbol{u}_i$$

And for the particle position update:
$$\boldsymbol{x}_i^{t+\Delta t} = \boldsymbol{x}_i^t + {\Delta t}\boldsymbol{u}_i^{t+\frac{\Delta t}{2}}\ and\ \boldsymbol{u}_i^{t+\frac{\Delta t}{2}} = \frac{1}{2}(\boldsymbol{u}_i^{t+\Delta t}+\boldsymbol{u}_i^t)$$

### Steps
* Key point and aim: update the position, velocity and stress.
* Known $\Delta x$, $\nu$, $E$, $D_{pq}^e$, $\rho_0$, $\boldsymbol{b} = \vec{g}$, and paras for D-P yield criteria $c$, $\varphi$, $\alpha_{\varphi}$ and $k_c$.
* Given $\boldsymbol{x}_i^1$, $\boldsymbol{u}_i^1$, $\boldsymbol{\sigma}_i^1$.
* Step 1: calculate terms $\boldsymbol{f}^{\sigma}$ and $\boldsymbol{f}^u$.
* Step 2: update boundary consitions and adapt the stress.
* Step 3: calculate the gradient terms $(\nabla\cdot\boldsymbol{f}^{\sigma})_i$ and $(\nabla\cdot\boldsymbol{f}^u)_i$.
* Step 4: calculate the additional terms for the momentum equation, mainly the body force $\boldsymbol{b}_i$ in which gravity is the only one considered. Also if included, the artificial viscosity is calculated here.
* Step 5: calculate the additional terms for the constitutive equation, mainly the plastic strain function $\boldsymbol{g}^{\varepsilon^p}_i$.
  * When calculating each particle, the stress state is checked to see if the yield criterion has been met. If the stress state lies within the elastic range, then $\boldsymbol{g}^{\varepsilon^p}_i = 0$. Otherwise, the plastic term is calculated and $\boldsymbol{g}^{\varepsilon^p}_i$ is non-zero.
  * The plastic term is a function of stress and velocity gradients.
  * For large deformation problems, the Jaumann stress rate $\tilde{\boldsymbol{\sigma}}_i$ is also updated. This involves gradients of the velocity.
* Step 6: compute $F_1$ and $F_2$ on particles.
* Step 7: calculate $\boldsymbol{u}_i^2$ and $\boldsymbol{\sigma}_i^2$.
* Step 8: if necessary, the boundary conditions and stress state are again updated.
* Step 9: repeat Steps 1-8 to obtain$\boldsymbol{u}_i^3$, $\boldsymbol{u}_i^4$, $\boldsymbol{\sigma}_i^3$ and $\boldsymbol{\sigma}_i^4$. Then update the velocity $\boldsymbol{u}_i^{t+\Delta t}$ and the stress $\boldsymbol{\sigma}_i^{t+\Delta t}$ at the subsequent time step, also the positions $\boldsymbol{x}_i^{t+\Delta t}$ of the particles.

> **QUESTIONS**
> 1. Just remaining one question: how does $\boldsymbol{g}^{\varepsilon^p}$ calculated through D-P criterion? **It may relates to an adaptation process in the Section 4.3.1 in Chalk's thesis!**

## Stress-Particle SPH

