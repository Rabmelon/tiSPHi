
# Basic mathematics and mechanics

## Tensor

Tensors are simply mathematical objects that can be used to describe physical properties, a tensor is something that transforms like a tensor. The laws of physics described in tensor automatically guarantee this property of being invariant with the reference frame.
The rank (or order) of a tensor is defined by the number of directions required to describe it. A scalar is a 0 rank tensor, a vector is a first rank tensor, a matrix is a two rank tensor. In general, in a 3D space, an $n^{th}$ rank tensor can be described by $3^n$ coefficients.

### Einstein summation convention

Einstein summation convention is a notational convention that implies summation over a set of indexed terms in a formula, thus achieving brevity.
* **Summation index (dummy index)**: an index that is summed over.
  $i$ in $y=\sum_{i=1}^3c_ix_i=c_1x_1+c_2x_2+c_3x_3=c_ix_i$
  * A dummy index can only appear twice in one item. If there are more dummy indexes in one item, it should be added a summation symbol:
    $a_1b_1c_1+a_2b_2c_2+a_3b_3c_3=\sum_{i=1}^3a_ib_ic_i$ **not just** $a_ib_ic_i$
* **Free index**: an index that is not summed over.
  $j$ in $v_i=\sum_{j=1}^3=a_ib_jx_j=a_ib_1x_1+a_ib_2x_2+a_ib_3x_3$
  * A free index cannot be written twice in one item.

Unless otherwise specified, $i,j,k,...$ represent 3D index, and $\alpha,\beta,\gamma,...$ represent 2D index: $\boldsymbol{a}\cdot\boldsymbol{b}=a_ib_i=a_1b_1+a_2b_2+a_3b_3$ and $\boldsymbol{a}\cdot\boldsymbol{b}=a_{\alpha}b_{\alpha}=a_1b_1+a_2b_2$

### Kronecker delta

### Levi-Civita symbol

### Tensor equation / algebra

### Some commonly used special tensors

## Chain rule in derivative

$$
\begin{aligned}
  h &= f(g(x)) \\ h &= (f\circ g)(x) \\ h'(x) &= f'(g(x))g'(x) \\ \frac{{\rm d}(f\circ g)}{{\rm d}x} &= \frac{{\rm d}f}{{\rm d}g}\frac{{\rm d}g}{{\rm d}x} \\ \frac{{\rm d}^2(f\circ g)}{{\rm d}x^2} &= \frac{{\rm d}^2f}{{\rm d}g^2}(\frac{{\rm d}g}{{\rm d}x})^2+\frac{{\rm d}f}{{\rm d}g}\frac{{\rm d}^2g}{{\rm d}x^2} \\ \frac{{\rm d}^3(f\circ g)}{{\rm d}x^3} &= \frac{{\rm d}^3f}{{\rm d}g^3} (\frac{{\rm d}g}{{\rm d}x})^3+3\frac{{\rm d}^2f}{{\rm d}g^2}\frac{{\rm d}g}{{\rm d}x}\frac{{\rm d}^2g}{{\rm d}x^2}+\frac{{\rm d}f}{{\rm d}g}\frac{{\rm d}^3g}{{\rm d}x^3}
\end{aligned}
$$


## The spatial derivative operators in 3D

$\nabla$ 算子的三个语义:

$$\nabla=\boldsymbol{i}\frac{\partial}{\partial x}+\boldsymbol{j}\frac{\partial}{\partial y}+\boldsymbol{k}\frac{\partial}{\partial z}$$

**梯度Gradient**：作用于**标量**$f(x, y, z)$得到**矢量**。$\mathbb{R}^1\rightarrow\mathbb{R}^3, \nabla$

$$grad\ f=\nabla f=(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z})$$

作用于张量则张量rank+1

$$\nabla\boldsymbol{v}=\frac{\partial v_i}{\partial x_j}\boldsymbol{e}_i\boldsymbol{e}_j=v_{i,j} $$

**散度Divergence**：作用于**矢量**$(f_x, f_y, f_z)$得到**标量**。$\mathbb{R}^3\rightarrow\mathbb{R}^1, \nabla\cdot$

$$div\ \boldsymbol{f}=\nabla\cdot \boldsymbol{f}=\frac{\partial f_x}{\partial x} + \frac{\partial f_y}{\partial y} + \frac{\partial f_z}{\partial z}$$

**旋度Curl**：作用于**矢量**$(f_x, f_y, f_z)$得到**矢量**。$\mathbb{R}^3\rightarrow\mathbb{R}^3, \nabla\times$

$$curl\ \boldsymbol{f}=\nabla\times\boldsymbol{f}=\begin{vmatrix} \boldsymbol{i} &\boldsymbol{j} &\boldsymbol{k}\\ \frac{\partial}{\partial x} &\frac{\partial}{\partial y} &\frac{\partial}{\partial z}\\ f_x &f_y &f_z \end{vmatrix}=(\frac{\partial f_z}{\partial y}-\frac{\partial f_y}{\partial z}, \frac{\partial f_x}{\partial z}-\frac{\partial f_z}{\partial x}, \frac{\partial f_y}{\partial x}-\frac{\partial f_x}{\partial y})$$

**拉普拉斯Laplace**: 梯度的散度，作用于任意维度的变量。 $\mathbb{R}^n\rightarrow\mathbb{R}^n, \nabla \cdot \nabla=\nabla^2$

$$laplace\ f=div(grad\ f)=\nabla^2f=\frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} + \frac{\partial^2 f}{\partial z^2}$$

## Material derivative

For a scalar field:

$$\frac{{\rm D}f}{{\rm D}t}=\frac{\partial f}{\partial t}+\boldsymbol{v}\cdot\nabla f,\ r=f_{,t}+v_if_{,i}$$

And for a vector field:

$$\frac{{\rm D}\boldsymbol{f}}{{\rm D}t}=\frac{\partial \boldsymbol{f}}{\partial t}+\boldsymbol{v}\cdot\nabla \boldsymbol{f},\ r_i=f_{i,t}+v_kf_{i,k}$$

This is **material derivative** in fluid mechanics, total derivative in math. 数学上的全导数，流体力学中的物质导数、随体导数，为流体质点在运动时所具有的物理量对时间的全导数。

[Wiki](https://en.wikipedia.org/wiki/Material_derivative): In continuum mechanics, the material derivative describes the time rate of change of some physical quantity (like heat or momentum) of a material element that is subjected to a space-and-time-dependent macroscopic velocity field. The material derivative can serve as a link between Eulerian and Lagrangian descriptions of continuum deformation.<br>
运动的流体微团的物理量随时间的变化率，它等于该物理量由当地时间变化所引起的变化率与由流体对流引起的变化率的和。<br>
从偏导数全微分的摡念出发，密度变化可以认为是密度分布函数（密度场）的时间偏导数项（不定常）和空间偏导数项（空间不均匀）的和。时间偏导项叫局部导数或就地导数。空间偏导项叫位变导数或对流导数。
中科院的李新亮研究员给出了一个更加形象的例子：高铁的电子显示屏上会实时显示车外的温度，如果我们将高铁看作是一个流体微元，它早上从北京出发，中午到达上海，显示屏上记录的室外温度的变化就是物质导数，它包含了两个部分，一是从北京到上海的地理位置的变化所带来的温度变化，即对流导数；二是由于早上到中午由于时间不同而引起的温度变化，即当地导数。)

## Fluid and solid

**Solid**: Applied tengential force/area (or shear stress) $\tau$ produces a proportional deformation angle (or strain) $\theta$. The constant of proportionality $G$ is called the *elastic modulus* and has the units of $force/area$.

$$\tau=G\theta$$

**Fluid**: Applied shear stress $\tau$ produces a proportional continuously-increasing deformation (or strain rate) $\dot\theta$. The constant of proportionality $\mu$ is called the *viscosity* and has the units of $force \times time/area$.

$$\tau=\mu\dot\theta$$

<div align="center">
  <img width="400px" src="/img/Solid_Fluid.png">
</div>

## Strain and strain-rate tensor

### Strain
For [strain](https://www.continuummechanics.org/strain.html), tensorial shear terms are written as $\epsilon_{ij}$ and are one-half of $\gamma_{ij}$ (*engineering shear strain*) such that $\gamma_{ij}=2\epsilon_{ij}, i\neq j$.

The [true shear strain](https://en.wikipedia.org/wiki/Deformation_(physics)#Engineering_strain) is defined as the change in the angle (in radians) between two material line elements initially perpendicular to each other in the undeformed or initial configuration. The engineering shear strain is defined as the tangent of that angle, and is equal to the length of deformation at its maximum divided by the perpendicular length in the plane of force application which sometimes makes it easier to calculate.

$$\epsilon_{Eng}=\frac{L_F-L_0}{L_0}=\frac{\Delta L}{L_0}$$

$$\epsilon_{True}=\int^{L_F}_{L_0}\frac{{\rm d}L}{L}=\ln(\frac{L_F}{L_0})$$

and

$$\epsilon_{True}=\ln(1+\epsilon_{Eng})$$

### Small strain

The best defination of [small strain](https://www.continuummechanics.org/smallstrain.html) is based on the deformation gradient, in terms of displacements $\boldsymbol{u}=\boldsymbol{x}-\boldsymbol{x}_0$, it can be written as:

$$\epsilon_{ij}=\frac{1}{2}(u_{i,j}+u_{j,i})$$

or

$$\boldsymbol{\epsilon}=\frac{1}{2}(\boldsymbol{F}+\boldsymbol{F}^T)-\boldsymbol{I} $$

where $\boldsymbol{F}$ is the [deformation gradient](https://www.continuummechanics.org/deformationgradient.html), which $F_{ij}=x_{i,j}=\delta_{ij}+u_{i,j}$. And $\boldsymbol{F}$ is a Lagrangian quantity.

But it is limited to applications involving small rotations and can only be used to calculate small strains without rotation or within very small rotation (like 5° may cause 3% differences).

### $\boldsymbol{U}−\boldsymbol{I}$ strain

In fact, a perfectly acceptable definition of strain, even for very large strains and rotations will be:

$$\boldsymbol{\epsilon}=\boldsymbol{U}-\boldsymbol{I}$$

where $\boldsymbol{U}$ is the stretch tensor and independent of rigid body rotation (because they are all contained in $\boldsymbol{R}$).

### Green strain tensor

But as $\boldsymbol{U}$ is very difficult to compute, **[Green strain tensor](https://www.continuummechanics.org/greenstrain.html)** is needed that is easy to calculate and is not corrupted by rigid body rotations:

$$E_{ij}=\frac{1}{2}(F_{ki}F_{kj}-\delta_{ij})=\frac{1}{2}(u_{i,j}+u_{j,i}+u_{k,i}u_{k,j})$$

or

$$\boldsymbol{E}=\frac{1}{2}(\boldsymbol{F}^T\cdot\boldsymbol{F}-\boldsymbol{I})$$

The terms can be grouped into *Green strain = Small strain terms + Quadratic terms*. The quadratic terms are what gives the Green strain tensor its rotation independence.

For smaller strains still, the **Green strain tensor** and $\boldsymbol{U}−\boldsymbol{I}$ will become very close to each other, regardless of the level of rotation. But the quadratic terms will affect actual strains when the strains are large.

### Hydrostatic strain

[Hydrostatic strain](https://www.continuummechanics.org/hydrodeviatoricstrain.html) is simply the average of the three normal strains of any strain tensor.

$$\epsilon_H=\frac{1}{3}\epsilon_{kk}$$

Note that hydrostatic strain is in fact a mathematical construct more than a direct physical measure of volume and its change. After all, it is the determinant of the deformation gradient that is the true measure of volume change, and hydrostatic strain is only a convenient approximation of that when the strains are small. Hydrostatic strain is only an approximation of volume change, not an exact measure.

At large strains, hydrostatic strain loses its link to volume because it is no longer an approximation of its change. It is reduced to a mere mathematical property of a strain tensor.

### Volumetric strain

The [volumetric strain](https://www.continuummechanics.org/hydrodeviatoricstrain.html) is defined from the volume change as:

$$\epsilon_{V}=\frac{\Delta V}{V_0}$$

For all strains, there is always the relationship between volumetric strain and principal strain:

$$\epsilon_{V}=(1+\epsilon_1)(1+\epsilon_2)(1+\epsilon_3)=\det(\boldsymbol{F})$$

Here $\det(\boldsymbol{F})$ gives a special symbol, $J$, and a special name, the *Jacobian*.

And for small strains, the relationship between volumetric strain and hydrostatic strain is:

$$\epsilon_{V}=\epsilon_1+\epsilon_2+\epsilon_3=\epsilon_H$$

### Deviatoric strain

[Deviatoric strain](https://www.continuummechanics.org/hydrodeviatoricstrain.html) means all the deformations that cause a shape change without changing the volume if the strains are small.

$$e_{ij}=\epsilon_{ij}'=\epsilon_{ij}-\frac{1}{3}\delta_{ij}\epsilon_{kk} $$

### Strain-rate tensor

> @[wiki: strain-rate tensor](https://en.wikipedia.org/wiki/Strain-rate_tensor#)

In continuum mechanics, the [gradient of the velocity](https://www.continuummechanics.org/velocitygradient.html) $\nabla\boldsymbol{v}$ is a second-order tensor:

$$\boldsymbol{L}=\nabla\boldsymbol{v}=\frac{\partial\boldsymbol{v}}{\partial\boldsymbol{x}}=\left[\begin{matrix} \frac{\partial v_x}{\partial x} &\frac{\partial v_x}{\partial y} &\frac{\partial v_x}{\partial z}\\ \frac{\partial v_y}{\partial x} &\frac{\partial v_y}{\partial y} &\frac{\partial v_y}{\partial z}\\ \frac{\partial v_z}{\partial x} &\frac{\partial v_z}{\partial y} &\frac{\partial v_z}{\partial z} \end{matrix}\right]$$

or

$$L_{ij}=\frac{\partial v_i}{\partial x_j}=v_{i,j} $$

$\boldsymbol{L}$ is an Eulerian quantity and can be decomposed into the sum of a symmetric matrix $\boldsymbol{E}$ and a skew-symmetric matrix $\boldsymbol{W}$:

$$\boldsymbol{E}=\frac{1}{2}(\boldsymbol{L}+\boldsymbol{L}^T)$$

$$\boldsymbol{W}=\frac{1}{2}(\boldsymbol{L}-\boldsymbol{L}^T)$$

$\boldsymbol{E}$ is called the **strain rate tensor** or the rate of deformation tensor and describes the rate of stretching and shearing. $\boldsymbol{W}$ is called the **spin tensor** and describes the rate of rotation.

Also, the strain-rate tensor can be noted as $\dot{\boldsymbol{\epsilon}}$ or $\dot{\epsilon}_{ij}$, and the spin rate tensor as $\dot{\boldsymbol{\omega}}$ or $\dot{\omega}_{ij}$.

And the rate of deformation equals the rate of [true strain](https://www.continuummechanics.org/truestrain.html): $\int E{\rm d}t=\epsilon_{True}$ and $E=\dot{\epsilon}_{True}$. However, things get complicated when the rate of deformation tensor is integrated over time to obtain true strain while rigid body rotations are present. But we can compute $\boldsymbol{\epsilon}_{True}=\int \boldsymbol{R}^T\cdot\boldsymbol{E}\cdot\boldsymbol{R}{\rm d}t$ instead of directly calculate $\int \boldsymbol{E}{\rm d}t$ when rotations are present. This gives a true strain result that is in the initial reference orientation.

For compressible materials, since the ratio of initial to final volume is $\epsilon_V^{True}$:

$$\epsilon_1^{True}+\epsilon_2^{True}+\epsilon_3^{True}=\epsilon_V^{True}$$

Also:

$$\dot{\epsilon}_1^{True}+\dot{\epsilon}_2^{True}+\dot{\epsilon}_3^{True}=\dot{\epsilon}_V^{True}$$

which means:

$$E_{kk}=\dot{\epsilon}_V^{True}$$

For incompressible materials, just take $\epsilon_V^{True}=0$ ($V_F/V_0=1$) and $\dot{\epsilon}_V^{True}=0$. And this above applies for finite strains, not just infinitesimal ones, and not just in principal orientations.

The strain-rate tensor describes the rate of change of the deformation of a material in the neighborhood of a certain point, at a certain moment of time. It can be defined as the derivative of the strain tensor with respect to time, or as the symmetric component of the Jacobian matrix of the flow velocity.

## Stress

Stress: $\boldsymbol{\sigma}$ or $\sigma_{ij}$ (Stretch for positive and Compress for negative)

Principal stress: $\sigma_1$, $\sigma_2$ and $\sigma_3$

Hydrostatic stress: $\sigma_H=\sigma_{mm}/3$

Hydrostatic pressure in geomechanic: $p=-\sigma_H$

Deviatoric stress: $\boldsymbol{s}=\boldsymbol{\sigma}'=\boldsymbol{\sigma}-\sigma_H\boldsymbol{I}$ or $s_{ij}=\sigma'_{ij}=\sigma_{ij}-\delta_{ij}\sigma_H$

## Invariants

Here we difine the firse invariant of the stress tensor $I_1$ and the second invariant of the deviatoric stress tensor $J_2$:

$$I_1 = \sigma_{xx}+\sigma_{yy}+\sigma_{zz}\ ,\ J_2 = \frac{1}{2}\boldsymbol{s}:\boldsymbol{s}$$

where $\boldsymbol{s}$ is the **deviatoric stress tensor**: $\boldsymbol{s} = \boldsymbol{\sigma}+p\boldsymbol{I}$ and $\boldsymbol{I}$ is the identity matrix. $\boldsymbol{s}:\boldsymbol{s}$ means $s_{ij}s_{ij}$.


## Hooke's Law

Hooke's Law can be written in matrix notation as:

$$\boldsymbol{\epsilon}=\frac{1}{E}[(1+\nu)\boldsymbol{\sigma}-\nu\boldsymbol{I}{\rm tr}(\boldsymbol{\sigma})]$$

and in tensor notation as:

$$\epsilon_{ij}=\frac{1}{E}[(1+\nu)\sigma_{ij}-\nu\delta_{ij}\sigma_{kk}] $$

The inverting Hooke's Law:

$$\sigma_{ij}=\frac{E}{1+\nu}[\epsilon_{ij}+\frac{\nu}{1-2\nu}\delta_{ij}\epsilon_{kk}] $$

In [Hooke's Law](https://www.continuummechanics.org/hookeslaw.html), we have $\tau_{xy}=G\gamma_{xy}=2G\epsilon_{xy}$ and $\epsilon_{xx}+\epsilon_{yy}+\epsilon_{zz}=\epsilon_{V}=\sigma_m/K=(\sigma_{xx}+\sigma_{yy}+\sigma_{zz})/3K$.

$E$ is the **elastic modulus** or **Young's modulus**, $\nu$ is the **Poisson's ratio**, $G = E/2(1+\nu)$ is the **shear modulus** and $K = E/3(1-2\nu)$ is the **elastic bulk modulus**.

And for stiffness tensor:

$$C_{ijkl}=\frac{E}{1+\nu}[\frac{1}{2}(\delta_{ik}\delta_{jl}+\delta_{jk}\delta_{il})+\frac{\nu}{1-2\nu}\delta_{ij}\delta_{kl}]$$

The deviatoric stress $\boldsymbol{s}=\boldsymbol{\sigma}'$ and strain $\boldsymbol{\epsilon}'$ are directly proportional to each other:

$$s_{ij}=2G\epsilon'_{ij}$$

## Other mathematical components
1. $\dot{\#}$ - the accent-dot indicates the time derivative of the vector/tensor quantities. *@Bui2021, 3.2.1.1. p15*
2. The speed of sound: 343m/s in air, 1481m/s in water, 5120m/s in iron.

