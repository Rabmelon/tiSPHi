# SPH for soil

## Constitutive model of soil

In the application of CFD approach to model geomaterials using SPH, the materials are considered to either be fluid-like materials (i.e. liquefied materials) or have reached its critical state. However, the key drawback of this type of constitutive model is that it cannot describe complex responses of geomaterials, including the hardening or/and softening processes before reaching the critical state of soils.
Advanced constitutive models were built on the basis of continuum plasticity theory

### A simple elastic-perfectly plastic model for soil
> @bui2021 3.2.1.1.

Standard CFD approach for $c-\varphi$ soils. The shear stresses increase linearly with the incresing shear strain and thus cannot capture the plastic response. A simple approach is to restrict the development of shear stresses when the materials enter the plastic flow regime without actually solving the plastic deformation. (即不计算塑性变形，当材料进入塑性流动状态时，直接按照M-C强度准则约束剪应力)
The stress tensor is decomposed into the isotropic pressure $p$ and deviatoric stress $\boldsymbol{s}$:

$$\boldsymbol{\sigma}=p\boldsymbol{I}+\boldsymbol{s}$$

$p$ is computed using an equation of state (EOS) which is often formulated as a function of density change and sound speed. For geomechanics applications, following the general Hooke's law:

$$p=K\frac{\Delta V}{V_0}=K(\frac{\rho}{\rho_0}-1)$$

On the other hand, the deviatoric shear stress can be estimated using the general Hooke's law for **elastic materials**:

$$\dot{\boldsymbol{s}}=2G(\dot{\boldsymbol{\varepsilon}}-\frac{1}{3}\boldsymbol{I}\dot{\varepsilon}_v)$$

The plastic regime for general soils can be determinded by the Mohr-Coulomb failure criterion:

$$\tau_f=c+p\tan{\varphi}$$

where $\tau_f=\sqrt{\frac{3}{2}\boldsymbol{s}:\boldsymbol{s}}$ is the maximum shear stress at failure. When the soil enters its plastic flow regime, the shear stress components are scaled back to the yield surface.

### $\mu(I)$-rheological constitutive model
> @bui2021 3.2.1.2. and yang2021 2.2

The $\mu(I)$-rheological model is one of the most commonl used and widely validated rheological models, developed to capture the rate-dependent and inertial effect of granular materials in the dense flow regime.
It was derived based on the Bingham constitutive relation for non-Newtonian fluids. It assumes the materials behave as a rigid body or stiff elastic response before yielding and then quickly reaching their plastic flow behaviour. (假设材料在屈服前表现为刚体或刚性弹性响应？然后迅速达到其塑性流动状态即屈服后的临界状态)
It separates the stress tensor into an isotropic pressure and viscous shear stress tensor, and the viscous shear stress is then defined as a function of total strain-rate:

$$\boldsymbol{\sigma}=-p\boldsymbol{I}+\boldsymbol{\tau}$$

$$\boldsymbol{\tau}=2\eta\dot{\boldsymbol{\varepsilon}},\ \eta=\frac{\mu(I)p}{\sqrt{2(\dot{\boldsymbol{\varepsilon}}:\dot{\boldsymbol{\varepsilon}})}},\ \mu(I)=\mu_s+\frac{\mu_2-\mu_s}{I_0/I+1}$$

where $\eta$ is an effective viscosity, when $\dot{\boldsymbol{\varepsilon}}\rightarrow0$, it diverges to infinity and this ensures the material behaviour is rigid or very stiff when the strain rate is very small or at the static condition and thus guaranteeing the existence of a field criterion;
$\dot{\boldsymbol{\varepsilon}}$ is the total strain-rate tensor;
$\mu$ is a frictional function dependent on the inertial number $I=d_s\sqrt{2(\dot{\boldsymbol{\varepsilon}}:\dot{\boldsymbol{\varepsilon}})}/\sqrt{p/\rho_s}$ with $d_s$ being the grain diameter, $\rho_s$ being the solid density;
$\mu_2$ and $I_0$ are both materials constants with $\mu_2$ being the critical friction angle at very high $I$;
and $\mu_s$ is the sratic friction coefficient, corresponding to the state of no plastic flow.

Under the condition of the strain rate tensor in the limit of 0 ($I\rightarrow0$), the second component of $\mu(I)$ will approach 0. This suggests that, under static condition, $\mu(I)=\mu_s$, which defines a yielding threshold above which yielding occurs. Accordingly, the following yield criterion, which takes the form of the Drucker-Prager-like criterion, can be defined:

$$|\boldsymbol{\tau}|\leq\mu_sp,\ |\boldsymbol{\tau}|=\sqrt{0.5(\boldsymbol{\tau}:\boldsymbol{\tau})}$$

The isotropic pressure can be defined alternltively, where the second one is commonly used in the SPH context to eodel quasi-comopressible fluids:

$$p=K\frac{\Delta V}{V_0}=K(\frac{\rho}{\rho_0}-1)\ or\ p=c^2(\rho-\rho_0)$$

where $c$ is the speed of sound, which is assumed to be $10 v_{max}$ (for Yang2021, it is $35m/s$ and for Bui2021, it is $600m/s$).

Finally, it is noted that when incorporating this model, to avoid unphysical behaviour, the **shear component of the stress tensor** should be set to 0 for negative pressure value.

In addition, the **initial strain rate tensor** should be set close to 0 (e.g. $10^{-7}$) as 0 strain rates can result in mathematically undefined behaviour.

To incorporate the shear strength of granular materials, here incorporates the Mohr-Coulomb yield criteria, which allows the yielding shear stress to be described as a function of pressure, as well as easily obtained material properties:

$$\tau_y=c+p\tan\varphi$$

where $c$ is cohesion and $\varphi$ is the internal angle of friction. The 1D modified Bingham shear stress:

$$\tau=\eta_0\dot{\boldsymbol{\varepsilon}}+c+p\tan\varphi$$

using an equivalent fluid viscosity, $\eta$, for use in Navier-Stokes solvers:

$$\eta=\eta_0+\frac{c+p\tan\varphi}{\dot{\boldsymbol{\varepsilon}}}$$

As for 3D simulation, the generalised form of the modified Bingham shear stress:

$$\boldsymbol{\tau}_i=\eta_0\dot{\boldsymbol{\varepsilon}}_i+(c+p\tan\varphi)\frac{\dot{\boldsymbol{\varepsilon}}_i}{\sqrt{\frac{1}{2}\dot{\boldsymbol{\varepsilon}}_i:\dot{\boldsymbol{\varepsilon}}_i}}$$

The above modified Bingham model can be thought of as a precursor to the $\mu(I)$ model, where the $\mu(I)$ model takes the dynamic viscosity $\eta_0$ and the cohesion $c$ as 0, and also exchanges $\tan\varphi$ for a scalar friction value.


### Drucker-Prager yield criteria
Constitutive model is to relate the soil stresses to the strain rates in the plane strain condition.
For **Drucker-Prager** yield criteria: $f=\sqrt{J_2}+\alpha_{\varphi}I_1-k_c=0$ and functions of the Coulomb material constants - the soil internal friction $\varphi$ and cohesion $c$:

$$\alpha_{\varphi}=\frac{\tan\varphi}{\sqrt{9+12\tan^2\varphi}}, k_c=\frac{3c}{\sqrt{9+12\tan^2\varphi}}$$

And for the elastoplastic constitutive equation of Drucker-Prager and *non-associated flow rule*, $g=\sqrt{J_2}+3I_1\cdot\sin\psi$, where $\psi$ is dilatancy angle and in Chalk's thesis $\psi=0$. Of *associated flow rule*, $g=\sqrt{J_2}+\alpha_{\varphi}I_1-k_c$. $g$ is the plastic potential function (塑性势函数).
And the **Von Mises** criterion is: $f = \sqrt{3J_2}-f_c$.
The Von Mises and D-P yield criteria are illustrated in two dimensions:
<div align="center">
  <img width="400px" src="/img/Yield_criterias.png">
</div>

Here we difine the firse invariant of the stress tensor $I_1$ and the second invariant of the deviatoric stress tensor $J_2$:

$$I_1 = \sigma_{xx}+\sigma_{yy}+\sigma_{zz}\ ,\ J_2 = \frac{1}{2}\boldsymbol{s}:\boldsymbol{s}$$

> **QUESTIONS**
> 1. How does the operator : calculated? **Answer**: double dot product of tensors, also a double tensorial contraction. The double dots operator "eats" two 2nd rank tensors and "spits out" a scalar. As for $\boldsymbol{s}:\boldsymbol{s}$, it represents the sum of squares of each element in $\boldsymbol{s}$.

The increment of the yield function after plastic loading or unloading:

$${\rm d}f=\frac{\partial f}{\partial \boldsymbol{\sigma}} {\rm d}\boldsymbol{\sigma}$$

The stress state is not allowed to exceed the yield surface, and the yield function increment cannot be greater than 0. ${\rm d}f=0$ ensures that the stress state remains on the yield surface during plastic loading.

> **QUESTIONS**
> 1. How to calculate ${\rm d}f$? **ANSWER**: ${\rm d}f = f^*-f$ in advection.

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

> @Bui2008 Section 3.3.1 and Chalk2019 Section 4.3.1

But the stress state is not allowed to exceed the yield surfae. The stress must be checked at every step and adapted if it does not lie within a valid range.
<div align="center">
  <img width="800px" src="/img/Adaptation_stress_states.png">
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
      2\sigma_{xy}\dot\omega_{xy}\\ 2\sigma_{xy}\dot\omega_{yx}\\
      \sigma_{xx}\dot\omega_{yx}+\sigma_{yy}\dot\omega_{xy}\\ 0
    \end{array} \right)
    = \left(\begin{array}{c}
      2\sigma_{xy}\dot\omega_{xy}\\ -2\sigma_{xy}\dot\omega_{xy}\\
      (\sigma_{yy}-\sigma_{xx})\dot\omega_{xy}\\ 0
\end{array} \right) \end{aligned}$$

$$\dot\omega_{\alpha\beta}=\frac{1}{2}(\frac{\partial u_{\alpha}}{\partial x_{\beta}}-\frac{\partial u_{\beta}}{\partial x_{\alpha}})\ ,\ \dot\omega_{xy} = \frac{1}{2}(\frac{\partial u_x}{\partial x_y}-\frac{\partial u_y}{\partial x_x})$$

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

$$\dot\varepsilon_{\alpha\beta}=\frac{1}{2}(\frac{\partial u_{\alpha}}{\partial x_{\beta}}+\frac{\partial u_{\beta}}{\partial x_{\alpha}}),\
\dot{\boldsymbol{\varepsilon}} = \begin{aligned} \left(\begin{array}{c}
      \dot \varepsilon_{xx}\\ \dot \varepsilon_{yy}\\
      \dot \varepsilon_{xy}\\ 0
\end{array} \right) \end{aligned} = \begin{aligned} \left(\begin{array}{c}
      \frac{\partial u_x}{\partial x}\\ \frac{\partial u_y}{\partial y}\\
      \frac{1}{2}(\frac{\partial u_x}{\partial y}+\frac{\partial u_y}{\partial x})\\ 0
\end{array} \right) \end{aligned}$$

$$\begin{aligned} \boldsymbol{D}^e = D^e_{pq} = \frac{E}{(1+\nu)(1-2\nu)} \left (\begin{array}{cccc}
    1-\nu  &\nu  &0  &\nu\\ \nu  &1-\nu  &0  &\nu\\
    0  &0  &(1-2\nu)/2  &0\\ \nu  &\nu  &0  &1-\nu\\
\end{array}\right) \end{aligned}$$

$D^e_{pq}$ is the **elastic constitutive tensor**, also the ealstic constitutive matrix reduces in plane strain condition.
$\boldsymbol{\tilde{\sigma}}$ is the **Jaumann stress-rate**, which is adopted to achieve an invariant stress rate with respect to rigid-body rotation for large deformation analysis.
$\dot{\omega}_{\alpha\beta}$ is the **spin rate tensor**.
And $\boldsymbol{g}^{\varepsilon^p}$ is a vector containing the plastic terms which is the only difference responsible for plastic deformations between the **elastoplastic** and **Perzyna** constitutive models. In both models, the plastic terms are functions of the plastic strain rate, which is dependent on the state of stress and material parameters.

For the elastoplastic model,

$$\boldsymbol{g}^{\varepsilon^p} = \dot{\lambda}\frac{G}{\sqrt{J_2}\boldsymbol{s}}$$

which is non-zero only when $f = \sqrt{J_2}+\alpha_{\varphi}I_1-k_c = 0$ (and ${\rm d}f=0$), according to the D-P yield criterion, where:

$$\dot{\lambda} = \frac{3\alpha_{\varphi}\dot{\varepsilon}_{kk}+(G/\sqrt{J_2})\boldsymbol{s}:\dot{\boldsymbol{\varepsilon}}}{27\alpha_{\varphi}K\sin{\psi}+G} = \frac{3\alpha_{\varphi}\dot{\varepsilon}_{kk}+(G/\sqrt{J_2})\boldsymbol{s}:\dot{\boldsymbol{\varepsilon}}}{G}$$

and $G = E/2(1+\nu)$ is the **shear modulus** and $K = E/3(1-2\nu)$ is the **elastic bulk modulus** (although $K$ is not used here).

And for the Perzyna model,

$$\boldsymbol{g}^{\varepsilon^p} = \boldsymbol{D}^e\frac{\partial \sqrt{3J_2}}{\partial \boldsymbol{\sigma}}(\frac{\sqrt{3J_2}-f_c}{f_c})^{\hat{N}}$$

which is non-zero only when $\sqrt{3J_2}>f_c$ (according to the Von mises yield criterion). And $\hat{N}$ is a model parameter.

> **QUESTIONS**
> 1. How does $\boldsymbol{g}^{\varepsilon^p}$ and $\boldsymbol{\dot\varepsilon}^p$ calculated? Maybe it is different in elastoplastic and Perzyna models. **ANSWER**: as it shows
> 2. How does $\dot{\lambda}$ calculated? **ANSWER**: as it shows
> 3. How does $\frac{\partial\sqrt{3J_2}}{\partial\boldsymbol{\sigma}}$ calculated?
> 4. What number should $\hat{N}$ choose?
> 5. What's the difference between $\dot{\boldsymbol{\varepsilon}}$ and $\dot{\boldsymbol{\varepsilon}^p}$? **ANSWER**: use $\nabla \boldsymbol{u}$.

### Conservation of mass
> @mit fluids lectures [f10](https://web.mit.edu/16.unified/www/FALL/fluids/Lectures/f10.pdf)

All the governing equations of fluid motion which were derived using control volume concepts can be recast in terms of the substantial derivative. We will employ the following general vector identity:

$$\nabla\cdot(a\boldsymbol{u}) = \boldsymbol{u}\cdot\nabla a + a\nabla\cdot\boldsymbol{u}$$

which is valid for any scalar $a$ and any vector $\boldsymbol{u}$.

Beginning with the conservation of mass and the constraint that the density within a moving volume of fluid remains constant, an equivalent condition required for [incompressible flow](https://en.wikipedia.org/wiki/Incompressible_flow) is that the divergence of the flow velocity vanishes. As the loss of mass equals to the net outflow: (控制体内质量的减少=净流出量). So:

$$\frac{\partial \rho}{\partial t}+\nabla\cdot(\rho\boldsymbol{u})=0,\ from\ \frac{\partial\rho}{\partial t}=-\nabla\cdot\boldsymbol{J}=0\ and\ \boldsymbol{J}=\rho\boldsymbol{u}$$

$$-\frac{\partial m}{\partial t} = -\frac{\partial \rho}{\partial t}{\rm d}x{\rm d}y{\rm d}z=[\frac{\partial (\rho u_x)}{\partial x}+\frac{\partial (\rho u_y)}{\partial y}+\frac{\partial (\rho u_z)}{\partial z}]{\rm d}x{\rm d}y{\rm d}z$$

$$\frac{\partial \rho}{\partial t}+\boldsymbol{u}\cdot\nabla\rho+\rho\nabla\cdot\boldsymbol{u}=0$$

The final form in Lagrangian method of density: (left 为微团密度的变化，right 为微团体积的变化。)

$$\frac{{\rm D}\rho}{{\rm D}t}=-\rho\nabla\cdot\boldsymbol{u}$$

### Conservation of momentum

[Cauchy momentum equation](https://en.wikipedia.org/wiki/Cauchy_momentum_equation) is a vector partial differential equation that describes the non-relativistic momentum transport in any continuum. And in convective (or Lagrangian) form is written as:

$$\frac{{\rm D}\boldsymbol{u}}{{\rm D}t}=\frac{1}{\rho}\nabla\cdot\boldsymbol{\sigma}+\boldsymbol{f}$$

### Constitutive equation

Unlike the CFD approach, the general elastoplastic constitutive modelling approach evolves the stress tensor over time using a unique stress-strain relationship that relates the stress-increment to the strain-increment. It is assumed that for an elastoplastic material, the total strain-increment tensor ${\rm d}\boldsymbol{\varepsilon}$ is decomposed into elastic and plastic components: ${\rm d}\boldsymbol{\varepsilon}={\rm d}\boldsymbol{\varepsilon}_e+{\rm d}\boldsymbol{\varepsilon}_p$

The stress increment is then calculated from specific rules: ${\rm d}\boldsymbol{\sigma}=\boldsymbol{D}^{ep}:{\rm d}\boldsymbol{\varepsilon}$

> **QUESTIONS**:
> 1. Is the stress derivative ? deviation? divergancy? the material derivative or partial derivative? It should be $\partial\sigma/\partial t$? Or the stress is also proper to be described in material derivative?

## Standard soil SPH

### Discretization
> @chalk2020 Section 3.1

The discrete governing equations of soil motion in the framework of standard SPH are therefore:

$$\frac{{\rm D} \rho_i}{{\rm D} t} = -\sum_j m_j(\boldsymbol{u}_j-\boldsymbol{u}_i)\cdot\nabla W_{ij}$$

$$\frac{{\rm D} \boldsymbol{u}_i}{{\rm D} t} = \sum_j m_j(\frac{\boldsymbol{f}_i^{\sigma}}{\rho_i^2}+\frac{\boldsymbol{f}_j^{\sigma}}{\rho_j^2})\cdot\nabla W_{ij}+\boldsymbol{f}^{ext}_i$$

$$\frac{{\rm D} \boldsymbol{\sigma}_i}{{\rm D} t} = \boldsymbol{\tilde{\sigma}}_i+\sum_j \frac{m_j}{\rho_j}(\boldsymbol{f}_j^u-\boldsymbol{f}_i^u)\cdot\nabla W_{ij}-\boldsymbol{g}_i^{\varepsilon^p}$$

In the current work, each SPH particle is assigned the same, constant density for the duration of the simulation. We treat the soil as incompressible and consequently do not update density through this way.

### Symp-Euler for standard soil SPH

* Known $\Delta x$, $\nu$, $E$, $D_{pq}^e$, $\rho_0$, $\boldsymbol{f}^{ext} = \vec{g}$, and paras for D-P yield criteria $c$, $\varphi$, $\alpha_{\varphi}$ and $k_c$.
* Given $\boldsymbol{x}_i^1$, $\boldsymbol{u}_i^1$, $\boldsymbol{\sigma}_i^1$.
* Update boundary
* Cal gradient of velocity tensor
* Cal strain tensor
* Cal spin rate and Jaumann stress rate tensor

<!-- TODO: clarify and add details of SE DPSPH -->


### RK4 for standard soil SPH
> @Chalk2020, Appendix B.

The considered governing SPH equations are summarised as:

$$
\frac{{\rm D} \boldsymbol{u}_i}{{\rm D} t} = \sum_j m_j(\frac{\boldsymbol{f}_i^{\sigma}}{\rho_i^2}+\frac{\boldsymbol{f}_j^{\sigma}}{\rho_j^2})\cdot\nabla W_{ij}+\boldsymbol{f}^{ext}_i = F_1(\boldsymbol{\sigma}_i)
$$

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
    * When calculating each particle, the stress state is checked to see if the yield criterion has been met. If the stress state lies within the elastic range ($f<0$ or $f=0,\ {\rm d}f>0$), then $\boldsymbol{g}^{\varepsilon^p}_i = 0$. Otherwise ($f=0,\ {\rm d}f=0$), the plastic term is calculated and $\boldsymbol{g}^{\varepsilon^p}_i$ is non-zero.
    * The plastic term is a function of stress $\boldsymbol{\sigma}$ and velocity gradients $\nabla \boldsymbol{u}$.
    * For large deformation problems, the Jaumann stress rate $\tilde{\boldsymbol{\sigma}}_i$ is also updated. This involves gradients of the velocity $\nabla \boldsymbol{u}$.
* Step 6: compute $F_1$ and $F_2$ on particles.
* Step 7: calculate $\boldsymbol{u}_i^2$ and $\boldsymbol{\sigma}_i^2$.
* Step 8: if necessary, the boundary conditions and stress state are again updated.
* Step 9: repeat Steps 1-8 to obtain$\boldsymbol{u}_i^3$, $\boldsymbol{\sigma}_i^3$, $\boldsymbol{u}_i^4$ and $\boldsymbol{\sigma}_i^4$. Then update the velocity $\boldsymbol{u}_i^{t+\Delta t}$ and the stress $\boldsymbol{\sigma}_i^{t+\Delta t}$ at the subsequent time step, also the positions $\boldsymbol{x}_i^{t+\Delta t}$ of the particles.

As for the calculation of strain item:
<div align="center">
  <img width="800px" src="/img/flowchart_item_strain_DP.svg">
</div>
As for the implementation of RK4:
<div align="center">
  <img width="300px" src="/img/flowchart_RK4_soil.svg">
</div>

