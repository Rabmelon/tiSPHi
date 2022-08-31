# SPH for soil

## Constitutive model of soil

Constitutive model is a core component of a computational framework used to describe how a material behaves under external loads. A constitutive equation is required to relate the soil stresses to the strain rates.

In the application of CFD approach to model geomaterials using SPH, the materials are considered to either be fluid-like materials (i.e. liquefied materials) or have reached its critical state. However, the key drawback of this type of constitutive model is that it cannot describe complex responses of geomaterials, including the hardening or/and softening processes before reaching the critical state of soils.

Advanced constitutive models were built on the basis of continuum plasticity theory.

### A simple elastic-perfectly plastic model for soil

> @bui2021 3.2.1.1.

Standard CFD approach for $c-\varphi$ soils. The shear stresses increase linearly with the incresing shear strain and thus cannot capture the plastic response. A simple approach is to restrict the development of shear stresses when the materials enter the plastic flow regime without actually solving the plastic deformation. (即不计算塑性变形，当材料进入塑性流动状态时，直接按照M-C强度准则约束剪应力)

The stress tensor is decomposed into the isotropic pressure $p$ and deviatoric stress $\boldsymbol{s}$:

$$\boldsymbol{\sigma}=p\boldsymbol{I}+\boldsymbol{s}$$

$p$ is computed using an equation of state (EOS) which is often formulated as a function of density change and sound speed. For geomechanics applications, following the general Hooke's law:

$$p=K\frac{\Delta V}{V_0}=K(\frac{\rho}{\rho_0}-1)$$

On the other hand, the deviatoric shear stress can be estimated using the general Hooke's law for **elastic materials**:

$$\dot{\boldsymbol{s}}=2G(\dot{\boldsymbol{\epsilon}}-\frac{1}{3}\boldsymbol{I}\dot{\epsilon}_v)$$

The plastic regime for general soils can be determinded by the Mohr-Coulomb failure criterion:

$$\tau_f=c+p\tan{\varphi}$$

where $\tau_f=\sqrt{\frac{3}{2}\boldsymbol{s}:\boldsymbol{s}}$ is the maximum shear stress at failure. When the soil enters its plastic flow regime, the shear stress components are scaled back to the yield surface.

### $\mu(I)$-rheological constitutive model

> @bui2021 3.2.1.2. and yang2021 2.2

The $\mu(I)$-rheological model is one of the most commonly used and widely validated rheological models, developed to capture the rate-dependent and inertial effect of granular materials in the dense flow regime.

It was derived based on the Bingham constitutive relation for non-Newtonian fluids. It assumes the materials behave as a rigid body or stiff elastic response before yielding and then quickly reaching their plastic flow behaviour. (假设材料在屈服前表现为刚体或刚性弹性响应？然后迅速达到其塑性流动状态即屈服后的临界状态)

It separates the stress tensor into an isotropic pressure and viscous shear stress tensor, and the viscous shear stress is then defined as a function of total strain-rate:

$$\boldsymbol{\sigma}=-p\boldsymbol{I}+\boldsymbol{\tau}$$

$$\boldsymbol{\tau}=2\eta\dot{\boldsymbol{\epsilon}},\ \eta=\frac{\mu(I)p}{\sqrt{2(\dot{\boldsymbol{\epsilon}}:\dot{\boldsymbol{\epsilon}})}},\ \mu(I)=\mu_s+\frac{\mu_2-\mu_s}{I_0/I+1}$$

where $\eta$ is an effective viscosity, when $\dot{\boldsymbol{\epsilon}}\rightarrow0$, it diverges to infinity and this ensures the material behaviour is rigid or very stiff when the strain rate is very small or at the static condition and thus guaranteeing the existence of a field criterion;

$\dot{\boldsymbol{\epsilon}}$ is the total strain-rate tensor;

$\mu$ is a frictional function dependent on the inertial number $I=d_s\sqrt{2(\dot{\boldsymbol{\epsilon}}:\dot{\boldsymbol{\epsilon}})}/\sqrt{p/\rho_s}$ with $d_s$ being the grain diameter, $\rho_s$ being the solid density;

$\mu_2$ and $I_0$ are both materials constants with $\mu_2$ being the critical friction angle at very high $I$;

and $\mu_s$ is the sratic friction coefficient, corresponding to the state of no plastic flow.

Under the condition of the strain rate tensor in the limit of 0 ($I\rightarrow0$), the second component of $\mu(I)$ will approach 0. This suggests that, under static condition, $\mu(I)=\mu_s$, which defines a yielding threshold above which yielding occurs. Accordingly, the following yield criterion, which takes the form of the Drucker-Prager-like criterion, can be defined:

$$|\boldsymbol{\tau}|\leq\mu_sp,\ |\boldsymbol{\tau}|=\sqrt{0.5(\boldsymbol{\tau}:\boldsymbol{\tau})}$$

The isotropic pressure can be defined alternltively, where the second one is commonly used in the SPH context to model quasi-compressible fluid:

$$p=K\frac{\Delta V}{V_0}=K(\frac{\rho}{\rho_0}-1)\ or\ p=c^2(\rho-\rho_0)$$

where $c$ is the speed of sound, which is assumed to be $10 v_{max}$ (for Yang2021, it is $35m/s$ and for Bui2021, it is $600m/s$).

Finally, it is noted that when incorporating this model, to avoid unphysical behaviour, the **shear component of the stress tensor** should be set to 0 for negative pressure value.

In addition, the **initial strain rate tensor** should be set close to 0 (e.g. $10^{-7}$) as 0 strain rates can result in mathematically undefined behaviour.

To incorporate the shear strength of granular materials, here incorporates the Mohr-Coulomb yield criteria, which allows the yielding shear stress to be described as a function of pressure, as well as easily obtained material properties:

$$\tau_y=c+p\tan\varphi$$

where $c$ is cohesion and $\varphi$ is the internal angle of friction. The 1D modified Bingham shear stress:

$$\tau=\eta_0\dot{\boldsymbol{\epsilon}}+c+p\tan\varphi$$

using an equivalent fluid viscosity, $\eta$, for use in Navier-Stokes solvers:

$$\eta=\eta_0+\frac{c+p\tan\varphi}{\dot{\boldsymbol{\epsilon}}}$$

As for 3D simulation, the generalised form of the modified Bingham shear stress:

$$\boldsymbol{\tau}_i=\eta_0\dot{\boldsymbol{\epsilon}}_i+(c+p\tan\varphi)\frac{\dot{\boldsymbol{\epsilon}}_i}{\sqrt{\frac{1}{2}\dot{\boldsymbol{\epsilon}}_i:\dot{\boldsymbol{\epsilon}}_i}}$$

The above modified Bingham model can be thought of as a precursor to the $\mu(I)$ model, where the $\mu(I)$ model takes the dynamic viscosity $\eta_0$ and the cohesion $c$ as 0, and also exchanges $\tan\varphi$ for a scalar friction value.

**NOTE**: the $\mu(I)$ model need the difference of density to generate pressure, so it is wrong to keep a constant density.

### Elastoplastic model

> @bui2021 3.2.2.

This model was built on basis of continuum plasticity theory, in which a single mathematical relationship that relates the stress-increment to the strain-increment was established for a homogenous representative volume element which is assumed to remain homogenous. A yield surface and a plastic potential function $g$ (or dilatancy rule) are then used to control the hardening or/and softening processes commonly observed in the materials. Elastic and plastic material behaviour are distinguished according to a specified yield function $f$.

The fundamental assumption of plasticity is that the total soil strain rate $\boldsymbol{\dot\epsilon}$ can be divided into an elastic and a plastic component:

$$\boldsymbol{\dot\epsilon} = \boldsymbol{\dot\epsilon}^e+\boldsymbol{\dot\epsilon}^p$$

The stress increment is then calculated from the generalised Hooke's Law:

$$\dot{\boldsymbol{\sigma}}=\boldsymbol{D}^e:\dot{\boldsymbol{\epsilon}}^e=\boldsymbol{D}^{ep}:\dot{\boldsymbol{\epsilon}}$$

We define the elastic strains according to the generalised Hooke's Law:

$$\dot{\boldsymbol{\epsilon}}^e = \frac{\dot{\boldsymbol{s}}}{2G}+\frac{1-2\nu}{3E}\dot{\sigma}_{mm}\boldsymbol{I},\ \dot{\sigma}_{mm}=\dot{\sigma}_{xx}+\dot{\sigma}_{yy}+\dot{\sigma}_{zz}$$

And in plasticity-based models, the plastic srtain rate is defined via the plastic flow rule:

$$\dot{\boldsymbol{\epsilon}}^p=\dot{\lambda}\frac{\partial g}{\partial \boldsymbol{\sigma}}$$

where $\dot{\lambda}$ is the so-called *consistency parameter*, a positive plastic-multiplier, and $g$ is the *plastic potential function*. The plastic potential function describes the direction of plastic flow as a function of the stress tensor. For $g=f$, the flow rule is said to be associated. otherwise, it is non-associated.

And in soil mechanics, the soil pressure $p$ is obtained directly from the equation for **hydrostatic pressure**:

$$p = -\frac{1}{3}(\sigma_{xx}+\sigma_{yy}+\sigma_{zz})$$

> **QUESTIONS**
>
> 1. the hydrostatic pressure $p$, is positive or negtive? $\boldsymbol{s}$ is only correct when $p$ is positive as Chalk2020's Appendix A, but in the main text of Chalk2020, $p$ is negtive. *STRETCH* for positive and *COMPRESS* for negative? **Answer**: Generally it's negtive. When it is positive, the meaning is the average normal stress $\sigma_m = -p$.

#### Yield criteria

For **Drucker-Prager** yield criteria:

$$f=\sqrt{J_2}+\alpha_{\varphi}I_1-k_c=0$$

where the functions of the Coulomb material constants - the soil internal friction $\varphi$ and cohesion $c$:

$$\alpha_{\varphi}=\frac{\tan\varphi}{\sqrt{9+12\tan^2\varphi}}, k_c=\frac{3c}{\sqrt{9+12\tan^2\varphi}}$$

And for the elastoplastic constitutive equation of Drucker-Prager and *non-associated flow rule*

$$g=\sqrt{J_2}+3I_1\sin\psi$$

where $\psi$ is dilatancy angle and in Chalk's thesis $\psi=0$.

And the **Von Mises** criterion is:

$$f = \sqrt{3J_2}-f_c$$

The Von Mises and D-P yield criteria are illustrated in two dimensions:

<div align="center">
  <img width="400px" src="/img/Yield_criterias.png">
</div>

The increment of the yield function after plastic loading or unloading:

$${\rm d}f=\frac{\partial f}{\partial \boldsymbol{\sigma}} {\rm d}\boldsymbol{\sigma}$$

The stress state is not allowed to exceed the yield surface, and the yield function increment cannot be greater than 0. ${\rm d}f=0$ ensures that the stress state remains on the yield surface during plastic loading.

> **QUESTIONS**
>
> 1. How to calculate ${\rm d}f$? **ANSWER**: ${\rm d}f = f^*-f$ in advection.

#### The elastoplastic constitutive equation

After rearranging:

$$\frac{\partial \sigma_{ij}}{\partial t}=2G\dot{e}_{ij}+K\dot{\epsilon}_{mm}\delta_{ij}-\dot{\lambda}((K-\frac{2}{3}G)\frac{\partial g}{\partial \sigma_{kl}}\delta_{kl}\delta_{ij}+2G\frac{\partial g}{\partial \sigma_{ij}})$$

The first two terms on the right hand side describe the elastic strain, while the latter term describes the plastic deformations (which is non-zero when plastic flow occurs).

Upon substitution of plastic potential function and the Drucker-Prager yield function into the equation above:

$$\frac{\partial \sigma_{ij}}{\partial t}=2G\dot{e}_{ij}+K\dot{\epsilon}_{mm}\delta_{ij}-\dot{\lambda}(9K\sin\psi\delta_{ij}+\frac{G}{\sqrt{J_2}}s_{ij}) $$

where

$$\dot{\lambda}=\frac{3\alpha_{\varphi}K\dot{\epsilon}_{mm}+(G/\sqrt{J_2})s_{ij}\dot{\epsilon}_{ij}}{27\alpha_{\varphi}K\sin\psi+G} $$

#### Stress adaptation

> @Bui2008 Section 3.3.1 and Chalk2019 Section 4.3.1

Consider both a **Von Mises** and a **Drucker-Prager** yield criterion to distinguish between elastic and plastic material behaviour.

In the elastoplastic model, the stress state is not allowed to exceed the yield surface and we should apply a stress adaptation to particles, after every calculation step. The stress must be checked at every step and adapted if it does not lie within a valid range.

<div align="center">
  <img width="800px" src="/img/Adaptation_stress_states.png">
</div>

First, the stress state must be adapted if it moves outside the apex of the yield surface, which is konwn as **tension cracking**, in the movement of the stress state at point E to point F. Tension cracking occurss when: $-\alpha_{\varphi}I_1+k_c<0$ or $f\ge\sqrt{J_2}$. And in such circumstances, the hydrostatic stress $I_1$ must be shifted back to the apex of the yield surface by adapting the normal stress components:

$$\hat{\boldsymbol{\sigma}} = \boldsymbol{\sigma}-\frac{1}{3}(I_1-\frac{k_c}{\alpha_{\varphi}})$$

The second corrective stress treatment must be performed when the stress state exceeds the yield surface during plastic loading, as shown by the path A to B. For the D-P yield criterion, this occurs when: $-\alpha_{\varphi}I_1+k_c<\sqrt{J_2}$ or $0<f<\sqrt{J_2}$. And the stress state must be scaleld back appropriately. For this, a scaling factor $r_{\sigma}$ is introduced: $r_{\sigma} = (-\alpha_{\varphi}I_1+k_c) / \sqrt{J_2}$. The deviatoric shear stress is then reduced via this scaling factor for all components of the stress tensor:

$$\hat{\sigma}_{ii} = r_{\sigma}s_{ii}+\frac{1}{3}I_1$$

$$\hat{\sigma}_{ij} = r_{\sigma}s_{ij}$$

The procedure of applying these two equations is referred to as the stress-scaling back procedure, or stress modification.

In the SPH implementation of the elastoplastic model, the two corrective treatments described above are applied to the particles that have a stress state outside of the valid range.

### Viscoplastic Perzyna model

The consistency parameter $\dot{\lambda}$ is defined as:

$$\dot{\lambda}=\gamma\langle\phi(F)\rangle$$

where $\gamma$ is a fluidity parameter (acts as the reciprocal of viscosity) and $\phi(F)$ is a yield-type function. The $\langle...\rangle$ symbol represents the Macaulay brackets:

$$\langle\phi\rangle=\begin{cases} \phi,&\phi\ge0\\ 0,&\phi<0\\ \end{cases}$$

The function $\phi(F)$ is therefore defined as:

$$\phi(F)=(\frac{F-F_0}{F_0})^N$$

where $N$ is a model parameter, $F$ is a function of the stress state (related to the yield function), and $F_0$ defines a critical stress value for plastic strains. Plastic flow occurs then $F>F_0$ (the function $F$ exceeds the critical value $F_0$) and plastic strains are non-zero.

... ...

In summary, the Perzyna constitutive model is defined as:

$$\frac{\partial \sigma_{ij}}{\partial t}=D^e_{ijkl}(\dot{\epsilon}_{kl}-\gamma\frac{\partial g}{\partial\sigma_{kl}}(\frac{F-F_0}{F_0})^N)$$

The implementation of the Von Mises yield criterion, with an associated flow rule ($f=g$):

$$F=\sqrt{3J_2},\ F_0=f_c$$

### A generalised system of equations

The general elastoplastic and viscoplastic Perzyna constitutive equations can be written in the following compact form:

$$\frac{\partial\sigma_{ij}}{\partial t}=D^e_{ijkl}\dot{\epsilon}_{kl}-g_{ij}^{\epsilon^p}$$

where $g_{ij}^{\epsilon^p}$ is a function of the plastic strain, depending on the choice of constitutive model:

$$g_{ij}^{\epsilon^p}=\dot{\lambda}((K-\frac{2}{3}G)\frac{\partial g}{\partial \sigma_{kl}}\delta_{kl}\delta_{ij}+2G\frac{\partial g}{\partial \sigma_{ij}})$$

or

$$g_{ij}^{\epsilon^p}=D^e_{ijkl}\gamma\frac{\partial g}{\partial\sigma_{kl}}(\frac{F-F_0}{F_0})^N$$

And for large deformation problems, the rate of stress must be adapted so that it is invariant with respect to large body rotations. The standard stress rate is replaced with the Jaumann stress rate:

$$\dot{\tilde{\sigma}}_{ij}=\dot{\sigma}_{ij}-\sigma_{im}\dot{\omega}_{jm}-\sigma_{mj}\dot{\omega}_{im}$$

or

$$\dot{\tilde{\boldsymbol{\sigma}}}=\dot{\boldsymbol{\sigma}}-\boldsymbol{\omega}\boldsymbol{\sigma}-\boldsymbol{\sigma}\boldsymbol{\omega}^T $$

Then the equation becomes:

$$\frac{\partial\sigma_{ij}}{\partial t}=\sigma_{im}\dot{\omega}_{jm}-\sigma_{mj}\dot{\omega}_{im}+D^e_{ijkl}\dot{\epsilon}_{kl}-g_{ij}^{\epsilon^p}$$

## Governing equations

Conservation of mass:

$$\frac{{\rm D} \rho}{{\rm D} t}=-\rho \nabla\cdot\boldsymbol{v}$$

Conservation of momentum:

$$\frac{{\rm D} \boldsymbol{v}}{{\rm D} t}=\frac{1}{\rho} \nabla\cdot\boldsymbol{\sigma}+\boldsymbol{f}^{ext}$$

Constitutive equation (in compact form):

$$\frac{{\rm D} \boldsymbol{f}^{\sigma}}{{\rm D} t}=\boldsymbol{\tilde{\sigma}} +\nabla\cdot\boldsymbol{f}^v-\boldsymbol{g}^{\epsilon^p}$$

where:

$$\begin{aligned} \boldsymbol{x} = \left (\begin{array}{c}
    x\\ y
\end{array}\right) \end{aligned}
,
\begin{aligned} \boldsymbol{v} = \left (\begin{array}{c}
    v_x\\ v_y
\end{array}\right) \end{aligned}
,
\begin{aligned} \boldsymbol{\sigma} = \left (\begin{array}{cc}
    \sigma_{xx}    &\sigma_{xy}\\    \sigma_{xy}    &\sigma_{yy}
\end{array}\right) \end{aligned}
,
\begin{aligned} \boldsymbol{f}^{ext} = \left (\begin{array}{c}
    f^{ext}_x\\ f^{ext}_y
\end{array}\right) \end{aligned}$$

$$\begin{aligned} \boldsymbol{f}^{\sigma} = \left (\begin{array}{c}
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

$$\begin{aligned} \boldsymbol{f}^v = \left (\begin{array}{cc}
    D^e_{11}v_x    &D^e_{12}v_y\\ D^e_{21}v_x    &D^e_{22}v_y\\
    D^e_{33}v_y    &D^e_{33}v_x\\ D^e_{41}v_x    &D^e_{42}v_y
\end{array}\right)\end{aligned}
,
\begin{aligned} \boldsymbol{g}^{\epsilon^p} = \left(\begin{array}{c}
      g^{\epsilon^p}_{xx}(\boldsymbol{\dot \epsilon}^p)\\
      g^{\epsilon^p}_{yy}(\boldsymbol{\dot \epsilon}^p)\\
      g^{\epsilon^p}_{xy}(\boldsymbol{\dot \epsilon}^p)\\
      g^{\epsilon^p}_{zz}(\boldsymbol{\dot \epsilon}^p)
\end{array} \right) \end{aligned}
,
\begin{aligned} \dot{\boldsymbol{\epsilon}}^p = \left(\begin{array}{c}
      \dot \epsilon^p_{xx}\\ \dot \epsilon^p_{yy}\\
      \dot \epsilon^p_{xy}\\ 0
\end{array} \right) \end{aligned}$$

$${\boldsymbol{f}^{\dot \epsilon}} = \begin{aligned} \left(\begin{array}{c}
      \dot \epsilon_{xx}\\ \dot \epsilon_{yy}\\
      \dot \epsilon_{xy}\\ 0
\end{array} \right) \end{aligned} = \begin{aligned} \left(\begin{array}{c}
      \frac{\partial v_x}{\partial x}\\ \frac{\partial v_y}{\partial y}\\
      \frac{1}{2}(\frac{\partial v_x}{\partial y}+\frac{\partial v_y}{\partial x})\\ 0
\end{array} \right) \end{aligned}$$

$$\begin{aligned} \boldsymbol{D}^e = D^e_{pq} = \frac{E}{(1+\nu)(1-2\nu)} \left (\begin{array}{cccc}
    1-\nu  &\nu  &0  &\nu\\ \nu  &1-\nu  &0  &\nu\\
    0  &0  &(1-2\nu)/2  &0\\ \nu  &\nu  &0  &1-\nu\\
\end{array}\right) \end{aligned}$$

$D^e_{pq}$ is the **elastic constitutive tensor**, also the ealstic constitutive matrix reduces in plane strain condition.

$\boldsymbol{\tilde{\sigma}}$ is the **Jaumann stress-rate**, which is adopted to achieve an invariant stress rate with respect to rigid-body rotation for large deformation analysis.

$\dot{\omega}_{\alpha\beta}$ is the **spin rate tensor**.

And $\boldsymbol{g}^{\epsilon^p}$ is a vector containing the plastic terms which is the only difference responsible for plastic deformations between the **elastoplastic** and **Perzyna** constitutive models. In both models, the plastic terms are functions of the plastic strain rate, which is dependent on the state of stress and material parameters.

For the elastoplastic model,

$$\boldsymbol{g}^{\epsilon^p} = \dot{\lambda}(9K\sin\psi\delta_{ij}+\frac{G}{\sqrt{J_2}}\boldsymbol{s})$$

which is non-zero only when $f = \sqrt{J_2}+\alpha_{\varphi}I_1-k_c = 0$ (and ${\rm d}f=0$), according to the Drucker-Prager yield criterion.

And for the Perzyna model,

$$\boldsymbol{g}^{\epsilon^p} = \boldsymbol{D}^e\frac{\partial \sqrt{3J_2}}{\partial \boldsymbol{\sigma}}(\frac{\sqrt{3J_2}-f_c}{f_c})^N$$

which is non-zero only when $\sqrt{3J_2}>f_c$ (according to the Von mises yield criterion).

> **QUESTIONS**
>
> 1. How does $\frac{\partial\sqrt{3J_2}}{\partial\boldsymbol{\sigma}}$ calculated?
> 2. What number should $N$ choose?

### Conservation of mass

> @mit fluids lectures [f10](https://web.mit.edu/16.unified/www/FALL/fluids/Lectures/f10.pdf)

All the governing equations of fluid motion which were derived using control volume concepts can be recast in terms of the substantial derivative. We will employ the following general vector identity:

$$\nabla\cdot(a\boldsymbol{u}) = \boldsymbol{u}\cdot\nabla a + a\nabla\cdot\boldsymbol{u}$$

which is valid for any scalar $a$ and any vector $\boldsymbol{u}$.

Beginning with the conservation of mass and the constraint that the density within a moving volume of fluid remains constant, an equivalent condition required for [incompressible flow](https://en.wikipedia.org/wiki/Incompressible_flow) is that the divergence of the flow velocity vanishes. As the loss of mass equals to the net outflow: (控制体内质量的减少=净流出量). So:

$$\frac{\partial \rho}{\partial t}+\nabla\cdot(\rho\boldsymbol{v})=0,\ from\ \frac{\partial\rho}{\partial t}=-\nabla\cdot\boldsymbol{J}=0\ and\ \boldsymbol{J}=\rho\boldsymbol{v}$$

$$-\frac{\partial m}{\partial t} = -\frac{\partial \rho}{\partial t}{\rm d}x{\rm d}y{\rm d}z=[\frac{\partial (\rho v_x)}{\partial x}+\frac{\partial (\rho v_y)}{\partial y}+\frac{\partial (\rho v_z)}{\partial z}]{\rm d}x{\rm d}y{\rm d}z$$

$$\frac{\partial \rho}{\partial t}+\boldsymbol{v}\cdot\nabla\rho+\rho\nabla\cdot\boldsymbol{v}=0$$

The final form in Lagrangian method of density: (left is the change of density, right is the change of volume)

$$\frac{{\rm D}\rho}{{\rm D}t}=-\rho\nabla\cdot\boldsymbol{v}$$

> @bui2021

The original form ($\rho=\sum_j m_jW_{ij}$) of SPH mass equation operator is not suitable because the density will drop in the boundary of calculating domain, not like astrophysics in which there is an infinite domain.

On the other hand, we use

$$\frac{{\rm D}\rho_i}{{\rm D}t}=\sum_jm_j(\boldsymbol{v}_i-\boldsymbol{v}_j)\cdot\nabla_iW_{ij}$$

to solve homogenous problem and use

$$\frac{{\rm D}\rho_i}{{\rm D}t}=\rho_i\sum_jV_j(\boldsymbol{v}_i-\boldsymbol{v}_j)\cdot\nabla_iW_{ij}$$

to solve non-homogenous problem.

### Conservation of momentum

[Cauchy momentum equation](https://en.wikipedia.org/wiki/Cauchy_momentum_equation) is a vector partial differential equation that describes the non-relativistic momentum transport in any continuum. And in convective (or Lagrangian) form is written as:

$$\frac{{\rm D}\boldsymbol{v}}{{\rm D}t}=\frac{1}{\rho}\nabla\cdot\boldsymbol{\sigma}+\boldsymbol{f}$$

> @bui2021

To exactly conserve momentum, we should use the symmetric form:

$$\frac{{\rm D}\boldsymbol{v}_i}{{\rm D}t}=\sum_jm_j(\frac{\boldsymbol{\sigma}_j}{\rho_j^2}+\frac{\boldsymbol{\sigma}_i}{\rho_i^2})\cdot\nabla_iW_{ij}+\boldsymbol{f}^{ext}_i$$

## Standard soil SPH

### Discretization

> @chalk2020 Section 3.1

The discrete governing equations of soil motion in the framework of standard SPH are therefore:

$$\frac{{\rm D} \rho_i}{{\rm D} t} = \rho_i\sum_j V_j(\boldsymbol{v}_i-\boldsymbol{v}_j)\cdot\nabla W_{ij}$$

$$\frac{{\rm D} \boldsymbol{v}_i}{{\rm D} t} = \sum_j m_j(\frac{\boldsymbol{\sigma}_i}{\rho_i^2}+\frac{\boldsymbol{\sigma}_j}{\rho_j^2})\cdot\nabla W_{ij}+\boldsymbol{f}^{ext}_i$$

$$\frac{{\rm D} \boldsymbol{f}^{\sigma}_i}{{\rm D} t} = \boldsymbol{\tilde{\sigma}}_i+\sum_j V_j(\boldsymbol{f}_j^v-\boldsymbol{f}_i^v)\cdot\nabla W_{ij}-\boldsymbol{g}_i^{\epsilon^p}$$

In the current work, each SPH particle is assigned the same, constant density for the duration of the simulation. We treat the soil as incompressible and consequently do not update density through this way.

The relationship of variables in Drucker-Prager model:

<div align="center">
  <img width="800px" src="/img/DP_variables.png">
</div>

### Symp-Euler for standard Drucker-Prager soil SPH

* Known $\Delta x$, $\nu$, $E$, $D_{pq}^e$, $\rho_0$, $\boldsymbol{f}^{ext} = \vec{g}$, $\psi=0$, and paras for D-P yield criteria $c$, $\varphi$, $\alpha_{\varphi}$ and $k_c$
* Given $\boldsymbol{x}_t$, $\boldsymbol{v}_t$, $\boldsymbol{\sigma}_t$ at each particle
* Update boundary
* Cal compact form $\boldsymbol{f}^{\sigma}$ and $\boldsymbol{f}^{v}$
* Cal stress terms $\sigma^H_t$, $s^{ij}_t$
* Cal gradient of velocity tensor $\nabla\cdot\boldsymbol{v}$ or $v_{i,j}$
* Cal strain rate tensor $\dot{\epsilon}_{ij}$, spin rate tensor $\dot{\omega}_{ij}$ and Jaumann stress rate vector $\tilde{\sigma}_{ij}$
* Cal the invariant terms $I_1$ and $J_2$
* Cal the consistency para $\dot{\lambda}$
* Cal the plastic potential vector $\boldsymbol{g}^{\epsilon^p}$
* Cal $\dot{\rho}$, $\dot{\boldsymbol{v}}$, $\dot{\boldsymbol{f}^{\sigma}}$
* Update $\boldsymbol{\sigma}$ and do adaptation
* Update $\rho$, $\boldsymbol{v}$ and $\boldsymbol{x}$

### RK4 for standard Drucker-Prager soil SPH

> @Chalk2020, Appendix B.

The considered governing SPH equations are summarised as:

$$\frac{{\rm D} \boldsymbol{v}_i}{{\rm D} t} = \sum_j V_j(\frac{\boldsymbol{\sigma}_i}{\rho_i^2}+\frac{\boldsymbol{\sigma}_j}{\rho_j^2})\cdot\nabla W_{ij}+\boldsymbol{f}^{ext}_i = F_1(\boldsymbol{\sigma}_i)$$

$$\frac{{\rm D} \boldsymbol{f}^{\sigma}_i}{{\rm D} t} = \boldsymbol{\tilde{\sigma}}_i+\sum_j V_j(\boldsymbol{f}_j^v-\boldsymbol{f}_i^v)\cdot\nabla W_{ij}-\boldsymbol{g}_i^{\epsilon^p} = F_2(\boldsymbol{v}_i,\boldsymbol{\sigma}_i)$$

Using the fourth order Runge-Kutta (RK4) method:

$$\boldsymbol{v}_i^{t+\Delta t} = \boldsymbol{v}_i^t + \frac{\Delta t}{6}(F_1(\boldsymbol{\sigma}^1_i)+2F_1(\boldsymbol{\sigma}^2_i)+2F_1(\boldsymbol{\sigma}^3_i)+F_1(\boldsymbol{\sigma}^4_i))$$

$$\boldsymbol{f}^{\sigma, t+\Delta t}_i = \boldsymbol{f}^{\sigma, t}_i + \frac{\Delta t}{6}(F_2(\boldsymbol{v}^1_i,\boldsymbol{\sigma}^1_i)+2F_2(\boldsymbol{v}^2_i,\boldsymbol{\sigma}^2_i)+2F_2(\boldsymbol{v}^3_i,\boldsymbol{\sigma}^3_i)+F_2(\boldsymbol{v}^4_i,\boldsymbol{\sigma}^4_i))$$

where:

$$\begin{aligned}
    \begin{array}{ll}
      \boldsymbol{v}^1_i = \boldsymbol{v}^t_i &\boldsymbol{f}^{\sigma, 1}_i = \boldsymbol{f}^{\sigma, t}_i\\
      \boldsymbol{v}^2_i = \boldsymbol{v}^t_i+\frac{\Delta t}{2}(F_1(\boldsymbol{\sigma}^1_i)) &\boldsymbol{f}^{\sigma, 2}_i = \boldsymbol{f}^{\sigma, t}_i+\frac{\Delta t}{2}(F_2(\boldsymbol{v}^1_i, \boldsymbol{\sigma}^1_i))\\
      \boldsymbol{v}^3_i = \boldsymbol{v}^t_i+\frac{\Delta t}{2}(F_1(\boldsymbol{\sigma}^2_i)) &\boldsymbol{f}^{\sigma, 3}_i = \boldsymbol{f}^{\sigma, t}_i+\frac{\Delta t}{2}(F_2(\boldsymbol{v}^2_i, \boldsymbol{\sigma}^2_i))\\
      \boldsymbol{v}^4_i = \boldsymbol{v}^t_i+\Delta t(F_1(\boldsymbol{\sigma}^3_i)) &\boldsymbol{f}^{\sigma, 4}_i = \boldsymbol{f}^{\sigma, t}_i+\Delta t(F_2(\boldsymbol{v}^3_i, \boldsymbol{\sigma}^3_i))
    \end{array}
\end{aligned}$$

In standard SPH, these eight eqs are spatially resolved at each calculation step by calculating $\boldsymbol{v}_i^{t+\Delta t}$ and $\boldsymbol{\sigma}_i^{t+\Delta t}$ at each particle.

* Key point and aim: update the position, velocity and stress.
* Known $\Delta x$, $\nu$, $E$, $D_{pq}^e$, $\rho_0$, $\boldsymbol{f}^{ext} = \vec{g}$, and paras for D-P yield criteria $c$, $\varphi$, $\alpha_{\varphi}$ and $k_c$.
* Given $\boldsymbol{x}_i^1$, $\boldsymbol{v}_i^1$, $\boldsymbol{\sigma}_i^1$.
* Step 1: calculate terms $\boldsymbol{f}^{\sigma}$ and $\boldsymbol{f}^v$.
* Step 2: update boundary conditions and adapt the stress.
* Step 3: calculate the gradient terms $(\nabla\cdot\boldsymbol{f}^{\sigma})_i$ and $(\nabla\cdot\boldsymbol{f}^v)_i$.
* Step 4: calculate the additional terms for the momentum equation, mainly the body force $\boldsymbol{f}^{ext}_i$ in which gravity is the only one considered. Also if included, the artificial viscosity is calculated here.
* Step 5: calculate the additional terms for the constitutive equation, mainly the plastic strain function $\boldsymbol{g}^{\epsilon^p}_i$.
    * When calculating each particle, the stress state is checked to see if the yield criterion has been met. If the stress state lies within the elastic range ($f<0$ or $f=0,\ {\rm d}f>0$), then $\boldsymbol{g}^{\epsilon^p}_i = 0$. Otherwise ($f=0,\ {\rm d}f=0$), the plastic term is calculated and $\boldsymbol{g}^{\epsilon^p}_i$ is non-zero.
    * The plastic term is a function of stress $\boldsymbol{\sigma}$ and velocity gradients $\nabla \boldsymbol{v}$.
    * For large deformation problems, the Jaumann stress rate $\tilde{\boldsymbol{\sigma}}_i$ is also updated. This involves gradients of the velocity $\nabla \boldsymbol{v}$.
* Step 6: compute $F_1$ and $F_2$ on particles.
* Step 7: calculate $\boldsymbol{v}_i^2$ and $\boldsymbol{\sigma}_i^2$.
* Step 8: if necessary, the boundary conditions and stress state are again updated.
* Step 9: repeat Steps 1-8 to obtain$\boldsymbol{u}_i^3$, $\boldsymbol{\sigma}_i^3$, $\boldsymbol{v}_i^4$ and $\boldsymbol{\sigma}_i^4$. Then update the velocity $\boldsymbol{v}_i^{t+\Delta t}$ and the stress $\boldsymbol{\sigma}_i^{t+\Delta t}$ at the subsequent time step, also the positions $\boldsymbol{x}_i^{t+\Delta t}$ of the particles.

As for the calculation of plastic potential function item:
<div align="center">
  <img width="800px" src="/img/flowchart_item_strain_DP.svg">
</div>

As for the implementation of RK4:
<div align="center">
  <img width="300px" src="/img/flowchart_RK4_soil.svg">
</div>

