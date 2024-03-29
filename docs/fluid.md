# SPH for fluid

## Navier-Stokes equation

### Forces for incompressible fluids

> @taichiCourse01-10 PPT p8-13

$$f = ma = {\color{Green} f_{ext}} + {\color{RoyalBlue} f_{pres}} + {\color{Orange} f_{visc}}$$

### N-S Equations

> @taichiCourse01-10 PPT p16-28

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

> @taichiCourse01-10 PPT p32

Integrate the incompressible N-S equation in steps (also reffered as "Operator splitting" or "Advection-Projection" in different contexts):

* Step 1: input $v^t$, output $v^{t+0.5\Delta t}$: $\rho\frac{{\rm D} v}{{\rm D} t}={\color{Green} \rho g} + {\color{Orange} \mu\nabla^2v}$
* Step 2: input $v^{t+0.5\Delta t}$, output $v^{t+\Delta t}$: $\rho\frac{{\rm D} v}{{\rm D} t}={\color{RoyalBlue} -\nabla p}\ and\ {\color{RoyalBlue} \nabla\cdot v=0}$ (构成了$\rho$和$v$的二元非线性方程组)

## Full time integration

> @taichiCourse01-10 PPT p33

$$\frac{{\rm D}v}{{\rm D}t}={\color{Green} g} {\color{RoyalBlue} -\frac{1}{\rho}\nabla p} + {\color{Orange} \nu\nabla^2v},\ \nu=\frac{\mu}{\rho_0}$$

* Given $x^t$, $u^t$:
* Step 1: Advection / external and viscosity force integration
    * Solve: ${\color{Purple} {\rm d}v} = {\color{Green} g} + {\color{Orange} \nu\nabla^2v_n}$
    * Update: $v^{t+0.5\Delta t} = v^t+0.5 \Delta t{\color{Purple} {\rm d}v}$
* Step 2: Projection / pressure solver
    * Solve: ${\color{red} {\rm d}v} = {\color{RoyalBlue} -\frac{1}{\rho}\nabla(k(\rho-\rho_0))}$ and ${\color{RoyalBlue} \frac{{\rm D} \rho}{{\rm D} t} = \nabla\cdot(v_{n+0.5}+{\color{red} {\rm d}v})=0}$
    * Update: $v^{t+\Delta t} = v^{t+0.5\Delta t} + 0.5 \Delta t {\color{red} {\rm d}v}$
* Step 3: Update position
    * Update: $x^{t+\Delta t} = x^t+\Delta tv^{t+\Delta t}$
* Return $x^{t+\Delta t}$, $v^{t+\Delta t}$

> **QUESTIONS**
>
> 1. In step 1 and 2, maybe the $\Delta t$ should also multiple 0.5? **ANSWER**: I think yes!

## The weakly compressible assumption

> @taichiCourse01-10 PPT p34-35

Storing the density $\rho$ as an individual variable that advect with the velocity field. Then the $p$ can be assumed as a variable related by time and the mass conserving equation is killed.

* Change in Step 2:
    * Solve: ${\color{red} {\rm d}v} = {\color{RoyalBlue} -\frac{1}{\rho}\nabla(k(\rho-\rho_0))}$
    * Update: $v^{t+\Delta t} = v^{t+0.5\Delta t} + \Delta t {\color{red} {\rm d}v}$

And step 2 and 1 can be merged. This is nothing but Symplectic Euler integration.

### Fluid dynamics with particles

> @taichiCourse01-10 PPT p43 and 75-78

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

        in taichiWCSPH code it's an approximation from @monaghan2005 :

        $$\nu\nabla^2v_i = 2\nu(dim+2)\sum_j \frac{m_j}{\rho_j}(\frac{v_{ij}\cdot r_{ij}}{\|r_{ij}\|^2+0.01h^2})\nabla W_{ij}$$

    * Evaluate pressure gradient (**sym**), where $p = k(\rho-\rho_0)$

        $$-\frac{1}{\rho_i}\nabla p_i = -\frac{\rho_i}{\rho_i}\sum_j m_j(\frac{p_j}{\rho_j^2}+\frac{p_i}{\rho_i^2})\nabla W_{ij} = -\sum_j m_j(\frac{p_j}{\rho_j^2}+\frac{p_i}{\rho_i^2})\nabla W_{ij}$$

        in taichiWCSPH code, $p = k_1((\rho/\rho_0)^{k_2}-1)$, where $k_1$ is a para about stiffness and $k_2$ is just an exponent.

    * Calculate the acceleration

    * Then do time integration using Symplectic Euler method:

        $$v_i^* = v_i+\Delta t\frac{{\rm d}v_i}{{\rm d}t},\ \ x_i^* = x_i+\Delta tv_i^*$$

### RK4 for WCSPH

> @By myself

The momentum equation of WCSPH is as:

$$\frac{{\rm D}v_i}{{\rm D}t}={\color{Green} g} {\color{RoyalBlue} -\frac{1}{\rho_i}\nabla p_i} + {\color{Orange} \nu\nabla^2v_i} = F(v_i)$$

and:

$$v_i^{t+\Delta t} = v_i^t+\frac{\Delta t}{6}(F(v_i^1)+2F(v_i^2)+2F(v_i^3)+F(v_i^4))$$

where:

$$\begin{aligned}
    \begin{array}{ll}
      v^1_i = v^t_i\\
      v^2_i = v^t_i+\frac{\Delta t}{2}(F(v^1_i))\\
      v^3_i = v^t_i+\frac{\Delta t}{2}(F(v^2_i))\\
      v^4_i = v^t_i+\frac{\Delta t}{2}(F(v^3_i))
    \end{array}
\end{aligned}$$
