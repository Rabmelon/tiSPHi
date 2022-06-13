
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

$$\nabla\boldsymbol{u}=\frac{\partial u_i}{\partial x_j}\boldsymbol{e}_i\boldsymbol{e}_j $$

**散度Divergence**：作用于**矢量**$(f_x, f_y, f_z)$得到**标量**。$\mathbb{R}^3\rightarrow\mathbb{R}^1, \nabla\cdot$

$$div\ \boldsymbol{f}=\nabla\cdot \boldsymbol{f}=\frac{\partial f_x}{\partial x} + \frac{\partial f_y}{\partial y} + \frac{\partial f_z}{\partial z}$$

**旋度Curl**：作用于**矢量**$(f_x, f_y, f_z)$得到**矢量**。$\mathbb{R}^3\rightarrow\mathbb{R}^3, \nabla\times$

$$curl\ \boldsymbol{f}=\nabla\times\boldsymbol{f}=\begin{vmatrix} \boldsymbol{i} &\boldsymbol{j} &\boldsymbol{k}\\ \frac{\partial}{\partial x} &\frac{\partial}{\partial y} &\frac{\partial}{\partial z}\\ f_x &f_y &f_z \end{vmatrix}=(\frac{\partial f_z}{\partial y}-\frac{\partial f_y}{\partial z}, \frac{\partial f_x}{\partial z}-\frac{\partial f_z}{\partial x}, \frac{\partial f_y}{\partial x}-\frac{\partial f_x}{\partial y})$$

**拉普拉斯Laplace**: 梯度的散度，作用于任意维度的变量。 $\mathbb{R}^n\rightarrow\mathbb{R}^n, \nabla \cdot \nabla=\nabla^2$

$$laplace\ f=div(grad\ f)=\nabla^2f=\frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} + \frac{\partial^2 f}{\partial z^2}$$

## Material derivative

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

## Other mathematical components

$\dot{\#}$ - the accent-dot indicates the time derivative of the vector/tensor quantities. *from @Bui2021, 3.2.1.1. p15*


