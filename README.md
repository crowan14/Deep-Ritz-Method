# Deep-Ritz-Method

Some examples of solving 1D and 2D scalar partial differential equations with Pytorch using an energy minimization approach and a neural network discretization of the solution. No data is used in forming the loss function. Pytorch makes the implementation of this method very streamlined. The energy functionals whose minima correspond to a solution for 1D and 2D scalar elliptic equations are

$$ \Pi_{1D} = \int \frac{1}{2}\Big(\frac{\partial u}{\partial x}\Big)^2 - f(x) u dx$$

$$ \Pi_{2D} = \int \frac{1}{2} \frac{\partial u}{\partial x_i} \frac{\partial u}{\partial x_i} - f(x_1,x_2) u d\Omega $$ 

These files could be helpful for anyone looking to get started using Pytorch to solve differential equations.
