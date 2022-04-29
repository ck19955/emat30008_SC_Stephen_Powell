# emat30008_SC_Stephen_Powell

This software is a general numerical continuation code that can track limit cycle oscillations of arbitrary ordinary differential equations when varying a parameter in the system. In addition, it can also track steady-states of second-order diffusive PDEs when varying a parameter. This is done by using the different files to perform numerical continuation on an ODE and outputting a bifurcation diagram of the solutions.

# ODE Solver
The ODE solver file has four different one-step integration methods, Euler, 4th order Runge-Kutta, improved Euler and Heun's method. All of which require the current x and t values, along with information about the ODE (the function and its parameters). Examples of how to plot the solutions of the ODE are given at the bottom of the ODE solver.

# Numerical Shooting
This uses data driven methods and iterative methods such as the Newton-Raphon method found within the function, fsolve(). The shooting method uses the previous ode solver and an external solver, solve_ivp() to find solutions. Examples of how to run the many different functions, is shown at the bottom of the file.

# Numerical Continuation
There are many examples of how to utilise this section at the bottom of the file. This section can plot bifurcation diagrams for the solutions of ODEs by either using pseudo-arclength continuation or natural parameter continuation. This section also examines how the parameters in PDEs can affect the steady states of the PDE. 

# PDE Solver
Here we find the steady states of a PDE using one of three discretisation methods, Forward Euler, Backward Euler and Crank Nicholson. To examine the comparisson of errors of the methods, examples are show in the bottom of the file. You can also select one of the four boundary conditions, homogenous, dirichlet, neumann or periodic.

# Function examples
This file simply keeps all the function examples used throughout the code

# Test Project
This tests many functions in the project to find whether functions are working as intended. To run the tests, run 'pytest' in the terminal when in the directory of the repository.
