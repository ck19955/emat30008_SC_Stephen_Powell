import numpy as np
from math import pi
import matplotlib.pyplot as plt


def u_I(x, l):
    # initial temperature distribution
    y = np.sin(pi*x/l)
    return y


def u_exact(x, t, k, l):
    # the exact solution to the temperature equation
    y = np.exp(-k*(pi**2/l**2)*t)*np.sin(pi*x/l)
    return y


def forward_euler(pde, L, lmbda, mx, mt):
    # Set up the solution variables
    if not 0 < lmbda < 1/2:
        raise RuntimeError("Invalid value for lmbda")
    A = np.diag([1-2*lmbda] * (mx - 1)) + np.diag([lmbda] * (mx - 2), -1) + np.diag([lmbda] * (mx - 2), 1)
    # Solve the matrix equation to return the next value of u
    x_vect = np.linspace(0, L, mx + 1)
    u_vect = np.array(x_vect[1:-1])  # Remove start and end values

    for i in range(len(u_vect)):
        u_vect[i] = pde(u_vect[i], L)
    solution_matrix = [0]*mt
    solution_matrix[0] = u_vect
    for i in range(0, mt-1):
        solution_matrix[i+1] = np.dot(A, solution_matrix[i])
    return solution_matrix


def pde_solver(pde, L, T, method, args):
    """
    :param pde:
    :param L: length of spatial domain
    :param T: total time to solve for
    :param args: diffusion constant
    :return:
    """

    # Set numerical parameters
    mx = 10     # number of gridpoints in space
    mt = 1000   # number of gridpoints in time

    # Set up the numerical environment variables
    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = args[0]*deltat/(deltax**2)    # mesh fourier number

    u_vect = method(pde, L, lmbda, mx, mt)[-1]

    print(u_vect)
    plt.plot(u_vect)
    plt.show()


# Set problem parameters/functions
k = 1.0   # diffusion constant
L = 3.0         # length of spatial domain
T = 0.5         # total time to solve for
pde_solver(u_I, L, T, forward_euler, np.array([k]))
