import numpy as np
from math import pi
import matplotlib.pyplot as plt
from function_examples import *


def forward_euler(pde, L, lmbda, mx, mt, bound_cond, p_func, q_func):
    # Set up the solution variables
    if not 0 < lmbda < 1/2:
        raise RuntimeError("Invalid value for lmbda")
    A = np.diag([1-2*lmbda] * (mx - 1)) + np.diag([lmbda] * (mx - 2), -1) + np.diag([lmbda] * (mx - 2), 1)
    # Solve the matrix equation to return the next value of u
    x_vect = np.linspace(0, L, mx + 1)
    u_vect = np.array(x_vect[1:-1])  # Remove start and end values

    if bound_cond == 'dirichlet':
        additive_vector = np.zeros(len(u_vect))

    for i in range(len(u_vect)):
        u_vect[i] = pde(u_vect[i], L)
    solution_matrix = [0]*mt
    solution_matrix[0] = u_vect
    for i in range(0, mt-1):
        if bound_cond == 'dirichlet':
            additive_vector[0] = p_func(i)
            additive_vector[-1] = q_func(i)
            solution_matrix[i+1] = np.dot(A, solution_matrix[i]) + lmbda*additive_vector
        else:
            solution_matrix[i+1] = np.dot(A, solution_matrix[i])
    return solution_matrix


def backward_euler(pde, L, lmbda, mx, mt, bound_cond, p_func, q_func):
    A = np.diag([1+2*lmbda] * (mx - 1)) + np.diag([-lmbda] * (mx - 2), -1) + np.diag([-lmbda] * (mx - 2), 1)
    # Solve the matrix equation to return the next value of u
    x_vect = np.linspace(0, L, mx + 1)
    u_vect = np.array(x_vect[1:-1])  # Remove start and end values

    if bound_cond == 'dirichlet':
        additive_vector = np.zeros(len(u_vect))

    for i in range(len(u_vect)):
        u_vect[i] = pde(u_vect[i], L)
    solution_matrix = [0]*mt
    solution_matrix[0] = u_vect
    for i in range(0, mt-1):
        if bound_cond == 'dirichlet':
            additive_vector[0] = p_func(i)
            additive_vector[-1] = q_func(i)
            solution_matrix[i+1] = np.linalg.solve(A, solution_matrix[i] + lmbda*additive_vector)
        else:
            solution_matrix[i+1] = np.linalg.solve(A, solution_matrix[i])
    return solution_matrix


def crank_nicholson(pde, L, lmbda, mx, mt, bound_cond, p_func, q_func):
    A = np.diag([1+lmbda] * (mx - 1)) + np.diag([-lmbda/2] * (mx - 2), -1) + np.diag([-lmbda/2] * (mx - 2), 1)
    B = np.diag([1-lmbda] * (mx - 1)) + np.diag([lmbda/2] * (mx - 2), -1) + np.diag([lmbda/2] * (mx - 2), 1)

    # Solve the matrix equation to return the next value of u
    x_vect = np.linspace(0, L, mx + 1)
    u_vect = np.array(x_vect[1:-1])  # Remove start and end values

    if bound_cond == 'dirichlet':
        additive_vector = np.zeros(len(u_vect))

    for i in range(len(u_vect)):
        u_vect[i] = pde(u_vect[i], L)
    solution_matrix = [0]*mt
    solution_matrix[0] = u_vect
    for i in range(0, mt-1):
        if bound_cond == 'dirichlet':
            additive_vector[0] = p_func(i)
            additive_vector[-1] = q_func(i)
            solution_matrix[i+1] = np.linalg.solve(A, np.dot(B, solution_matrix[i]) + lmbda*additive_vector)
        else:
            solution_matrix[i+1] = np.linalg.solve(A, np.dot(B, solution_matrix[i]))
    return solution_matrix


def pde_solver(pde, L, T, method, bound_cond, p_func, q_func, args):
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

    u_vect = method(pde, L, lmbda, mx, mt, bound_cond, p_func, q_func)[-1]

    print(u_vect)
    plt.plot(u_vect)
    plt.show()


if __name__ == '__main__':
    # Set problem parameters/functions
    k = 1.0   # diffusion constant
    L = 3.0         # length of spatial domain
    T = 0.5         # total time to solve for

    # Boundary examples
    # homogenous
    # dirichlet
    # neumann
    # periodic

    pde_solver(u_I, L, T, backward_euler, 'dirichlet', p, q, np.array([k]))
    pde_solver(u_I, L, T, forward_euler, 'dirichlet', p, q, np.array([k]))
    pde_solver(u_I, L, T, crank_nicholson, 'dirichlet', p, q, np.array([k]))
