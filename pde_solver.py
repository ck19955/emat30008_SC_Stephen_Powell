import numpy as np
from math import pi
import matplotlib.pyplot as plt
from function_examples import *


def forward_euler(pde, L, lmbda, mx, mt, bound_cond, p_func, q_func):
    """
    Solves a given pde using the forward euler method

    Parameters:
    ----------
        pde : function
            The PDE to be solved
        L : float
            The length of the spatial domain
        lmbda : float
            Mesh fourier number
        mx : integer
            Number of gridpoints in space
        mt : integer
            Number of gridpoints in time
        bound_cond : string
            boundary condition type for the PDE
        p_func : function
            The value at u(0, t)
        q_func : function
            The value at u(L, t)

    Returns:
    ----------
        solution_matrix : numpy array
            Matrix of all data points in the space at any given time
    """

    # Check whether forward_euler method is suitable
    if not 0 < lmbda < 1/2:
        raise RuntimeError("Invalid value for lmbda")

    # Evaluate initial solution values
    u_vect = np.linspace(0, L, mx + 1)
    for i in range(mx+1):
        u_vect[i] = pde(u_vect[i], L)

    # Initialise the matrix of solutions with the initial solution values
    solution_matrix = np.zeros((mt, mx+1))
    solution_matrix[0] = u_vect

    # Check boundary conditions
    if bound_cond == 'dirichlet' or bound_cond == 'vary_p' or bound_cond == 'vary_q':
        A = np.diag([1-2*lmbda] * (mx - 1)) + np.diag([lmbda] * (mx - 2), -1) + np.diag([lmbda] * (mx - 2), 1)
        additive_vector = np.zeros(mx - 1)

    elif bound_cond == 'neumann':
        A = np.diag([1-2*lmbda] * (mx + 1)) + np.diag([lmbda] * mx, -1) + np.diag([lmbda] * mx, 1)
        A[0, 1] = 2*A[0, 1]
        A[-1, -2] = 2*A[-1, -2]
        deltax = L/mx
        additive_vector = np.zeros(mx+1)

    elif bound_cond == 'periodic':
        A = np.diag([1-2*lmbda] * mx) + np.diag([lmbda] * (mx - 1), -1) + np.diag([lmbda] * (mx - 1), 1)
        A[0, -1] = lmbda
        A[-1, 0] = lmbda
        solution_matrix = np.zeros((mt, mx))
        solution_matrix[0] = u_vect[:-1]

    elif bound_cond == 'homogenous':
        A = np.diag([1-2*lmbda] * (mx - 1)) + np.diag([lmbda] * (mx - 2), -1) + np.diag([lmbda] * (mx - 2), 1)

    # Iterate over time values
    for i in range(0, mt-1):
        if bound_cond == 'dirichlet':
            additive_vector[0] = p_func(i)
            additive_vector[-1] = q_func(i)
            solution_matrix[i+1][1:-1] = np.dot(A, solution_matrix[i][1:-1]) + lmbda*additive_vector
            solution_matrix[i+1][0] = additive_vector[0]
            solution_matrix[i+1][-1] = additive_vector[-1]
        elif bound_cond == 'neumann':
            additive_vector[0] = -p_func(i)
            additive_vector[-1] = q_func(i)
            solution_matrix[i+1] = np.dot(A, solution_matrix[i]) + 2*deltax*lmbda*additive_vector

        elif bound_cond == 'periodic':
            solution_matrix[i+1] = np.dot(A, solution_matrix[i])

        else:
            solution_matrix[i+1][1:-1] = np.dot(A, solution_matrix[i][1:-1])

    return solution_matrix


def backward_euler(pde, L, lmbda, mx, mt, bound_cond, p_func, q_func):
    """
    Solves a given pde using the backward euler method

    Parameters:
    ----------
        pde : function
            The PDE to be solved
        L : float
            The length of the spatial domain
        lmbda : float
            Mesh fourier number
        mx : integer
            Number of gridpoints in space
        mt : integer
            Number of gridpoints in time
        bound_cond : string
            boundary condition type for the PDE
        p_func : function
            The value at u(0, t)
        q_func : function
            The value at u(L, t)

    Returns:
    ----------
        solution_matrix : numpy array
            Matrix of all data points in the space at any given time
    """

    # Evaluate initial solution values
    u_vect = np.linspace(0, L, mx + 1)
    for i in range(mx+1):
        u_vect[i] = pde(u_vect[i], L)

    # Initialise the matrix of solutions with the initial solution values
    solution_matrix = np.zeros((mt, mx+1))
    solution_matrix[0] = u_vect

    # Check boundary conditions
    if bound_cond == 'dirichlet':
        A = np.diag([1+2*lmbda] * (mx - 1)) + np.diag([-lmbda] * (mx - 2), -1) + np.diag([-lmbda] * (mx - 2), 1)
        additive_vector = np.zeros(mx - 1)

    elif bound_cond == 'neumann':
        A = np.diag([1+2*lmbda] * (mx + 1)) + np.diag([-lmbda] * mx, -1) + np.diag([-lmbda] * mx, 1)
        A[0, 1] = 2*A[0, 1]
        A[-1, -2] = 2*A[-1, -2]
        deltax = L/mx
        additive_vector = np.zeros(mx + 1)

    elif bound_cond == 'periodic':
        A = np.diag([1+2*lmbda] * mx) + np.diag([-lmbda] * (mx - 1), -1) + np.diag([-lmbda] * (mx - 1), 1)
        A[0, -1] = -lmbda
        A[-1, 0] = -lmbda
        solution_matrix = np.zeros((mt, mx))
        solution_matrix[0] = u_vect[:-1]

    else:
        A = np.diag([1+2*lmbda] * (mx - 1)) + np.diag([-lmbda] * (mx - 2), -1) + np.diag([-lmbda] * (mx - 2), 1)

    # Iterate over time values
    for i in range(0, mt-1):
        if bound_cond == 'dirichlet':
            additive_vector[0] = p_func(i)
            additive_vector[-1] = q_func(i)
            solution_matrix[i+1][1:-1] = np.linalg.solve(A, solution_matrix[i][1:-1] + lmbda*additive_vector)
            solution_matrix[i+1][0] = additive_vector[0]
            solution_matrix[i+1][-1] = additive_vector[-1]
        elif bound_cond == 'neumann':
            additive_vector[0] = -p_func(i)
            additive_vector[-1] = q_func(i)
            solution_matrix[i+1] = np.linalg.solve(A, solution_matrix[i] + 2*deltax*lmbda*additive_vector)
        elif bound_cond == 'periodic':
            solution_matrix[i+1] = np.linalg.solve(A, solution_matrix[i])
        else:
            solution_matrix[i+1][1:-1] = np.linalg.solve(A, solution_matrix[i][1:-1])
    return solution_matrix


def crank_nicholson(pde, L, lmbda, mx, mt, bound_cond, p_func, q_func):
    """
    Solves a given pde using the Crank Nicholson method

    Parameters:
    ----------
        pde : function
            The PDE to be solved
        L : float
            The length of the spatial domain
        lmbda : float
            Mesh fourier number
        mx : integer
            Number of gridpoints in space
        mt : integer
            Number of gridpoints in time
        bound_cond : string
            boundary condition type for the PDE
        p_func : function
            The value at u(0, t)
        q_func : function
            The value at u(L, t)

    Returns:
    ----------
        solution_matrix : numpy array
            Matrix of all data points in the space at any given time
    """

    # Evaluate initial solution values
    u_vect = np.linspace(0, L, mx + 1)
    for i in range(mx+1):
        u_vect[i] = pde(u_vect[i], L)

    # Initialise the matrix of solutions with the initial solution values
    solution_matrix = np.zeros((mt, mx+1))
    solution_matrix[0] = u_vect

    # Check boundary conditions
    if bound_cond == 'dirichlet':
        A = np.diag([1+lmbda] * (mx - 1)) + np.diag([-lmbda/2] * (mx - 2), -1) + np.diag([-lmbda/2] * (mx - 2), 1)
        B = np.diag([1-lmbda] * (mx - 1)) + np.diag([lmbda/2] * (mx - 2), -1) + np.diag([lmbda/2] * (mx - 2), 1)
        additive_vector = np.zeros(mx - 1)

    elif bound_cond == 'neumann':
        A = np.diag([1+lmbda] * (mx + 1)) + np.diag([-lmbda/2] * mx, -1) + np.diag([-lmbda/2] * mx, 1)
        B = np.diag([1-lmbda] * (mx + 1)) + np.diag([lmbda/2] * mx, -1) + np.diag([lmbda/2] * mx, 1)
        A[0, 1] = 2*A[0, 1]
        A[-1, -2] = 2*A[-1, -2]
        B[0, 1] = 2*B[0, 1]
        B[-1, -2] = 2*B[-1, -2]
        deltax = L/mx
        additive_vector = np.zeros(mx + 1)

    elif bound_cond == 'periodic':
        A = np.diag([1+lmbda] * mx) + np.diag([-lmbda/2] * (mx - 1), -1) + np.diag([-lmbda/2] * (mx - 1), 1)
        B = np.diag([1-lmbda] * mx) + np.diag([lmbda/2] * (mx - 1), -1) + np.diag([lmbda/2] * (mx - 1), 1)
        A[0, -1] = -lmbda/2
        A[-1, 0] = -lmbda/2
        B[0, -1] = lmbda/2
        B[-1, 0] = lmbda/2
        solution_matrix = np.zeros((mt, mx))
        solution_matrix[0] = u_vect[:-1]

    else:
        A = np.diag([1+lmbda] * (mx - 1)) + np.diag([-lmbda/2] * (mx - 2), -1) + np.diag([-lmbda/2] * (mx - 2), 1)
        B = np.diag([1-lmbda] * (mx - 1)) + np.diag([lmbda/2] * (mx - 2), -1) + np.diag([lmbda/2] * (mx - 2), 1)

    # Iterate over time values
    for i in range(0, mt-1):
        if bound_cond == 'dirichlet':
            additive_vector[0] = p_func(i)
            additive_vector[-1] = q_func(i)
            solution_matrix[i+1][1:-1] = np.linalg.solve(A, np.dot(B, solution_matrix[i][1:-1]) + lmbda*additive_vector)
            solution_matrix[i+1][0] = additive_vector[0]
            solution_matrix[i+1][-1] = additive_vector[-1]
        elif bound_cond == 'neumann':
            additive_vector[0] = -p_func(i)
            additive_vector[-1] = q_func(i)
            solution_matrix[i+1] = np.linalg.solve(A, np.dot(B, solution_matrix[i]) +
                                                   2*deltax*lmbda*additive_vector)
        elif bound_cond == 'periodic':
            solution_matrix[i+1] = np.linalg.solve(A, np.dot(B, solution_matrix[i]))
        else:
            solution_matrix[i+1][1:-1] = np.linalg.solve(A, np.dot(B, solution_matrix[i][1:-1]))
    return solution_matrix


def pde_solver(pde, L, T, method, bound_cond, p_func, q_func, args):
    """
    Uses a given finite difference method to solve a given PDE

    Parameters:
    ----------
        pde : function
            The PDE to be solved
        L : float
            The length of the spatial domain
        lmbda : float
            Mesh fourier number
        method : function
            The method to solve the PDE
        bound_cond : string
            boundary condition type for the PDE
        p_func : function
            The value at u(0, t)
        q_func : function
            The value at u(L, t)
        args : list
            arguments required for given PDE (E.g for heat diffusion this includes kappa)
    Returns:
    ----------
        u_vect : numpy array
            Final solution vector at time, T
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

    # Solve PDE
    solution_matrix = method(pde, L, lmbda, mx, mt, bound_cond, p_func, q_func)
    return solution_matrix


def find_steady_state(solution_matrix):
    steady_state = None
    for i in range(len(solution_matrix)-1):
        if np.allclose(solution_matrix[i], solution_matrix[i+1], rtol=1e-05):
            steady_state = solution_matrix[i]
            break
    return steady_state


if __name__ == '__main__':
    # Set problem parameters/functions
    k = 3.0   # diffusion constant
    L = 1.0         # length of spatial domain
    T = 0.5         # total time to solve for

    # Boundary examples
    # homogenous
    # dirichlet
    # neumann
    # periodic
    boundary_cond1 = 'homogenous'
    boundary_cond2 = 'dirichlet'
    boundary_cond3 = 'neumann'
    boundary_cond4 = 'periodic'

    pde_solver(u_I, L, T, backward_euler, boundary_cond2, p, q, np.array([k]))
    pde_solver(u_I, L, T, forward_euler, boundary_cond2, p, q, np.array([k]))
    pde_solver(u_I, L, T, crank_nicholson, boundary_cond2, p, q, np.array([k]))
