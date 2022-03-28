import numpy as np
from math import pi

# Set problem parameters/functions
kappa = 1.0   # diffusion constant
L=3.0         # length of spatial domain
T=0.5         # total time to solve for


def u_I(x):
    # initial temperature distribution
    y = np.sin(pi*x/L)
    return y


def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y


# Set numerical parameters
mx = 10     # number of gridpoints in space
mt = 1000   # number of gridpoints in time

# Set up the numerical environment variables
x = np.linspace(0, L, mx+1)     # mesh points in space
t = np.linspace(0, T, mt+1)     # mesh points in time
deltax = x[1] - x[0]            # gridspacing in x
deltat = t[1] - t[0]            # gridspacing in t
lmbda = kappa*deltat/(deltax**2)    # mesh fourier number

# Set up the solution variables

A = np.diag([1-2*lmbda] * (mx - 1)) + np.diag([lmbda] * (mx - 2), -1) + np.diag([lmbda] * (mx - 2), 1)
# Solve the matrix equation to return the next value of u
x_vect = np.linspace(0, L, mx + 1)
x_vect = x_vect[1:-1]
u_vect = np.array(x_vect)
for i in range(len(x_vect)):
    u_vect[i] = u_I(u_vect[i])
for i in range(0, mt):
    u_vect = np.dot(A, u_vect)

print(np.linalg.solve(A, u_vect[:-1]))

