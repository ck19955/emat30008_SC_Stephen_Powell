# Solves ODEs
import math


def euler_step(x_n, t_n, step_size):
    x_n_1 = x_n + step_size * f(t_n, x_n)
    return x_n_1


def solve_to(x_0, t_0, t_end, deltaT_max):
    x = x_0
    t = t_0
    while t < t_end:
        if t + deltaT_max <= t_end:
            x = euler_step(x, t, deltaT_max)
            t = t + deltaT_max
        else:
            deltaT_max = t_end - t
    return x


def solve_ode(x_0, t_values, deltaT_max):
    x_values = [0] * len(t_values)
    x_values[0] = x_0
    for i in range(len(t_values)-1):
        x_values[i+1] = solve_to(x_values[i], t_values[i], t_values[i + 1], deltaT_max)
    return x_values


def f(t, x):
    return math.exp(t)


