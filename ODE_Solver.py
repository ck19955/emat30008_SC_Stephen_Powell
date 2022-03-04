# Solves ODEs
import math


def euler_step(t_n, x_n, step_size):
    x = x_n + step_size * f(t_n, x_n)
    return x


def RK4(t_n, x_n, step_size):
    k1 = f(t_n, x_n)
    k2 = f(t_n + step_size/2, x_n + k1*(step_size/2))
    k3 = f(t_n + step_size/2, x_n + k1*(step_size/2))
    k4 = f(t_n + step_size, x_n + k3*step_size)
    x = x_n + ((k1 + 2*k2 + 2*k3 + k4)/6)*step_size
    return x


def solve_to(t_0, t_end, x_0, deltaT_max, method):
    x = x_0
    t = t_0
    while t < t_end:
        if t + deltaT_max <= t_end:
            x = method(t, x, deltaT_max)
            t = t + deltaT_max
        else:
            deltaT_max = t_end - t
    return x


def solve_ode(t_values, x_0, deltaT_max, method):
    x_values = [0] * len(t_values)
    x_values[0] = x_0
    for i in range(len(t_values)-1):
        x_values[i+1] = solve_to(t_values[i], t_values[i + 1], x_values[i], deltaT_max, method)
    return x_values


def f(t, x):
    return math.exp(t)


def error_plot(t_value, predict):
    x_value = f(t_value, 0)
    error = predict - x_value
    return error


time = [0, 1]
x_val = solve_ode(time, 1, 0.1, RK4)
errors = error_plot(time[1], x_val[1])

