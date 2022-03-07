# Solves ODEs
import math
import numpy as np
import matplotlib.pyplot as plt
import time


def euler_step(t_n, x_n, step_size):
    x = x_n + step_size * ode(t_n, x_n)
    return x


def RK4(t_n, x_n, step_size):
    k1 = ode(t_n, x_n)
    k2 = ode(t_n + step_size/2, x_n + k1*(step_size/2))
    k3 = ode(t_n + step_size/2, x_n + k1*(step_size/2))
    k4 = ode(t_n + step_size, x_n + k3*step_size)
    x = x_n + ((k1 + 2*k2 + 2*k3 + k4)/6)*step_size
    return x


def solve_to(t_0, t_end, x_0, deltaT_max, method):
    x = x_0
    t = t_0
    while t < t_end:
        if t + deltaT_max <= t_end:
            x = method(t, x, deltaT_max)
            t += deltaT_max
        else:
            deltaT_max = t_end - t
    return x


def solve_ode(t_values, x_0, deltaT_max, method):
    x_values = [0] * len(t_values)
    x_values[0] = x_0
    for i in range(len(t_values)-1):
        x_values[i+1] = solve_to(t_values[i], t_values[i + 1], x_values[i], deltaT_max, method)
    return x_values


def ode(t, x):
    #x_array = np.array([x[1], -x[0]])
    #return x_array
    return x


def exact(t, x):
    #a = x[0]
    #b = x[1]
    #return np.array([a*math.cos(t) + b*math.sin(t), -a*math.sin(t) + b*math.cos(t)])
    return math.exp(t)


def error_plot(t_values, x_0):
    x_value = exact(t_values[1], 0)
    step_sizes = np.logspace(-6, 0, 10)
    error_eul = np.zeros(len(step_sizes))
    error_RK4 = np.zeros(len(step_sizes))
    error_match = 1e-2
    time_eul = 0
    time_RK4 = 0
    for i in range(len(step_sizes)):
        init_time = time.perf_counter()
        predict_eul = solve_ode(t_values, x_0, step_sizes[i], euler_step)
        time_1 = time.perf_counter() - init_time
        predict_RK4 = solve_ode(t_values, x_0, step_sizes[i], RK4)
        time_2 = time.perf_counter() - init_time
        error_eul[i] = abs(predict_eul[-1] - x_value)
        error_RK4[i] = abs(predict_RK4[-1] - x_value)
        if math.isclose(error_match, error_eul[i], abs_tol=1e-3):
            time_eul = time_1
        if math.isclose(error_match, error_RK4[i], abs_tol=1e-3):
            time_RK4 = time_2
    plt.loglog(step_sizes, error_eul, label='Euler Method')
    plt.loglog(step_sizes, error_RK4, label='RK4 Method')
    plt.show()
    return error_eul, error_RK4, time_eul, time_RK4


def plot_approx(t_values, x_values, step_size):
    RK4_values = np.asarray(solve_ode(t_values, x_values, step_size, RK4))
    print(RK4_values)
    euler_values = np.asarray(solve_ode(t_values, x_values, step_size, euler_step))
    print(euler_values)

    exact_values = []
    for i in range(len(t_values)):
        exact_values.append(exact(t_values[i], x_values))

    error_eul = abs(euler_values - exact_values)
    error_RK4 = abs(RK4_values - exact_values)
    print(error_eul)
    print(error_RK4)
    #plt.plot(t_values, error_eul, label='RK4 Method')
    #plt.plot(t_values, error_RK4, label='Euler Method')
    plt.plot([item[0] for item in exact_values], [item[1] for item in exact_values])
    plt.plot(RK4_values[:, 0], RK4_values[:, 1])
    plt.plot(euler_values[:, 0], euler_values[:, 1])

    plt.legend()
    plt.show()
    return


#times = np.linspace(0, 100, num=100)
times = [0, 1]
error_1, error_2, time_euler, time_RungeKutta = error_plot(times, 1)
# print(time_euler)
# print(time_RungeKutta)
#plot_approx(times, np.array([3, 4]), 0.1)
#print(solve_ode(times, [3, 4], 0.1, RK4))
#print(exact(10, [3, 4]))
