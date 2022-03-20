import numpy as np
import matplotlib.pyplot as plt
from ODE_Solver import *


def ode_num(t, x_values, a, b, d):
    x = x_values[0]
    y = x_values[1]
    x_array = np.array([x*(1-x) - (a*x*y)/(d+x), b*y*(1-(y/x))])
    return x_array


fig, axs = plt.subplots(2, 2)
times1 = np.linspace(0, 200, num=1000)
RK4_values_1 = np.asarray(solve_ode(times1, np.array([1, 1]), 0.1, RK4, ode_num, np.array([1, 0.15, 0.1])))
RK4_values_2 = np.asarray(solve_ode(times1, np.array([1, 1]), 0.1, RK4, ode_num, np.array([1, 0.25, 0.1])))
RK4_values_3 = np.asarray(solve_ode(times1, np.array([1, 1]), 0.1, RK4, ode_num, np.array([1, 0.27, 0.1])))
RK4_values_4 = np.asarray(solve_ode(times1, np.array([1, 1]), 0.1, RK4, ode_num, np.array([1, 0.5, 0.1])))


axs[0, 0].plot(RK4_values_1[:, 0])
axs[0, 0].plot(RK4_values_1[:, 1])
axs[0, 0].set_title('b = 0.15')
axs[0, 1].plot(RK4_values_2[:, 0])
axs[0, 1].plot(RK4_values_2[:, 1])
axs[0, 1].set_title('b = 0.25')
axs[1, 0].plot(RK4_values_3[:, 0])
axs[1, 0].plot(RK4_values_3[:, 1])
axs[1, 0].set_title('b = 0.27')
axs[1, 1].plot(RK4_values_4[:, 0])
axs[1, 1].plot(RK4_values_4[:, 1])
axs[1, 1].set_title('b = 0.5')
for ax in axs.flat:
    ax.set(xlabel='t', ylabel='u')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
plt.show()



