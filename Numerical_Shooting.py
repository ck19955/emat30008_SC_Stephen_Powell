import numpy as np
import matplotlib.pyplot as plt
from ODE_Solver import *
from scipy.signal import argrelextrema
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


def ode_num(t, x_values, a, b, d):
    x = x_values[0]
    y = x_values[1]
    x_array = np.array([x*(1-x) - (a*x*y)/(d+x), b*y*(1-(y/x))])
    return x_array


def isolate_orbit(ode_data, time_data):
    x_data = ode_data[:, 0]
    y_data = ode_data[:, 1]
    maximums = argrelextrema(x_data, np.greater)[0]
    previous_value = False
    previous_time = 0
    for i in maximums:
        if previous_value:
            if math.isclose(x_data[i], previous_value, abs_tol=1e-4):
                period = time_data[i] - previous_time
                return x_data[i], y_data[i], period
        previous_value = x_data[i]
        previous_time = time_data[i]
    return


def shooting_conditions(u0, args):
    x0 = u0[:-1]
    t0 = u0[-1]
    sol = solve_ivp(ode_num, (0,t0), x0, max_step=1e-2, args=args)
    x_condition = x0 - sol.y[:,-1]
    t_condition = np.asarray(ode_num(t0,x0,*args)[0])
    g_condition = np.concatenate((x_condition, t_condition), axis=None)
    return g_condition


times1 = np.linspace(0, 200, num=1000)
RK4_values = np.asarray(solve_ode(times1, np.array([1, 1]), 0.1, RK4, ode_num, np.array([1, 0.2, 0.1])))
print(isolate_orbit(RK4_values, times1))
args = np.array([1, 0.2, 0.1])
final = fsolve(shooting_conditions, np.array([1, 1, 20]), args=args)
print(final)


'''
plt.plot(times1, RK4_values[:, 0])
plt.plot(times1, RK4_values[:, 1])
plt.xlabel('time')
plt.ylabel('u')
plt.show()
'''

'''
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
'''



