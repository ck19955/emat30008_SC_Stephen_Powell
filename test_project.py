# Testing the entirety of the project
from ODE_Solver import *
from function_examples import *


def exact_sol(exact_func, t_range, t_space, x):
    times = np.linspace(t_range[0], t_range[1], num=t_space)
    solution = np.zeros((t_space, len(x)))
    for i in range(t_space):
        solution[i] = exact_func(times[i], x)
    return solution


# Test ode euler solutions
def test_euler():
    times = np.linspace(0, 20, num=100)
    euler_values = np.asarray(solve_ode(times, np.array([3, 4]), 0.01, euler_step, ode_second_order, []))
    exact_values = exact_sol(exact_second_order, [0, 20], 100, np.array([3, 4]))
    assert np.allclose(euler_values, exact_values, rtol=1)  # euler has low accuracy therefore a high tolerance is used


# Test ode rk4 solutions
def test_rk4():
    times = np.linspace(0, 20, num=100)
    rk4_values = np.asarray(solve_ode(times, np.array([3, 4]), 0.01, RK4, ode_second_order, []))
    exact_values = exact_sol(exact_second_order, [0, 20], 100, np.array([3, 4]))
    assert np.allclose(rk4_values, exact_values, rtol=1)


# Test orbit isolator


# Test shooting


# Test natural parameter


# Test pseudo arclength


# Test pde forward euler


# Test pde backward euler


# Test pde crank nicholson


# Test steady state


# Different boundary conditions


# Test pde continuation


