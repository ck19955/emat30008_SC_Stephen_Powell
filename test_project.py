# Testing the entirety of the project
from ode_solver import *
from function_examples import *
from numerical_shooting import *
from numerical_continuation import *


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
    rk4_values = np.asarray(solve_ode(times, np.array([3, 4]), 0.01, rk4, ode_second_order, []))
    exact_values = exact_sol(exact_second_order, [0, 20], 100, np.array([3, 4]))
    assert np.allclose(rk4_values, exact_values, rtol=1)


# Test ode rk4 solutions
def test_improved_euler():
    times = np.linspace(0, 20, num=100)
    improved_euler_values = np.asarray(solve_ode(times, np.array([3, 4]), 0.01, improved_euler_step, ode_second_order, []))
    exact_values = exact_sol(exact_second_order, [0, 20], 100, np.array([3, 4]))
    assert np.allclose(improved_euler_values, exact_values, rtol=1)


# Test ode rk4 solutions
def test_heun():
    times = np.linspace(0, 20, num=100)
    heun_values = np.asarray(solve_ode(times, np.array([3, 4]), 0.01, heun_step, ode_second_order, []))
    exact_values = exact_sol(exact_second_order, [0, 20], 100, np.array([3, 4]))
    assert np.allclose(heun_values, exact_values, rtol=1)


# Test orbit isolator
def test_orbit():
    times = np.linspace(0, 400, num=1000)
    solution_values = np.asarray(solve_ode(times, np.array([1, 1]), 0.1, rk4, pred_prey, np.array([1, 0.2, 0.1])))
    initial_orbit_values = isolate_orbit(solution_values, times)
    data_values = np.asarray(solve_ode(times, initial_orbit_values[:-1], 0.1, rk4, pred_prey, np.array([1, 0.2, 0.1])))
    orbit_index = int(initial_orbit_values[-1]/(400/1000))
    assert np.allclose(initial_orbit_values[:-1], data_values[orbit_index][:-1], rtol=1)


# Test shooting
def test_shooting():
    # Can it find the orbit
    args = np.array([0.04])
    times = np.linspace(0, 400, num=1000)
    solution_values = np.asarray(solve_ode(times, np.array([0.5, 0.5]), 0.1, rk4, hopf_bif, args))
    initial_orbit_values = isolate_orbit(solution_values, times)
    shooting_orbit_values = shooting(hopf_bif, initial_orbit_values, False, True, args)
    assert np.allclose(initial_orbit_values[-1], shooting_orbit_values[-1], rtol=1)


# Test natural parameter
def test_natural_parameter():
    pass


# Test pseudo arclength
def test_pseudo_arclength():
    pass


# Test pde forward euler
def test_forward_euler():
    L = 1
    T = 0.5
    boundary_cond = 'homogenous'
    k = 3
    mx = 10
    mt = 1000
    solution_matrix = pde_solver(u_I, L, T, mx, mt, forward_euler, boundary_cond, p, q, np.array([k]))
    centre_solution = solution_matrix[int(mt/2), int(mx/2)]
    exact_solution = u_exact(L/2, T/2, k, L)
    assert np.isclose(centre_solution, exact_solution, rtol=1)


# Test pde backward euler
def test_backward_euler():
    L = 1
    T = 0.5
    boundary_cond = 'homogenous'
    k = 3
    mx = 10
    mt = 1000
    solution_matrix = pde_solver(u_I, L, T, mx, mt, backward_euler, boundary_cond, p, q, np.array([k]))
    centre_solution = solution_matrix[int(mt/2), int(mx/2)]
    exact_solution = u_exact(L/2, T/2, k, L)
    assert np.isclose(centre_solution, exact_solution, rtol=1)


# Test pde crank nicholson
def test_crank_nicholson():
    L = 1
    T = 0.5
    boundary_cond = 'homogenous'
    k = 3
    mx = 10
    mt = 1000
    solution_matrix = pde_solver(u_I, L, T, mx, mt, crank_nicholson, boundary_cond, p, q, np.array([k]))
    centre_solution = solution_matrix[int(mt/2), int(mx/2)]
    exact_solution = u_exact(L/2, T/2, k, L)
    assert np.isclose(centre_solution, exact_solution, rtol=1)


# Test steady state
def test_steady_states():
    pass


# Test pde continuation
def test_pde_continuation():
    pass


