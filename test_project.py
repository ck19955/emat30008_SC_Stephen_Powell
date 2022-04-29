# Testing the entirety of the project
from ode_solver import *
from function_examples import *
from numerical_shooting import *
from numerical_continuation import *


def exact_sol(exact_func, t_range, t_space, x, args=None):
    """
    Parameters:
    ----------
        exact_func : function
            The differential equation
        t_range : tuple
            The range of t values to solve over
        t_space : integer
            Number of data points in the time range
        x : array
            Initial value
        args : numpy array
            The parameters of the ODE

    Returns:
    ----------
        solution : numpy array
            Array of all data points over the range of time specified
    """
    if args is None:  # args cannot be passed through as None because this induces an error when trying to unpack
        args = []  # This allows for unpacking

    # Set up the solution array
    times = np.linspace(t_range[0], t_range[1], num=t_space)
    solution = np.zeros((t_space, len(x)))

    # For each time value, find the exact solution
    for i in range(t_space):
        solution[i] = exact_func(times[i], x, *args)
    return solution


# Test ode euler solutions
def test_euler():
    times = np.linspace(0, 20, num=100)
    euler_values = np.asarray(solve_ode(times, np.array([3, 4]), 0.01, euler_step, ode_second_order, []))
    exact_values = exact_sol(exact_second_order, [0, 20], 100, np.array([3, 4]))
    assert np.allclose(euler_values, exact_values, rtol=1)  # Check whether the two arrays are similar


# Test ode rk4 solutions
def test_rk4():
    times = np.linspace(0, 20, num=100)
    rk4_values = np.asarray(solve_ode(times, np.array([3, 4]), 0.01, rk4, ode_second_order, []))
    exact_values = exact_sol(exact_second_order, [0, 20], 100, np.array([3, 4]))
    assert np.allclose(rk4_values, exact_values, rtol=1)  # Check whether the two arrays are similar


# Test ode rk4 solutions
def test_improved_euler():
    times = np.linspace(0, 20, num=100)
    improved_euler_values = np.asarray(solve_ode(times, np.array([3, 4]), 0.01, improved_euler_step, ode_second_order))
    exact_values = exact_sol(exact_second_order, [0, 20], 100, np.array([3, 4]))
    assert np.allclose(improved_euler_values, exact_values, rtol=1)  # Check whether the two arrays are similar


# Test ode rk4 solutions
def test_heun():
    times = np.linspace(0, 20, num=100)
    heun_values = np.asarray(solve_ode(times, np.array([3, 4]), 0.01, heun_step, ode_second_order))
    exact_values = exact_sol(exact_second_order, [0, 20], 100, np.array([3, 4]))
    assert np.allclose(heun_values, exact_values, rtol=1)  # Check whether the two arrays are similar


# Test orbit isolator
def test_orbit():
    times = np.linspace(0, 400, num=1000)
    solution_values = np.asarray(solve_ode(times, np.array([1, 1]), 0.1, rk4, pred_prey, np.array([1, 0.2, 0.1])))
    initial_orbit_values = isolate_orbit(solution_values, times)
    data_values = np.asarray(solve_ode(times, initial_orbit_values[:-1], 0.1, rk4, pred_prey, np.array([1, 0.2, 0.1])))
    orbit_index = int(initial_orbit_values[-1]/(400/1000))

    # Check of isolate orbit gives an orbit
    assert np.allclose(initial_orbit_values[:-1], data_values[orbit_index][:-1], rtol=1)


# Test shooting
def test_shooting():
    # Check whether the orbit found from shooting is similar to the isolated orbit
    args = np.array([0.04])
    times = np.linspace(0, 400, num=1000)
    solution_values = np.asarray(solve_ode(times, np.array([0.5, 0.5]), 0.1, rk4, hopf_bif, args))
    initial_orbit_values = isolate_orbit(solution_values, times)
    shooting_orbit_values = shooting(hopf_bif, initial_orbit_values, shooting_conditions, False, True, args)
    assert np.allclose(initial_orbit_values[-1], shooting_orbit_values[-1], rtol=1)


# Test hopf bifurcation
def test_hopf():
    # Set up parameters
    args = np.array([0.5])
    times = np.linspace(0, 400, num=1000)

    # Find solution to the hopf bifurcation
    rk4_sol = np.asarray(solve_ode(times, [0.5, 0.5], 0.01, rk4, hopf_bif, args))

    # Isolate an orbit
    isolated_orbit = isolate_orbit(rk4_sol, times)

    # Find the orbit using numerical shooting
    initial_orb_cond = shooting(hopf_bif, isolated_orbit, shooting_conditions, [], True, args)

    # Find the exact solution of the hopf bifurcation using the initial conditions of the orbit
    exact_solution = exact_sol(exact_hopf_bif, [0, 400], 1000, initial_orb_cond[:-1], np.array([0.5, 2*np.pi]))

    # Solve the ode using the initial conditions of the orbit as the initial values
    predicted_sol = np.asarray(solve_ode(times, initial_orb_cond[:-1], 0.01, rk4, hopf_bif, args))

    assert np.allclose(initial_orb_cond[-1], 2*np.pi)  # Check the orbit is correct
    assert np.allclose(exact_solution[1:], predicted_sol[1:], atol=1e-3)  # Check the dependant variables are correct


# Test pde forward euler
def test_forward_euler():
    # Define necessary parameters
    L = 1  # final space values]
    T = 0.5  # final time value
    boundary_cond = 'homogenous'
    k = 3  # thermal constant
    mx = 10  # number of x data points
    mt = 1000  # number of t data points

    # Solve the PDE
    solution_matrix = pde_solver(u_I, L, T, mx, mt, forward_euler, boundary_cond, p, q, np.array([k]))
    # Find the middle value of the matrix
    centre_solution = solution_matrix[int(mt/2), int(mx/2)]
    # Find the exact value for the middle data value
    exact_solution = u_exact(L/2, T/2, k, L)
    assert np.isclose(centre_solution, exact_solution, rtol=1)  # Compare solutions


# Test pde backward euler
def test_backward_euler():
    # Define necessary parameters
    L = 1  # final space values]
    T = 0.5  # final time value
    boundary_cond = 'homogenous'
    k = 3  # thermal constant
    mx = 10  # number of x data points
    mt = 1000  # number of t data points

    # Solve the PDE
    solution_matrix = pde_solver(u_I, L, T, mx, mt, backward_euler, boundary_cond, p, q, np.array([k]))
    # Find the middle value of the matrix
    centre_solution = solution_matrix[int(mt/2), int(mx/2)]
    # Find the exact value for the middle data value
    exact_solution = u_exact(L/2, T/2, k, L)
    assert np.isclose(centre_solution, exact_solution, rtol=1)  # Compare solutions


# Test pde crank nicholson
def test_crank_nicholson():
    # Define necessary parameters
    L = 1  # final space values]
    T = 0.5  # final time value
    boundary_cond = 'homogenous'
    k = 3  # thermal constant
    mx = 10  # number of x data points
    mt = 1000  # number of t data points

    # Solve the PDE
    solution_matrix = pde_solver(u_I, L, T, mx, mt, crank_nicholson, boundary_cond, p, q, np.array([k]))

    # Find the middle value of the matrix
    centre_solution = solution_matrix[int(mt/2), int(mx/2)]
    # Find the exact value for the middle data value
    exact_solution = u_exact(L/2, T/2, k, L)
    assert np.isclose(centre_solution, exact_solution, rtol=1)  # Compare solutions


