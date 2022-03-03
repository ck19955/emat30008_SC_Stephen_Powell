# Solves ODEs

def euler_step(function, x_n, t_n, step_size):
    x_n_1 = x_n + step_size*function(t_n, x_n)
    return x_n_1


def solve_to(function, x_0, t_0, t_end, deltaT_max):
    x = x_0
    t = t_0
    while t < t_end:
        if t + deltaT_max <= t_end:
            x = euler_step(function, x, t, deltaT_max)
            t = t + deltaT_max
        else:
            deltaT_max = t_end - t
    return x


def solve_ode():
    pass
