import numpy as np
from scipy.integrate import odeint, solve_ivp


def integrate_dynamics(W, gains, params, initial_cond):
    
    params['gains'] = gains
    def rate_dynamics_ode (X, t, W, params)
        return params['over_tau'] * (-X + np.dot(W, params.f(X, params)))

    # Add noise to the initial condition if indicated
    if params['initial_cond_noise'] != 0:
        initial_cond = initial_cond + np.random.normal(scale=params['initial_cond_noise'], size=initial_cond.shape)

    # Solve the ODEs governing the neuronal dynamics,
    t_span = [0, params['tfinal']]
    t_eval = np.linspace(*t_span, params['n_timepoints'])
    # fun, t_span, y0
    result = solve_ivp(rate_dynamics_ode, t_span, initial_cond, args=(W, params), t_eval=t_eval)
    X = result.y

    # Convert neuronal activities into firing rates
    R = params.ff(X, params)

    return {'t': t, 'R': R, 'param': params}

# # Example usage:
# # Assuming you've already loaded the data and the function integrate_dynamics

# # Call the function
# output = integrate_dynamics(data['W_rec'], np.ones(len(data['W_rec'])), data['params'], data['initial_cond'])

# # Print the output or perform further operations
# print(output)
