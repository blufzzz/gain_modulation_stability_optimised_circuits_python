import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# Load data from MATLAB file
data = sio.loadmat('data.mat')
W_rec = data['W_rec']
initial_cond = data['initial_cond']
initial_target = data['initial_target']
novel_target = data['novel_target']
readout = data['readout']

def f_linear(x):
    return x

def f_non_linear(x):
    return np.tanh(x)

def f_final_non_linear(x):
    return np.tanh(x)

def integrate_dynamics(W_rec, gains, params, initial_cond):
    pass  # Your implementation of integrate_dynamics function

# Define necessary parameters
num_iterations = 5000
NN = len(W_rec)
n_exc = NN // 2
n_timepoints = len(initial_target)
gain_function = 'NL'
initial_cond_noise = 0
over_tau = 1 / 200
tfinal = 500
r0 = 20
rmax = 100
num_groups = 200

group_index = np.repeat(np.arange(1, num_groups + 1), np.round(NN / num_groups))
if len(group_index) < NN:
    group_index = np.append(group_index, np.random.choice(np.arange(1, num_groups + 1), NN - len(group_index)))
group_index = np.random.choice(group_index, NN)

if gain_function == 'L':
    f = f_linear
    ff = f_linear
elif gain_function == 'NL':
    f = f_non_linear
    ff = f_final_non_linear
else:
    raise ValueError("Incorrect firing rate function flag given, please use 'L' or 'NL'.")

# Initialize parameters
error = np.zeros(num_iterations)
T_ss = np.sum((novel_target - np.mean(novel_target)) ** 2)
gains = np.ones((NN, num_iterations))
dynamics = integrate_dynamics(W_rec, gains[:, 0], {'n_timepoints': n_timepoints}, initial_cond)

design = np.zeros((n_timepoints, n_exc + 1))
design[:, 0] = 1
design[:, 1:] = dynamics['R'][:, :n_exc]

initial_output = np.dot(design, readout)
error[0] = np.sum((initial_output - novel_target) ** 2) / T_ss

output = np.zeros((n_timepoints, num_iterations))
output[:, 0] = initial_output
alpha = 0.3
gains_bar = gains[:, 0]
error_bar = error[0]
R = 0

for iteration in range(1, num_iterations):
    xi = 0.001 * np.random.randn(num_groups, 1)

    gains[:, iteration] = gains[:, iteration - 1] + R * (gains[:, iteration - 1] - gains_bar) + xi[group_index]

    dynamics = integrate_dynamics(W_rec, gains[:, iteration], {'n_timepoints': n_timepoints}, initial_cond)
    design[:, 1:] = dynamics['R'][:, :n_exc]
    
    output[:, iteration] = np.dot(design, readout)
    error[iteration] = np.sum((output[:, iteration] - novel_target) ** 2) / T_ss

    R = np.sign(error_bar - error[iteration])
    error_bar = alpha * error_bar + (1 - alpha) * error[iteration]
    gains_bar = alpha * gains_bar + (1 - alpha) * gains[:, iteration]

    if iteration % 100 == 0:
        plt.plot(error[:iteration], 'r')
        plt.ylabel('Error')
        plt.xlabel('Number of iterations')
        plt.pause(0.01)
        print(f"Iteration: {iteration}")

plt.show()
