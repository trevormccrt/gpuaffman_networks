import cupy as cp
import numpy as np

import ragged_task_evolution, ragged_general_network


def sort_by_performance(input_state, functions, connectivity, used_connectivity, f_eval, n_trajectories=10000, n_timesteps=10, p_error=0.01, ):
    errors = np.squeeze(ragged_task_evolution.evaluate_populations(
        input_state, n_trajectories, functions, connectivity, used_connectivity, n_timesteps, p_error, f_eval), -1)
    perf_idx = np.argsort(errors)
    return functions[perf_idx], connectivity[perf_idx], used_connectivity[perf_idx], errors[perf_idx]


def run_dynamics_forward_save_state(input_state, functions, connections, used_connections, n_timesteps, p_noise):
    xp = np
    if isinstance(input_state, cp.ndarray):
        xp = cp
    states = np.tile(np.expand_dims(input_state, 0), (n_timesteps + 2, *([1] * len(input_state.shape))))
    states[1] = ragged_general_network.ragged_k_state_update(states[0], functions, connections, used_connections)
    noise = xp.random.binomial(1, p_noise, (n_timesteps, *input_state.shape)).astype(np.bool_)
    for i in range(n_timesteps):
        states[i+2] = ragged_general_network.ragged_k_state_update(np.bitwise_xor(states[i+1], noise[i]), functions, connections, used_connections)
    return states, noise


def generate_ft_curve(physical_error_rates, input_state, functions, connectivity, used_connectivity, f_eval, n_traj=5000, n_timesteps=10):
    logical_error_rates = []
    for error_rate in physical_error_rates:
        logical_error_rates.append(np.squeeze(ragged_task_evolution.evaluate_populations(input_state, n_traj, functions, connectivity, used_connectivity, n_timesteps, error_rate, f_eval), -1))
    return np.array(logical_error_rates)


def compute_influence(function):
    function_nbits = np.rint(np.log2(function.shape[-1])).astype(np.uint8)
    influences = np.zeros((*function.shape[:-1], function_nbits))
    indicies = np.arange(start=0, stop=function.shape[-1], step=1).astype(np.uint8)
    for to_flip in range(function_nbits):
        flip = np.array(1<<to_flip, dtype=np.uint8)
        shifted_ind = np.bitwise_xor(indicies, flip)
        inf = np.mean(function[..., indicies] != function[..., shifted_ind], axis=-1)
        influences[..., to_flip] = inf
    return influences


def death_prob_vs_time(input_state, functions, connections, used_connections, n_trajectories, p_noise, f_eval, t_start, t_end):
    xp = np
    if isinstance(input_state, cp.ndarray):
        xp = cp
    states, _ = run_dynamics_forward_save_state(np.broadcast_to(np.expand_dims(input_state, 0), (n_trajectories, *input_state.shape)), functions, connections, used_connections, t_end, p_noise)
    errors = []
    for state in states[t_start:]:
        errors.append(f_eval(state))
    return xp.array(errors)


