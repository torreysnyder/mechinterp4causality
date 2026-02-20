import argparse
import random
import numpy as np
import hashlib

def compute_total_effects(weights):
    n_nodes = weights.shape[0]
    weights = weights.astype(float)
    total_effects_transposed = np.eye(n_nodes, dtype=float)
    for j in range(n_nodes):
        for i in range(j):
            effect_i_on_j = 0.0
            for k in range(i, j):
                if weights[k, j] != 0:
                    effect_i_on_j += total_effects_transposed[i, k] * weights[k, j]
            total_effects_transposed[i, j] = effect_i_on_j
    total_effects = total_effects_transposed.T
    return total_effects

def gaussian_scm_inference(weights, observation, intervention, noise_var):
    n_nodes = weights.shape[0]
    weights = weights.astype(float)
    total_effects = compute_total_effects(weights)
    obs_indices = [i for i, val in enumerate(observation) if not np.isnan(val)]
    obs_vec = observation[obs_indices]
    obs_mat = total_effects[obs_indices, :]
    prior_mean = np.zeros(n_nodes)
    prior_cov = np.eye(n_nodes) * noise_var
    if len(obs_indices) > 0:
        sigma_uu = prior_cov
        sigma_xx = obs_mat @ sigma_uu @ obs_mat.T
        sigma_ux = sigma_uu @ obs_mat.T
        sigma_xx_stable = sigma_xx + np.eye(sigma_xx.shape[0]) * 1e-9
        k_transpose = np.linalg.solve(sigma_xx_stable, sigma_ux.T)
        kalman_gain = k_transpose.T
        residual = obs_vec - obs_mat @ prior_mean
        posterior_mean = prior_mean + kalman_gain @ residual
        posterior_cov = sigma_uu - kalman_gain @ obs_mat @ sigma_uu
    else:
        posterior_mean = prior_mean
        posterior_cov = prior_cov

    cf_means = np.zeros(n_nodes)
    for i in range(n_nodes):
        if not np.isnan(intervention[i]):
            cf_means[i] = intervention[i]
        else:
            cf_means[i] = posterior_mean[i] + weights[i, i]
            for j in range(i):
                cf_means[i] += weights[j, i] * cf_means[j]
    mod_noise_cov = posterior_cov.copy()
    for i in range(n_nodes):
        if not np.isnan(intervention[i]):
            mod_noise_cov[i, :] = 0.0
            mod_noise_cov[:, i] = 0.0
            mod_noise_cov[i, i] = 0.0
    cf_weights = weights.copy()
    for i in range(n_nodes):
        if not np.isnan(intervention[i]):
            cf_weights[:, i] = 0.0

    cf_total_effects = compute_total_effects(cf_weights)
    cf_cov = cf_total_effects @ mod_noise_cov @ cf_total_effects.T
    cf_cov = (cf_cov + cf_cov.T) / 2 + np.eye(n_nodes) * 1e-9
    diag_vals = np.diag(cf_cov)
    diag_vals = np.maximum(diag_vals, 1e-9)
    np.fill_diagonal(cf_cov, diag_vals)
    return {
        'means': cf_means,
        'stds': np.sqrt(np.diag(cf_cov))
    }

def parse_prompt(line):
    tokens = line.strip().split()
    assert tokens[0] == 'INFERENCE', f"Expected INFERENCE, got {tokens[0]}"
    scm_index = ' '.join(tokens[1:5])
    observations = {}
    interventions = {}
    query_var = None
    VARS = {'V1', 'V2', 'V3', 'V4'}
    i = 5
    while i < len(tokens):
        if tokens[i] == 'OBS':
            var = tokens[i + 1]
            val = float(tokens[i + 2])
            observations[var] = val
            i += 3
        elif tokens[i] == 'DO':
            var = tokens[i + 1]
            val = float(tokens[i + 2])
            interventions[var] = val
            i += 3
        elif tokens[i] in VARS:
            query_var = tokens[i]
            i += 1
        else:
            i += 1
    return scm_index, observations, interventions, query_var

def scm_index_to_weights(scm_index, noise_var=0.1):
    seed = int(hashlib.md5(scm_index.encode()).hexdigest(), 16) % (2 ** 31)
    rng = np.random.RandomState(seed)
    weights = np.zeros((4, 4))
    for i in range(4):
        weights[i, i] = rng.choice([-1, 0, 1])
    for i in range(4):
        for j in range(i + 1, 4):
            weights[i, j] = rng.choice([-1, 0, 1])
    return weights

def generate_accurate_values(input_file, output_file, noise_var=0.1):
    print(f"Processing {input_file}...")
    with open(input_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    var_to_idx = {'V1': 0, 'V2': 1, 'V3': 2, 'V4': 3}
    output_lines = []
    pair_count = 0
    for i in range(0, len(lines), 2):
        clean_line = lines[i]
        pair_count += 1
        scm_index, observations, interventions, query_var = parse_prompt(clean_line)
        assert query_var is not None, f"No query variable found in line {i+1}: {clean_line}"
        observation = np.full(4, np.nan)
        intervention = np.full(4, np.nan)
        for var, val in observations.items():
            observation[var_to_idx[var]] = val
        for var, val in interventions.items():
            intervention[var_to_idx[var]] = val
        weights = scm_index_to_weights(scm_index, noise_var)
        results = gaussian_scm_inference(weights, observation, intervention, noise_var)
        q_idx = var_to_idx[query_var]
        mean = results['means'][q_idx]
        std = results['stds'][q_idx]
        output_lines.append(f"{mean:.4f} {std:.4f}")
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines) + '\n')
    print(f"Wrote values to {output_file}")
    return pair_count

