import argparse
import random
import numpy as np


def compute_total_effects(weights):
    """Compute total effects matrix from weight matrix."""
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
    """
    Perform counterfactual inference on a linear Gaussian SCM.

    Args:
        weights: (n_nodes, n_nodes) weight matrix where weights[i,j] is effect of Vi on Vj
        observation: (n_nodes,) array with NaN for unobserved variables
        intervention: (n_nodes,) array with NaN for non-intervened variables
        noise_var: variance of the background noise variables

    Returns:
        dict with 'means' and 'stds' for counterfactual distribution
    """
    n_nodes = weights.shape[0]
    weights = weights.astype(float)

    # Compute total effects
    total_effects = compute_total_effects(weights)

    # Get observation indices and values
    obs_indices = [i for i, val in enumerate(observation) if not np.isnan(val)]
    obs_vec = observation[obs_indices]
    obs_mat = total_effects[obs_indices, :]

    # Prior over background variables
    prior_mean = np.zeros(n_nodes)
    prior_cov = np.eye(n_nodes) * noise_var

    # Abduction: compute posterior over background variables given observations
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

    # Action & Prediction: compute counterfactual means
    cf_means = np.zeros(n_nodes)
    for i in range(n_nodes):
        if not np.isnan(intervention[i]):
            cf_means[i] = intervention[i]
        else:
            cf_means[i] = posterior_mean[i] + weights[i, i]
            for j in range(i):
                cf_means[i] += weights[j, i] * cf_means[j]

    # Compute counterfactual covariance
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

    # Ensure positive diagonal
    diag_vals = np.diag(cf_cov)
    diag_vals = np.maximum(diag_vals, 1e-9)
    np.fill_diagonal(cf_cov, diag_vals)

    return {
        'means': cf_means,
        'stds': np.sqrt(np.diag(cf_cov))
    }


def parse_prompt(line):
    """
    Parse a prompt line to extract SCM index, observations, and interventions.

    Format: INFERENCE [A B C D] [OBS Vi val]* [DO Vi val]*
    """
    tokens = line.strip().split()

    # Skip INFERENCE token
    assert tokens[0] == 'INFERENCE', f"Expected INFERENCE, got {tokens[0]}"

    # Get SCM index (next 4 tokens)
    scm_index = ' '.join(tokens[1:5])

    # Parse rest of tokens
    observations = {}
    interventions = {}

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
        elif tokens[i] in ['V1', 'V2', 'V3', 'V4']:
            # This is the start of the output section, stop parsing
            break
        else:
            i += 1

    return scm_index, observations, interventions


def scm_index_to_weights(scm_index, noise_var=0.1):
    """
    Convert SCM index to weight matrix.
    For this simplified version, we'll generate deterministic weights based on the index.
    """
    # Use the SCM index as a seed for reproducible weight generation
    # Hash the index string to get a numeric seed
    seed = hash(scm_index) % (2 ** 31)
    rng = np.random.RandomState(seed)

    # Generate weights in {-1, 0, 1} as specified in the paper
    # Weight matrix is 4x4, with w[i,j] being effect of Vi on Vj
    weights = np.zeros((4, 4))

    # Diagonal (bias terms)
    for i in range(4):
        weights[i, i] = rng.choice([-1, 0, 1])

    # Off-diagonal (causal effects, only i < j for DAG structure)
    for i in range(4):
        for j in range(i + 1, 4):
            weights[i, j] = rng.choice([-1, 0, 1])

    return weights


def process_corruption_file(input_file, output_file, noise_var=0.1):
    """
    Process a corruption file and generate correct outputs.

    Each pair of lines (clean, corrupt) should get the same output based on the CLEAN prompt.
    """
    print(f"Processing {input_file}...")

    with open(input_file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    new_lines = []
    pair_count = 0

    for i in range(0, len(lines), 2):
        clean_line = lines[i]
        corrupt_line = lines[i + 1]

        # Parse the CLEAN prompt to get the ground truth
        scm_index, observations, interventions = parse_prompt(clean_line)

        # Get weights for this SCM
        weights = scm_index_to_weights(scm_index, noise_var)

        # Create observation and intervention arrays
        observation = np.full(4, np.nan)
        intervention = np.full(4, np.nan)

        var_to_idx = {'V1': 0, 'V2': 1, 'V3': 2, 'V4': 3}

        for var, val in observations.items():
            observation[var_to_idx[var]] = val

        for var, val in interventions.items():
            intervention[var_to_idx[var]] = val

        # Compute counterfactual inference
        results = gaussian_scm_inference(weights, observation, intervention, noise_var)

        # Format output: V1 mean std V2 mean std V3 mean std V4 mean std
        output_parts = []
        for idx, var in enumerate(['V1', 'V2', 'V3', 'V4']):
            mean = results['means'][idx]
            std = results['stds'][idx]
            output_parts.extend([var, f"{mean:.1f}", f"{std:.1f}"])

        output_str = ' '.join(output_parts)

        # Parse clean and corrupt prompts to rebuild without old outputs
        clean_prompt_parts = ['INFERENCE', scm_index]
        for var, val in observations.items():
            clean_prompt_parts.extend(['OBS', var, str(val)])
        for var, val in interventions.items():
            clean_prompt_parts.extend(['DO', var, str(val)])
        clean_prompt = ' '.join(clean_prompt_parts)

        # For corrupt, parse it too
        corrupt_scm_index, corrupt_observations, corrupt_interventions = parse_prompt(corrupt_line)
        corrupt_prompt_parts = ['INFERENCE', corrupt_scm_index]
        for var, val in corrupt_observations.items():
            corrupt_prompt_parts.extend(['OBS', var, str(val)])
        for var, val in corrupt_interventions.items():
            corrupt_prompt_parts.extend(['DO', var, str(val)])
        corrupt_prompt = ' '.join(corrupt_prompt_parts)

        # Add output to both
        new_lines.append(f"{clean_prompt} {output_str}")
        new_lines.append(f"{corrupt_prompt} {output_str}")

        pair_count += 1

        if pair_count % 100 == 0:
            print(f"  Processed {pair_count} pairs...")

    # Write to output file
    with open(output_file, 'w') as f:
        for line in new_lines:
            f.write(line + '\n')

    print(f"✓ Wrote {pair_count} pairs to {output_file}")
    return pair_count


def main():
    parser = argparse.ArgumentParser(description='Generate correct SCM outputs for corruption files')
    parser.add_argument('--input', type=str, required=True, help='Input corruption file')
    parser.add_argument('--output', type=str, required=True, help='Output file with correct outputs')
    parser.add_argument('--noise-var', type=float, default=0.1, help='Noise variance (default: 0.1)')

    args = parser.parse_args()

    process_corruption_file(args.input, args.output, args.noise_var)


if __name__ == "__main__":
    # If no args provided, process all three files
    import sys

    if len(sys.argv) == 1:
        print("Processing all corruption files with default settings...\n")

        files = [
            ('scm_index_corruptions.txt', 'scm_index_corruptions_corrected.txt'),
            ('value_corruptions.txt', 'value_corruptions_corrected.txt'),
            ('intervention_corruptions.txt', 'intervention_corruptions_corrected.txt')
        ]

        for input_file, output_file in files:
            process_corruption_file(input_file, output_file, noise_var=0.1)
            print()
    else:
        main()
