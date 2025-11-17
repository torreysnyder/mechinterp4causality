import torch
import causal_transformers as ct
import time
import hydra


def compute_total_effects(weights):
    n_nodes = weights.shape[0]
    weights = weights.float()
    total_effects_transposed = torch.eye(n_nodes, dtype=weights.dtype, device=weights.device)
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
    weights = weights.float()
    assert weights.shape[0] == weights.shape[1], "weights must be square matrix"
    assert observation.shape[0] == n_nodes, "observation must have same size as weights"
    assert intervention.shape[0] == n_nodes, "intervention must have same size as weights"
    assert noise_var > 0, "noise variance must be positive"

    total_effects = compute_total_effects(weights)
    obs_indices = [i for i, val in enumerate(observation) if not torch.isnan(val)]
    obs_vec = observation[obs_indices]
    obs_mat = total_effects[obs_indices, :]
    prior_mean = torch.zeros(n_nodes, device=weights.device)
    prior_cov = torch.eye(n_nodes, device=weights.device) * noise_var
    if len(obs_indices) > 0:
        sigma_uu = prior_cov
        sigma_xx = obs_mat @ sigma_uu @ obs_mat.T
        sigma_ux = sigma_uu @ obs_mat.T
        sigma_xx_stable = sigma_xx + torch.eye(sigma_xx.shape[0], device=weights.device) * 1e-9
        k_transpose = torch.linalg.solve(sigma_xx_stable, sigma_ux.T)
        kalman_gain = k_transpose.T
        residual = obs_vec - obs_mat @ prior_mean
        posterior_mean = prior_mean + kalman_gain @ residual
        posterior_cov = sigma_uu - kalman_gain @ obs_mat @ sigma_uu
    else:
        posterior_mean = prior_mean
        posterior_cov = prior_cov
    cf_means = torch.zeros(n_nodes)
    for i in range(n_nodes):
        if not torch.isnan(intervention[i]):
            cf_means[i] = intervention[i]
        else:
            cf_means[i] = posterior_mean[i] + weights[i, i]
            for j in range(i):
                cf_means[i] += weights[j, i] * cf_means[j]
    mod_noise_cov = posterior_cov.clone()
    for i in range(n_nodes):
        if not torch.isnan(intervention[i]):
            mod_noise_cov[i, :] = 0.0
            mod_noise_cov[:, i] = 0.0
            mod_noise_cov[i, i] = 0.0
    cf_weights = weights.clone()
    for i in range(n_nodes):
        if not torch.isnan(intervention[i]):
            cf_weights[i] = 0.0
    cf_total_effects = compute_total_effects(cf_weights)
    cf_cov = cf_total_effects @ mod_noise_cov @ cf_total_effects.T
    cf_cov = (cf_cov + cf_cov.T)/2 + torch.eye(n_nodes, device=weights.device) * 1e-9
    diagonal_view = cf_cov.diagonal()
    diagonal_view.clamp_(min=1e-9)
    return {
        'means': cf_means,
        'cov': cf_cov,
        'vars': torch.diag(cf_cov),
        'stds': torch.sqrt(torch.diag(cf_cov)),
        'noise_mean': posterior_mean,
        'noise_cov': posterior_cov,
        'total_effects': total_effects
    }


def run_test_case(name, weights, observation, intervention, noise_var, expected_results):
    print(f"--- Test Case: {name} ---")
    print("Inputs:")
    print(f" Weights:\n{weights.numpy()}")
    print(f" Observation: {observation.numpy()}")
    print(f" Intervention: {intervention.numpy()}")
    print(f" Noise Variance: {noise_var}")
    results = gaussian_scm_inference(weights, observation, intervention, noise_var)
    print("\nExpected Results")
    for key, val in expected_results.items():
        print(f" {key}:\n{val.numpy()}")
    print("\nActual Results")
    for key in expected_results.keys():
        print(f" {key}:\n{results[key].detach().numpy()}")
    print("\nVerification:")
    all_match = True
    for key, expected_val in expected_results.items():
        actual_val = results[key]
        match = torch.allclose(actual_val, expected_val, atol=1e-3)
        print(f" {key}: {'Match' if match else 'MISMATCH'}")
        if not match:
            all_match = False
            print(f" Expected: {expected_val.numpy()}")
            print(f" Actual: {actual_val.detach().numpy()}")
    print(f"\nOverall Result: {'PASSED' if all_match else 'FAILED'}")
    print("-" * (len(name) + 18))
    return results


def T(data):
    return torch.tensor(data, dtype=torch.float32)


@hydra.main(version_base=None, config_path=ct.paths.CONFIG_PATH, config_name="config")
def main(cfg):
    scm_index = 12002
    weights = ct.utils.get_world_weights(cfg)[scm_index]
    n_nodes = weights.shape[0]
    observation = torch.full((n_nodes,), float('nan'))
    intervention = torch.full((n_nodes,), float('nan'))
    intervention[2] = -1.5
    results = gaussian_scm_inference(weights, observation, intervention, noise_var=cfg.dataset.world.noise_var, get_cov=True)
    mean = results['means']
    cov = results['cov']
    print(f"Posterior mean: {mean.numpy()}")
    print(f"Posterior covariance: {cov.numpy()}")
    sample = torch.distributions.MultivariateNormal(mean, cov).sample((1,))
    print(f"Sample under intervention {intervention.numpy()}: {sample.numpy()}")
    print()
    print()
    observation = torch.full((n_nodes,), float('nan'))
    intervention = torch.full((n_nodes,), float('nan'))
    observation[2] = 2.8
    intervention[1] = -0.3
    results = gaussian_scm_inference(weights, observation, intervention, noise_var=cfg.dataset.world.noise_var, get_cov=True)
    print(f"Posterior mean: {results['means'].numpy()}")
    print(f"Posterior stds: {results['stds'].numpy()}")


if __name__ == "__main__":
    main()


