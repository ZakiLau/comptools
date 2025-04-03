import numpy as np
import statsmodels.api as sm
from scipy.stats import norm

# simulate_regression_with_ar1_errors
def simulate_regression_with_ar1_errors(T, beta0, beta1, phi_x, phi_u, sigma):
    np.random.seed(0)
    x = np.zeros(T)
    u = np.zeros(T)
    x[0] = np.random.normal()
    u[0] = np.random.normal(scale=sigma)
    for t in range(1, T):
        x[t] = phi_x * x[t-1] + np.random.normal()
        u[t] = phi_u * u[t-1] + np.random.normal(scale=sigma)
    y = beta0 + beta1 * x + u
    return x, y, u

# bootstrap 
def moving_block_bootstrap(x, y, block_length, num_bootstrap):
    T = len(y)
    num_blocks = T // block_length
    beta_estimates = np.zeros((num_bootstrap, 2))
    for i in range(num_bootstrap):
        indices = np.random.randint(0, num_blocks, num_blocks)
        sample_x = np.concatenate([x[j*block_length:(j+1)*block_length] for j in indices])
        sample_y = np.concatenate([y[j*block_length:(j+1)*block_length] for j in indices])
        X = sm.add_constant(sample_x)
        model = sm.OLS(sample_y, X).fit()
        beta_estimates[i] = model.params
    return beta_estimates

##
T_values = [100, 500]
beta0_true = 1
beta1_true = 2
phi_x = 0.5
phi_u = 0.5
sigma = 1
block_length = 12
num_bootstrap = 1000
num_simulations = 1000
confidence_level = 0.95
z_score = norm.ppf(1 - (1 - confidence_level) / 2)

# monte Carlo 
for T in T_values:
    coverage_count_bootstrap = 0
    coverage_count_asymptotic = 0
    for _ in range(num_simulations):
        x, y, _ = simulate_regression_with_ar1_errors(T, beta0_true, beta1_true, phi_x, phi_u, sigma)
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        beta1_hat = model.params[1]
        se_asymptotic = model.bse[1]
        ci_asymptotic = (beta1_hat - z_score * se_asymptotic, beta1_hat + z_score * se_asymptotic)
        if ci_asymptotic[0] <= beta1_true <= ci_asymptotic[1]:
            coverage_count_asymptotic += 1
        bootstrap_estimates = moving_block_bootstrap(x, y, block_length, num_bootstrap)
        se_bootstrap = np.std(bootstrap_estimates[:, 1])
        ci_bootstrap = (beta1_hat - z_score * se_bootstrap, beta1_hat + z_score * se_bootstrap)
        if ci_bootstrap[0] <= beta1_true <= ci_bootstrap[1]:
            coverage_count_bootstrap += 1
    coverage_rate_asymptotic = coverage_count_asymptotic / num_simulations
    coverage_rate_bootstrap = coverage_count_bootstrap / num_simulations
    print(f"sample size T={T}:")
    print(f"  coverage_rate_asymptotic: {coverage_rate_asymptotic:.2f}")
    print(f"  coverage_rate_bootstrap: {coverage_rate_bootstrap:.2f}")
