import numpy as np
import matplotlib.pyplot as plt

# Generate some sample data
np.random.seed(42)
x = np.random.normal(0, 1, size=100)
y = 2*x + np.random.normal(0, 1, size=100)

# Define the function to bootstrap
def bootstrap(data, func, n_bootstraps):
    resamples = np.random.choice(data, size=(n_bootstraps, len(data)), replace=True)
    return np.array([func(resample) for resample in resamples])

# Define the function to calculate the confidence interval
def ci(data, alpha=0.05):
    return np.percentile(data, [(alpha/2)*100, (1-alpha/2)*100])

# Bootstrap the data to get the confidence intervals
n_bootstraps = 1000
bootstrap_means = bootstrap(y, np.mean, n_bootstraps)
ci_means = ci(bootstrap_means)

# Plot the data and the confidence intervals
plt.scatter(x, y, alpha=0.5)
plt.plot(np.sort(x), np.sort(x)*2, color='black', linestyle='--', label='True line')
plt.fill_between(np.sort(x), np.sort(x)*2 + ci_means[0], np.sort(x)*2 + ci_means[1], alpha=0.2, color='blue', label='95% CI')
plt.legend()
plt.show()
