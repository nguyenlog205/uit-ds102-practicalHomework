import numpy as np
from numpy.linalg import det, inv

class GaussianMixtureModel:
    def __init__(self, n_components, max_iter=100, tol=1e-4, random_state=None):
        self.n_components = n_components  # K
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        np.random.seed(random_state)

        self.means = None       # Mu_k
        self.covariances = None # Sigma_k
        self.weights = None     # Phi_k

    def _multivariate_gaussian_pdf(self, x, mean, covariance):
        """
        Calculates the probability density function for a multivariate Gaussian distribution.
        """
        D = len(x)
        try:
            # Add a small epsilon to the diagonal for numerical stability (avoid singular matrices)
            eps = 1e-6 * np.eye(D)
            inv_covariance = inv(covariance + eps)
            det_covariance = det(covariance + eps)
            
            if det_covariance <= 0:
                # Handle cases where covariance might become non-positive definite
                # This can happen during early iterations or with bad initializations
                return 0.0 # Return 0 probability if covariance is problematic
                
        except np.linalg.LinAlgError:
            # Handle singular matrix error
            return 0.0 # Return 0 probability if inverse fails

        exponent = -0.5 * (x - mean).T @ inv_covariance @ (x - mean)
        return (1 / ((2 * np.pi)**(D/2) * np.sqrt(det_covariance))) * np.exp(exponent)

    def _initialize_parameters(self, X):
        """
        Initializes means, covariances, and weights.
        Using K-means for initial means is a common practice, but for simplicity
        here we'll randomly pick data points as initial means.
        Covariances are initialized as identity matrices, and weights are uniform.
        """
        N, D = X.shape

        # Randomly pick K data points as initial means
        random_indices = np.random.choice(N, self.n_components, replace=False)
        self.means = X[random_indices].astype(float)

        # Initialize covariances as identity matrices
        self.covariances = np.array([np.eye(D) for _ in range(self.n_components)])

        # Initialize weights uniformly
        self.weights = np.array([1.0 / self.n_components] * self.n_components)

    def _e_step(self, X):
        """
        Expectation step: Calculate responsibilities (gamma_ik).
        """
        N, D = X.shape
        responsibilities = np.zeros((N, self.n_components))

        for i in range(N):
            denominator = 0
            for k in range(self.n_components):
                numerator_term = self.weights[k] * self._multivariate_gaussian_pdf(
                    X[i], self.means[k], self.covariances[k]
                )
                responsibilities[i, k] = numerator_term
                denominator += numerator_term
            
            # Normalize responsibilities to sum to 1 for each data point
            if denominator > 0:
                responsibilities[i] /= denominator
            else:
                # If denominator is 0, it means the data point is very unlikely under all components.
                # Assign uniform responsibility to avoid NaNs.
                responsibilities[i] = 1.0 / self.n_components

        return responsibilities

    def _m_step(self, X, responsibilities):
        """
        Maximization step: Update parameters (means, covariances, weights).
        """
        N, D = X.shape

        # Update weights (phi_k)
        self.weights = np.sum(responsibilities, axis=0) / N

        for k in range(self.n_components):
            Nk = np.sum(responsibilities[:, k]) # Sum of responsibilities for component k

            # Update means (mu_k)
            if Nk > 0: # Avoid division by zero if a component has no responsibility
                self.means[k] = np.sum(responsibilities[:, k, np.newaxis] * X, axis=0) / Nk
            else:
                # If Nk is zero, this component is "dead". Re-initialize its mean to a random point.
                # This is a heuristic to prevent a component from getting stuck.
                self.means[k] = X[np.random.choice(N)]

            # Update covariances (Sigma_k)
            diff = X - self.means[k]
            self.covariances[k] = np.dot((responsibilities[:, k, np.newaxis] * diff).T, diff) / Nk
            
            # Add a small regularization term to the covariance to ensure numerical stability
            # and prevent singular matrices.
            self.covariances[k] += 1e-6 * np.eye(D) # Epsilon regularization

    def _calculate_log_likelihood(self, X):
        """
        Calculates the log-likelihood of the data given the current model parameters.
        Used for convergence check.
        """
        N, D = X.shape
        log_likelihood = 0.0
        for i in range(N):
            component_likelihoods = []
            for k in range(self.n_components):
                pdf_val = self._multivariate_gaussian_pdf(X[i], self.means[k], self.covariances[k])
                component_likelihoods.append(self.weights[k] * pdf_val)
            
            # Sum component likelihoods and take log, handle small values
            sum_component_likelihoods = np.sum(component_likelihoods)
            if sum_component_likelihoods > 0:
                log_likelihood += np.log(sum_component_likelihoods)
            else:
                # If sum is zero, means point is extremely unlikely, add a small penalty
                log_likelihood += -1e10 # A very small number
                
        return log_likelihood

    def fit(self, X):
        """
        Fits the GMM to the data using the EM algorithm.
        """
        N, D = X.shape

        self._initialize_parameters(X)

        prev_log_likelihood = -np.inf

        for iteration in range(self.max_iter):
            # E-step
            responsibilities = self._e_step(X)

            # M-step
            self._m_step(X, responsibilities)

            # Calculate log-likelihood for convergence check
            current_log_likelihood = self._calculate_log_likelihood(X)
            # print(f"Iteration {iteration+1}: Log-Likelihood = {current_log_likelihood}")

            if abs(current_log_likelihood - prev_log_likelihood) < self.tol:
                print(f"GMM converged at iteration {iteration + 1}.")
                break
            
            prev_log_likelihood = current_log_likelihood
        else:
            print(f"GMM reached max_iter ({self.max_iter}) without converging within tolerance.")

    def predict_proba(self, X):
        """
        Predicts the posterior probabilities (responsibilities) of each sample belonging to each component.
        """
        N, D = X.shape
        responsibilities = np.zeros((N, self.n_components))

        for i in range(N):
            denominator = 0
            for k in range(self.n_components):
                numerator_term = self.weights[k] * self._multivariate_gaussian_pdf(
                    X[i], self.means[k], self.covariances[k]
                )
                responsibilities[i, k] = numerator_term
                denominator += numerator_term
            
            if denominator > 0:
                responsibilities[i] /= denominator
            else:
                responsibilities[i] = 1.0 / self.n_components # Default to uniform if all are zero
        return responsibilities

    def predict(self, X):
        """
        Predicts the component index for each sample.
        """
        responsibilities = self.predict_proba(X)
        return np.argmax(responsibilities, axis=1)

# --- Example Usage ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal

    # Generate synthetic data from a true GMM
    np.random.seed(42)
    
    # True parameters
    true_means = np.array([[0, 0], [3, 3], [-3, 3]])
    true_covariances = np.array([
        [[1, 0.5], [0.5, 1]],
        [[0.8, -0.2], [-0.2, 0.5]],
        [[1.5, 0.1], [0.1, 0.7]]
    ])
    true_weights = np.array([0.3, 0.4, 0.3])

    num_samples = 300
    X = np.zeros((num_samples, 2))
    
    # Generate data points based on true weights
    for i in range(num_samples):
        component_idx = np.random.choice(len(true_weights), p=true_weights)
        X[i] = np.random.multivariate_normal(true_means[component_idx], true_covariances[component_idx])

    # Plot the generated data
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], s=20, alpha=0.7, label='Generated Data')
    plt.title('Generated Data for GMM')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Create and fit the GMM
    n_components = 3 # We know the true number of components
    gmm = GaussianMixtureModel(n_components=n_components, max_iter=100, tol=1e-5, random_state=42)
    gmm.fit(X)

    print("\n--- Learned GMM Parameters ---")
    print("Learned Means:\n", gmm.means)
    print("Learned Covariances:\n", gmm.covariances)
    print("Learned Weights:\n", gmm.weights)

    # Visualize the GMM clusters and components
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], s=20, alpha=0.7, label='Data Points')

    # Plot learned components (mean and covariance ellipses)
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    for k in range(n_components):
        mean = gmm.means[k]
        covariance = gmm.covariances[k]
        weight = gmm.weights[k]

        # Plot mean
        plt.scatter(mean[0], mean[1], marker='x', s=100, color=colors[k], linewidth=2, label=f'Component {k+1} Mean')

        # Plot covariance ellipse
        # This part requires some linear algebra for plotting ellipses
        # You can use matplotlib.patches.Ellipse for a more robust visualization
        # For simplicity, we'll plot a few points around the mean based on covariance
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigenvalues) # Scaling factor for ellipse size

        from matplotlib.patches import Ellipse
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                          edgecolor=colors[k], facecolor='none', linestyle='--', linewidth=2, alpha=0.8)
        plt.gca().add_patch(ellipse)

    plt.title('GMM Clustering and Components')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Predict clusters for the data
    cluster_assignments = gmm.predict(X)
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=cluster_assignments, cmap='viridis', s=20, alpha=0.7)
    plt.title('GMM Predicted Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster Assignment')
    plt.grid(True)
    plt.show()

    # Predict probabilities for a new point
    new_point = np.array([1, 1])
    probabilities = gmm.predict_proba(new_point.reshape(1, -1))
    print(f"\nProbabilities for new point {new_point}: {probabilities[0]}")
    predicted_cluster = gmm.predict(new_point.reshape(1, -1))
    print(f"Predicted cluster for new point {new_point}: {predicted_cluster[0]}")