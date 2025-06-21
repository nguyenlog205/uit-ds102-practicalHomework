import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None

    def fit(self, X):
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvectors by eigenvalues in descending order
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_idx]
        eigenvalues = eigenvalues[sorted_idx]

        # Select the top n_components
        self.components_ = eigenvectors[:, :self.n_components].T

    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
if __name__ == "__main__":
    # Generate a more sophisticated example: 2D data with correlation and some noise
    np.random.seed(42)
    n_samples = 100
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    a, b = 10, 1
    X = np.empty((n_samples, 2))
    X[:, 0] = a * np.cos(theta) + np.random.normal(scale=0.5, size=n_samples)
    X[:, 1] = b * np.sin(theta) + np.random.normal(scale=0.5, size=n_samples)

    pca = PCA(n_components=1)
    X_transformed = pca.fit_transform(X)
    print("Original X:\n", X)
    print("Transformed X:\n", X_transformed)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Plot original data
    axs[0].scatter(X[:, 0], X[:, 1], label='Original Data')
    # Plot the first principal component as an arrow
    mean = pca.mean_
    component = pca.components_[:, 0]
    
    axs[0].set_xlabel('Feature 1')
    axs[0].set_ylabel('Feature 2')
    axs[0].set_title('Original Data with Principal Component')
    axs[0].legend()
    axs[0].grid()

    # Plot PCA transformed data
    axs[1].scatter(X_transformed[:, 0], np.zeros(X_transformed.shape[0]), label='PCA Transformed Data', color='red')
    axs[1].set_xlabel('Principal Component 1')
    axs[1].set_yticks([])
    axs[1].set_title('PCA Transformed Data')
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    plt.show()