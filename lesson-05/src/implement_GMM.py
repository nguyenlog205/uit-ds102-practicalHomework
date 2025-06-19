import numpy as np

class GMM:
    def __init__(self, n_components=2, max_iter=100, tol=1e-4):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol

    def _initialize_params(self, X):
        N, D = X.shape
        self.pi = np.full(self.K, 1 / self.K)
        shuffled = X[np.random.choice(N, self.K, replace=False)]
        self.mu = shuffled
        self.sigma = np.array([np.cov(X.T) + 1e-6 * np.eye(D) for _ in range(self.K)])

    def _gaussian(self, X, mu, sigma):
        D = X.shape[1]
        inv = np.linalg.inv(sigma)
        det = np.linalg.det(sigma)
        norm_const = 1 / np.sqrt((2 * np.pi) ** D * det)
        diff = X - mu
        exp_term = np.exp(-0.5 * np.sum(diff @ inv * diff, axis=1))
        return norm_const * exp_term

    def _e_step(self, X):
        N = X.shape[0]
        gamma = np.zeros((N, self.K))
        for k in range(self.K):
            gamma[:, k] = self.pi[k] * self._gaussian(X, self.mu[k], self.sigma[k])
        gamma /= np.sum(gamma, axis=1, keepdims=True)
        return gamma

    def _m_step(self, X, gamma):
        N, D = X.shape
        for k in range(self.K):
            Nk = np.sum(gamma[:, k])
            self.pi[k] = Nk / N
            self.mu[k] = np.sum(gamma[:, k][:, np.newaxis] * X, axis=0) / Nk
            diff = X - self.mu[k]
            self.sigma[k] = (gamma[:, k][:, np.newaxis] * diff).T @ diff / Nk + 1e-6 * np.eye(D)

    def fit(self, X):
        self._initialize_params(X)
        log_likelihoods = []
        for _ in range(self.max_iter):
            gamma = self._e_step(X)
            self._m_step(X, gamma)

            # Compute log likelihood
            ll = np.sum(np.log(np.sum([self.pi[k] * self._gaussian(X, self.mu[k], self.sigma[k]) for k in range(self.K)], axis=0)))
            log_likelihoods.append(ll)

            if len(log_likelihoods) > 1 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < self.tol:
                break

    def predict(self, X):
        gamma = self._e_step(X)
        return np.argmax(gamma, axis=1)

if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    X, y = make_blobs(n_samples=30000, centers=100, random_state=42)

    model = GMM(n_components=100)
    model.fit(X)
    y_pred = model.predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
    plt.title("GMM clustering")
    plt.show()
