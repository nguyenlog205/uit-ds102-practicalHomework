import numpy as np

class gaussian_mixture_model:
    def __init__(self, n_components=2, max_iter=100, tol=1e-4, val_ratio=0.2, random_state=None):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.val_ratio = val_ratio
        self.random_state = random_state

    def _initialize_params(self, X):
        N, D = X.shape
        self.pi = np.full(self.K, 1 / self.K)
        indices = np.random.choice(N, self.K, replace=False)
        self.mu = X[indices]
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

    def _compute_log_likelihood(self, X):
        probs = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            probs[:, k] = self.pi[k] * self._gaussian(X, self.mu[k], self.sigma[k])
        return np.sum(np.log(np.sum(probs, axis=1)))

    def _split_data(self, X):
        N = X.shape[0]
        if self.random_state is not None:
            np.random.seed(self.random_state)
        indices = np.random.permutation(N)
        val_size = int(self.val_ratio * N)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
        return X[train_idx], X[val_idx]

    def fit(self, X):
        # 1. Tự chia dữ liệu
        X_train, X_val = self._split_data(X)

        # 2. Huấn luyện trên train
        self._initialize_params(X_train)
        log_likelihoods = []
        converged = False

        for i in range(self.max_iter):
            gamma = self._e_step(X_train)
            self._m_step(X_train, gamma)
            ll = self._compute_log_likelihood(X_train)
            log_likelihoods.append(ll)

            if i > 0 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < self.tol:
                converged = True
                break

        # 3. Tính log-likelihood trên validation
        val_log_likelihood = self._compute_log_likelihood(X_val)

        # 4. Trả về metrics
        return {
            'train_log_likelihoods': log_likelihoods,
            'val_log_likelihood': val_log_likelihood,
            'n_iter': i + 1,
            'converged': converged
        }

    def predict(self, X):
        gamma = self._e_step(X)
        return np.argmax(gamma, axis=1)

if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    X, y = make_blobs(n_samples=30000, centers=10, random_state=42)

    model = gaussian_mixture_model(n_components=100)
    model.fit(X)
    y_pred = model.predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
    plt.title("GMM clustering")
    plt.show()
