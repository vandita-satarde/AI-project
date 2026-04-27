import numpy as np

def apply_pca(X, k):
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    cov_matrix = np.dot(X_centered, X_centered.T)

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    idx = np.argsort(-eigenvalues)
    eigenvectors = eigenvectors[:, idx]

    eigenvectors = eigenvectors[:, :k]

    eigenfaces = np.dot(eigenvectors.T, X_centered)

    return eigenfaces, mean, X_centered
