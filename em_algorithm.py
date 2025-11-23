import numpy as np
import matplotlib

matplotlib.use('MacOSX')  # You could use other matplotlib backends (TkAgg, Agg, Qt5Agg, etc.)
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import os

OUTPUT_DIR = 'outputs'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def generate_data(n_samples_per_cluster=100):
    np.random.seed(42)

    # Cluster 1 parameters
    mean1 = np.array([0, 0])
    cov1 = np.array([[1.0, 0.0],
                     [0.0, 1.0]])
    cluster1 = np.random.multivariate_normal(mean1, cov1, n_samples_per_cluster)

    # Cluster 2 parameters
    mean2 = np.array([5, 5])
    cov2 = np.array([[1.5, 0.5],
                     [0.5, 1.5]])
    cluster2 = np.random.multivariate_normal(mean2, cov2, n_samples_per_cluster)

    # Cluster 3 parameters
    mean3 = np.array([0, 5])
    cov3 = np.array([[1.0, -0.3],
                     [-0.3, 1.0]])
    cluster3 = np.random.multivariate_normal(mean3, cov3, n_samples_per_cluster)

    # Combine all data
    X = np.vstack([cluster1, cluster2, cluster3])
    true_labels = np.array([0] * n_samples_per_cluster +
                           [1] * n_samples_per_cluster +
                           [2] * n_samples_per_cluster)

    true_params = {
        'means': [mean1, mean2, mean3],
        'covariances': [cov1, cov2, cov3],
        'weights': [1 / 3, 1 / 3, 1 / 3]
    }

    return X, true_labels, true_params


def plot_ground_truth(X, true_labels, true_params):
    plt.figure(figsize=(10, 8))

    colors = ['red', 'blue', 'green']
    labels = ['Cluster 1', 'Cluster 2', 'Cluster 3']

    # Plot data points
    for k in range(3):
        mask = true_labels == k
        plt.scatter(X[mask, 0], X[mask, 1],
                    c=colors[k], label=labels[k],
                    alpha=0.6, s=30)

    # Plot true cluster centers
    for k, mean in enumerate(true_params['means']):
        plt.scatter(mean[0], mean[1],
                    c=colors[k], marker='x',
                    s=300, linewidths=4)

    plt.title('Ground Truth - True Clusters', fontsize=14, fontweight='bold')
    plt.xlabel('X1', fontsize=12)
    plt.ylabel('X2', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1_ground_truth.png'), dpi=150)
    plt.show()

    print("✓ Step 2 completed: Ground truth visualized")


def initialize_parameters(X, K):
    N, D = X.shape

    # Randomly select K points (for means)
    random_indices = np.random.choice(N, K, replace=False)
    mu = X[random_indices].copy()

    # Identity covariance matrix for each cluster
    sigma = [np.eye(D) for _ in range(K)]

    # Equal weights
    pi = np.ones(K) / K

    print(f"✓ Step 3 completed: Initial parameters created for {K} clusters")
    print(f"  - Initial means: {mu}")
    print(f"  - Initial weights: {pi}")

    return mu, sigma, pi


def plot_initial_state(X, mu):
    plt.figure(figsize=(10, 8))

    # All data points
    plt.scatter(X[:, 0], X[:, 1],
                c='gray', alpha=0.5, s=30,
                label='Data Points')

    # Initial means
    plt.scatter(mu[:, 0], mu[:, 1],
                c='red', marker='X',
                s=400, linewidths=2,
                edgecolors='black',
                label='Initial Means')

    # Label the means
    for i, (x, y) in enumerate(mu):
        plt.annotate(f'μ{i + 1}', (x, y),
                     fontsize=12, fontweight='bold',
                     xytext=(10, 10), textcoords='offset points')

    plt.title('Initial State - Randomly Selected Means',
              fontsize=14, fontweight='bold')
    plt.xlabel('X1', fontsize=12)
    plt.ylabel('X2', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '2_initial_state.png'), dpi=150)
    plt.show()

    print("✓ Step 4 completed: Initial state visualized")


def e_step(X, mu, sigma, pi):
    N = X.shape[0]
    K = len(mu)
    responsibilities = np.zeros((N, K))

    # Compute likelihood for each cluster
    for k in range(K):
        # π_k * N(x_n | μ_k, Σ_k)
        responsibilities[:, k] = pi[k] * multivariate_normal.pdf(
            X, mean=mu[k], cov=sigma[k]
        )

    # Normalize (each row should sum to 1)
    responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)

    return responsibilities


def m_step(X, responsibilities):
    N, D = X.shape
    K = responsibilities.shape[1]

    # Effective number of points for each cluster
    N_k = responsibilities.sum(axis=0)  # (K,)

    # Compute new means
    mu = np.dot(responsibilities.T, X) / N_k[:, np.newaxis]  # (K x D)

    # Compute new covariance matrices
    sigma = []
    for k in range(K):
        diff = X - mu[k]  # (N x D)
        # Weighted covariance
        weighted_sum = np.dot(responsibilities[:, k] * diff.T, diff)  # (D x D)
        sigma.append(weighted_sum / N_k[k])

    # Compute new weights
    pi = N_k / N  # (K,)

    return mu, sigma, pi


def compute_log_likelihood(X, mu, sigma, pi):
    N = X.shape[0]
    K = len(mu)
    likelihood = np.zeros((N, K))

    # Compute likelihood for each cluster
    for k in range(K):
        likelihood[:, k] = pi[k] * multivariate_normal.pdf(
            X, mean=mu[k], cov=sigma[k]
        )

    # Log-likelihood
    return np.sum(np.log(likelihood.sum(axis=1)))


def em_algorithm(X, K, max_iter=100, tol=1e-4, verbose=True):
    mu, sigma, pi = initialize_parameters(X, K)

    log_likelihoods = []

    print(f"\n{'=' * 60}")
    print(f"EM ALGORITHM STARTING")
    print(f"{'=' * 60}")
    print(f"Number of clusters: {K}")
    print(f"Number of data points: {X.shape[0]}")
    print(f"Maximum iterations: {max_iter}")
    print(f"Convergence threshold: {tol}")
    print(f"{'=' * 60}\n")

    for iteration in range(max_iter):
        # E-STEP: Compute responsibilities
        responsibilities = e_step(X, mu, sigma, pi)

        # M-STEP: Update parameters
        mu, sigma, pi = m_step(X, responsibilities)

        # Compute log-likelihood
        log_likelihood = compute_log_likelihood(X, mu, sigma, pi)
        log_likelihoods.append(log_likelihood)

        # Progress information
        if verbose and (iteration % 5 == 0 or iteration < 3):
            print(f"Iteration {iteration:3d}: Log-likelihood = {log_likelihood:.4f}")

        # Convergence check
        if iteration > 0:
            improvement = abs(log_likelihoods[-1] - log_likelihoods[-2])
            if improvement < tol:
                print(f"\n{'=' * 60}")
                print(f"✓ CONVERGENCE ACHIEVED!")
                print(f"{'=' * 60}")
                print(f"Number of iterations: {iteration}")
                print(f"Final log-likelihood: {log_likelihood:.4f}")
                print(f"Improvement: {improvement:.6f} < {tol}")
                print(f"{'=' * 60}\n")
                break
    else:
        print(f"\n⚠ Maximum number of iterations ({max_iter}) reached")

    return mu, sigma, pi, responsibilities, log_likelihoods


def plot_convergence(log_likelihoods):
    plt.figure(figsize=(10, 6))
    plt.plot(log_likelihoods, marker='o', linewidth=2, markersize=6)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Log-Likelihood', fontsize=12)
    plt.title('EM Algorithm Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3_convergence.png'), dpi=150)
    plt.show()

    print("✓ Convergence graph created")


def plot_final_clusters(X, mu, sigma, responsibilities):
    # Find the cluster with highest probability for each point
    predicted_labels = np.argmax(responsibilities, axis=1)

    plt.figure(figsize=(10, 8))

    colors = ['red', 'blue', 'green']

    # Plot data points
    for k in range(len(mu)):
        mask = predicted_labels == k
        plt.scatter(X[mask, 0], X[mask, 1],
                    c=colors[k], alpha=0.6, s=30,
                    label=f'Cluster {k + 1}')

    # Plot final means
    plt.scatter(mu[:, 0], mu[:, 1],
                c='black', marker='X',
                s=400, linewidths=2,
                edgecolors='yellow',
                label='Final Means', zorder=5)

    # Draw Gaussian distributions as ellipses
    for k in range(len(mu)):
        plot_gaussian_ellipse(mu[k], sigma[k], colors[k])

    plt.title('Final Clustering Result', fontsize=14, fontweight='bold')
    plt.xlabel('X1', fontsize=12)
    plt.ylabel('X2', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '4_final_clusters.png'), dpi=150)
    plt.show()

    print("✓ Final clustering result visualized")


def plot_gaussian_ellipse(mean, cov, color, n_std=2.0):
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Ellipse parameters
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    # Draw ellipse
    ellipse = Ellipse(mean, width, height, angle=angle,
                      facecolor='none', edgecolor=color,
                      linewidth=2, linestyle='--', alpha=0.7)
    plt.gca().add_patch(ellipse)


def plot_comparison(X, true_labels, predicted_labels):
    """
    Ground truth vs Predicted comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = ['red', 'blue', 'green']

    # Ground Truth
    for k in range(3):
        mask = true_labels == k
        axes[0].scatter(X[mask, 0], X[mask, 1],
                        c=colors[k], alpha=0.6, s=30)
    axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('X1', fontsize=12)
    axes[0].set_ylabel('X2', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')

    # Predicted
    for k in range(3):
        mask = predicted_labels == k
        axes[1].scatter(X[mask, 0], X[mask, 1],
                        c=colors[k], alpha=0.6, s=30)
    axes[1].set_title('EM Algorithm Result', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('X1', fontsize=12)
    axes[1].set_ylabel('X2', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '5_comparison.png'), dpi=150)
    plt.show()

    print("✓ Comparison graph created")


def calculate_accuracy(true_labels, predicted_labels):
    """
    Computes simple accuracy (with cluster label matching)
    """
    from scipy.optimize import linear_sum_assignment

    K = len(np.unique(true_labels))
    confusion_matrix = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            confusion_matrix[i, j] = np.sum((true_labels == i) & (predicted_labels == j))

    # Find best matching using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

    accuracy = confusion_matrix[row_ind, col_ind].sum() / len(true_labels)

    return accuracy


def main():
    """
    Runs all steps of the EM algorithm project
    """
    print("\n" + "=" * 60)
    print("EM ALGORITHM PROJECT")
    print("Expectation-Maximization with Gaussian Mixture Model")
    print("=" * 60 + "\n")

    # STEP 1: Generate data
    print("STEP 1: Generating data...")
    X, true_labels, true_params = generate_data(n_samples_per_cluster=100)
    print(f"✓ {X.shape[0]} data points with {X.shape[1]} dimensions generated\n")

    # STEP 2: Plot ground truth
    print("STEP 2: Visualizing ground truth...")
    plot_ground_truth(X, true_labels, true_params)
    print()

    # STEP 3 & 4: Initialize and plot initial state
    print("STEP 3 & 4: Initializing parameters...")
    K = 3  # Number of clusters
    mu_init, sigma_init, pi_init = initialize_parameters(X, K)
    plot_initial_state(X, mu_init)
    print()

    # STEP 5, 6, 7: Run EM algorithm
    print("STEP 5, 6, 7: Running EM algorithm...")
    mu_final, sigma_final, pi_final, responsibilities, log_likelihoods = em_algorithm(
        X, K, max_iter=100, tol=1e-4, verbose=True
    )

    # Visualize results
    print("\nVisualizing results...")
    plot_convergence(log_likelihoods)
    plot_final_clusters(X, mu_final, sigma_final, responsibilities)

    predicted_labels = np.argmax(responsibilities, axis=1)
    plot_comparison(X, true_labels, predicted_labels)

    # Calculate accuracy
    accuracy = calculate_accuracy(true_labels, predicted_labels)
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Total iterations: {len(log_likelihoods)}")
    print(f"Final log-likelihood: {log_likelihoods[-1]:.4f}")
    print(f"Clustering accuracy: {accuracy * 100:.2f}%")
    print(f"{'=' * 60}\n")

    print("\n✓ All steps completed!")
    print(f"✓ All graphs saved to 'outputs' folder")
    print(f"  - 1_ground_truth.png")
    print(f"  - 2_initial_state.png")
    print(f"  - 3_convergence.png")
    print(f"  - 4_final_clusters.png")
    print(f"  - 5_comparison.png")


if __name__ == "__main__":
    np.random.seed(42)

    main()
