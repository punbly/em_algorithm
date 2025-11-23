# EM Algorithm - Gaussian Mixture Model Implementation

Implementation of the Expectation-Maximization (EM) algorithm for clustering using Gaussian Mixture Models (GMM).

## 📋 Project Overview

This project implements the EM algorithm with the following steps:
1. **Data Generation** - Creates synthetic data from 3 Gaussian distributions
2. **Ground Truth Visualization** - Plots the true clusters
3. **Parameter Initialization** - Randomly initializes means, covariances, and weights
4. **Initial State Visualization** - Shows the initial parameter guesses
5. **E-Step (Expectation)** - Computes posterior probabilities using Bayes theorem
6. **M-Step (Maximization)** - Updates parameters using weighted averages
7. **Convergence Loop** - Iterates until log-likelihood converges

## 🚀 Features

- ✅ Complete EM algorithm implementation from scratch
- ✅ Interactive visualizations with matplotlib
- ✅ Convergence monitoring with log-likelihood tracking
- ✅ Gaussian ellipse visualization
- ✅ Accuracy calculation with Hungarian algorithm
- ✅ All plots automatically saved to `outputs/` folder

## 📦 Requirements

```bash
pip install numpy matplotlib scipy
```

## 💻 Usage

```bash
python em_algorithm_project.py
```

The script will:
- Generate 300 data points from 3 clusters
- Run the EM algorithm until convergence
- Display interactive plots (close each to continue)
- Save all plots to `outputs/` folder

## 📊 Output

The program generates 5 visualizations:
1. `1_ground_truth.png` - True cluster assignments
2. `2_initial_state.png` - Random initial parameters
3. `3_convergence.png` - Log-likelihood over iterations
4. `4_final_clusters.png` - Final clustering result with Gaussian ellipses
5. `5_comparison.png` - Side-by-side comparison of ground truth vs EM result

## 🎯 Results

- **Convergence**: Typically converges in ~30 iterations
- **Accuracy**: Achieves ~98% clustering accuracy
- **Log-Likelihood**: Monitors convergence with threshold of 1e-4

## 📐 Algorithm Details

### E-Step (Expectation)
Computes the responsibility of each cluster for each data point using Bayes theorem:

```
γ(z_nk) = [π_k * N(x_n|μ_k, Σ_k)] / Σ_j [π_j * N(x_n|μ_j, Σ_j)]
```

### M-Step (Maximization)
Updates parameters using weighted averages:

```
μ_k = Σ_n [γ(z_nk) * x_n] / Σ_n γ(z_nk)
Σ_k = Σ_n [γ(z_nk) * (x_n - μ_k)(x_n - μ_k)ᵀ] / Σ_n γ(z_nk)
π_k = Σ_n γ(z_nk) / N
```

## 🔧 Configuration

You can modify parameters in the `main()` function:
- `n_samples_per_cluster`: Number of points per cluster (default: 100)
- `K`: Number of clusters (default: 3)
- `max_iter`: Maximum iterations (default: 100)
- `tol`: Convergence threshold (default: 1e-4)

## 📝 Project Structure

```
.
├── em_algorithm_project.py    # Main implementation
├── outputs/                    # Generated plots
│   ├── 1_ground_truth.png
│   ├── 2_initial_state.png
│   ├── 3_convergence.png
│   ├── 4_final_clusters.png
│   └── 5_comparison.png
└── README.md
```

## 🎓 Educational Purpose

This project was created as part of a Computer Vision course to understand:
- Unsupervised learning algorithms
- Expectation-Maximization framework
- Gaussian Mixture Models
- Soft clustering vs hard clustering
- Convergence analysis

## 📄 License

MIT License - Feel free to use for educational purposes.
