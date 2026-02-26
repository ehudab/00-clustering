## Clustering From Scratch (K-Means, DBSCAN, HDBSCAN)

This project demonstrates core clustering algorithms implemented **from scratch** in NumPy, alongside comparisons with a production HDBSCAN implementation. It is designed as an educational notebook that walks through:

- Vectorized K-Means using only functions (no classes)
- Function-based DBSCAN
- A simplified HDBSCAN-style algorithm (mutual reachability + MST + stability selection)
- Comparison with the official `hdbscan` library
- Internal metrics to evaluate and compare cluster quality

All code lives in the notebook:

- `clustering.ipynb`

and the dataset is:

- `clustering.csv`

You can run everything either in VS Code (recommended) or in a standard Jupyter environment.

---

## 1. Environment Setup

### 1.1. Python and virtual environment

From the project root (`clustering`), create and activate a virtual environment (example with `venv`):

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
& .venv\Scripts\Activate.ps1
```

### 1.2. Install dependencies

Install required packages (either from `pyproject.toml` via your tool, or directly with `pip`):

```bash
pip install numpy pandas matplotlib scikit-learn hdbscan requests
```

If you plan to run the notebook in Google Colab, you may also see a cell that installs:

```python
!pip install numpy scikit-learn matplotlib pandas google-colab requests hdbscan
```

---

## 2. Data

The main dataset is stored as:

- `clustering.csv` – contains at least two numeric columns `x` and `y` used as features.

---

## 3. Notebook Structure

Open `clustering.ipynb` in VS Code or Jupyter. The notebook is structured into the following sections:

### 3.1. Setup and data loading

- Installs Python packages (if needed).
- Downloads `clustering.csv` from a Google Drive URL when required.
- Loads the data into a NumPy array `X` with columns `x` and `y`.

### 3.2. Visualization helper

- `plot_clusters(X, labels, title)`: utility to scatter-plot 2D clusters, highlighting noise points as `x` markers and giving each cluster a distinct color.

### 3.3. K-Means from scratch

Implemented using **pure functions** (no classes):

- `initialize_centroids(X, k, random_state)`
  - Randomly chooses `k` data points as initial centroids.
- `assign_clusters(X, centroids)`
  - Computes Euclidean distances from each point to each centroid using broadcasting, returns the index of the closest centroid for each point.
- `kmeans_update_centroids(X, labels, centroids)`
  - Recomputes each centroid as the mean of the points assigned to that cluster, keeping the old centroid if a cluster becomes empty.
- `kmeans_fit_predict(X, k, max_iters, random_state)`
  - Full K-Means loop (initialize → assign → update → repeat) until convergence or maximum iterations.

The notebook then runs K-Means (e.g. with `k=7`) and plots the resulting clusters.

### 3.4. DBSCAN from scratch

DBSCAN is implemented via:

- `dbscan_region_query(X, idx, eps)`
  - Returns indices of all points within distance `eps` of point `idx`.
- `dbscan_expand_cluster(X, labels, idx, cluster_id, eps, min_samples)`
  - Starting from a core point, grows a cluster by repeatedly adding density-reachable neighbors.
- `dbscan_fit_predict(X, eps, min_samples)`
  - Main driver that loops over points, launches `dbscan_expand_cluster` when it finds an unvisited point, labels noise as `-1`, and returns final labels.

This section demonstrates density-based clustering and how noise points are identified.

### 3.5. Simplified HDBSCAN from scratch

This is a **didactic** (not production) implementation that mimics the key ideas of HDBSCAN:

1. `pairwise_distances(X)`
	- Computes the full pairwise Euclidean distance matrix using broadcasting.
2. `core_distances(X, min_samples)`
	- For each point, takes the distance to its `min_samples`-th nearest neighbor (core distance).
3. `mutual_reachability_distances(X, min_samples)`
	- Builds a dense matrix `mreach` where `mreach(i, j) = max(distance(i, j), core(i), core(j))`.
4. `kruskal_mst(mreach)`
	- Runs Kruskal's algorithm on the complete graph defined by `mreach` to build a minimum spanning tree (MST).
5. `build_cluster_tree(n_points, edges)`
	- Interprets MST edges as merges in increasing-distance order to create a merge tree (hierarchy), tracking birth/death levels and cluster sizes.
6. `select_clusters(root, children, stability, min_stability)`
	- Selects internal nodes (clusters) whose stability exceeds a threshold, preferring more stable clusters.
7. `label_points(n_points, selected, children)`
	- Assigns final labels to original points based on the selected cluster nodes; points not covered by any cluster remain noise.
8. `hdbscan_scratch(X, min_samples)`
	- Ties all the above together and returns cluster labels for a (possibly downsampled) subset of `X`.

Because this version uses dense `n×n` matrices and an explicit edge list, it is suitable only for **small to medium** datasets (e.g. up to a few thousand points). The notebook therefore samples a subset (e.g. 2000 points) when running this section.


### 3.6. Metrics and comparison

The notebook computes three standard internal clustering metrics (on non-noise points):

- **Silhouette score** (higher is better, range -1 to 1)
- **Davies–Bouldin index** (lower is better)
- **Calinski–Harabasz index** (higher is better)

For each algorithm (K-Means, DBSCAN, HDBSCAN), a small helper function `score_all` returns these three values and the notebook displays them in a table (`metrics_df_rounded`) for easy comparison.

---

## 4. Running the Notebook

1. Activate your virtual environment and install dependencies (see Section 1).
2. Open `clustering.ipynb` in VS Code or Jupyter.
3. Run all cells in order:
	- Data loading
	- Plotting helper
	- K-Means from scratch
	- DBSCAN from scratch
	- HDBSCAN (scratch, on a subset)
	- Metrics table

If you see kernel memory issues when running the scratch HDBSCAN on a large dataset, reduce the subset size (e.g. 1000–2000 points).

