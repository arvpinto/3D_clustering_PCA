import sys
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to read data from a file
def read_data(file_path):
    try:
        data = np.loadtxt(file_path)
        frame_numbers = np.arange(1, len(data) + 1)  # Frame numbers start from 1
        return data, frame_numbers
    except Exception as e:
        print(f"Error reading the file: {e}")
        sys.exit(1)

# Function to calculate density-based point and KDE
def calculate_density_based_point(cluster_points):
    kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(cluster_points)
    log_density = kde.score_samples(cluster_points)
    density_based_point = cluster_points[np.argmax(log_density)]
    return density_based_point, kde

# Function to find the indices of the closest frames
def find_closest_frames(cluster_points, density_based_point):
    distances = np.linalg.norm(cluster_points - density_based_point, axis=1)
    closest_indices = np.argsort(distances)[:5]
    return closest_indices

# Function to normalize densities to [0, 1] within each cluster
def normalize_densities(densities):
    min_density = np.min(densities)
    max_density = np.max(densities)
    normalized_densities = (densities - min_density) / (max_density - min_density)
    return normalized_densities

# Function to print GMM cluster information
def print_gmm_info(gmm_clusters, pca_vectors, frame_numbers):
    unique_clusters, counts = np.unique(gmm_clusters, return_counts=True)
    total_clusters = len(unique_clusters)

    print(f"Total number of clusters (GMM): {total_clusters}")

    with open("frames.dat", "w") as f_frames:
        for cluster_num in unique_clusters:
            cluster_indices = np.where(gmm_clusters == cluster_num)
            cluster_frame_numbers = frame_numbers[cluster_indices]

            density_based_point, _ = calculate_density_based_point(pca_vectors[cluster_indices])
            closest_indices = find_closest_frames(pca_vectors[cluster_indices], density_based_point)

            f_frames.write(f"Cluster {cluster_num}: {cluster_frame_numbers[closest_indices].tolist()} \n")

            print(f"Cluster {cluster_num}: {len(cluster_indices[0])} frames")
            print(f"Top 5 closest frames for Cluster {cluster_num}: {cluster_frame_numbers[closest_indices]}")

# Function to plot density distributions
def plot_density_distributions(pca_vectors, gmm_clusters):
    num_clusters = len(np.unique(gmm_clusters))
    fig, axs = plt.subplots(1, pca_vectors.shape[1], figsize=(15, 5))

    kde_curves = []  # To store the KDE curves for later extraction of y-values

    for i in range(pca_vectors.shape[1]):
        kde_curves_i = []  # To store the KDE curves for the current dimension
        for cluster_num in np.unique(gmm_clusters):
            cluster_indices = np.where(gmm_clusters == cluster_num)[0]
            cluster_data = pca_vectors[cluster_indices][:, i].reshape(-1, 1)

            density_based_point, kde = calculate_density_based_point(cluster_data)
            sns.kdeplot(cluster_data.flatten(), label=f'Cluster {cluster_num}', ax=axs[i])

            # Store the KDE curve
            kde_curve = axs[i].lines[-1]
            kde_curves_i.append((cluster_num, kde_curve))

            # Scatter plot for the top-ranked frames
            top_ranked_frames = pca_vectors[cluster_indices, i][find_closest_frames(pca_vectors[cluster_indices], density_based_point)]
            axs[i].scatter(top_ranked_frames, np.zeros_like(top_ranked_frames), color=f'C{cluster_num}', marker='o', label=f'Frames {cluster_num}')

        kde_curves.append((i, kde_curves_i))

        axs[i].set_title(f'Kernel Density Estimation - PC{i+1}')
        axs[i].set_xlabel(f'PC{i+1}')
        axs[i].set_ylabel('Density')
        axs[i].legend()

    # Extract y-values from KDE curves and set them for point markers
    for dim, curves in kde_curves:
        for cluster_num, kde_curve in curves:
            x_values = kde_curve.get_xdata()
            y_values = kde_curve.get_ydata()

            for marker in axs[dim].collections:
                if marker.get_label() == f'Frames {cluster_num}':
                    marker_x = marker.get_offsets()[:, 0]
                    marker_y = np.interp(marker_x, x_values, y_values)
                    marker.set_offsets(np.column_stack((marker_x, marker_y)))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Check for correct number of command-line arguments
    if len(sys.argv) != 5:
        print("Usage: python pca_dbscan_gmm.py <data_file> <eps> <min_samples> <n_components>")
        sys.exit(1)

    # Parse command-line arguments
    data_file, eps, min_samples, n_components = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

    # Read data from the file
    pca_vectors, frame_numbers = read_data(data_file)

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    clusters = dbscan.fit_predict(pca_vectors)

    # Remove outliers from the dataset
    non_outlier_indices = np.where(clusters != -1)
    non_outlier_data = pca_vectors[non_outlier_indices]

    num_dbscan_outliers = np.sum(clusters == -1)
    print(f"Number of DBSCAN outliers: {num_dbscan_outliers}")

    # Apply Gaussian Mixture Model (GMM) clustering
    gmm = GaussianMixture(n_components=n_components, tol=1e-8, max_iter=5000)
    gmm_clusters = gmm.fit_predict(non_outlier_data)

    # Print GMM cluster information
    print_gmm_info(gmm_clusters, pca_vectors, frame_numbers)

    # Assign GMM cluster labels to all data points
    all_clusters = np.full_like(clusters, fill_value=-1)
    all_clusters[non_outlier_indices] = gmm_clusters

    # Calculate density-based points and distances to them
    density_based_points = np.array([calculate_density_based_point(non_outlier_data[gmm_clusters == i])[0] for i in range(n_components)])
    distances_to_density_based_points = np.linalg.norm(pca_vectors - density_based_points[all_clusters], axis=1)
    max_distance = np.max(distances_to_density_based_points)

    # Adjusted_scaled_point_size now uses normalized densities for each cluster
    size_scaling_factor = 0.75
    adjusted_scaled_point_size = normalize_densities(1 / (distances_to_density_based_points + 1) * size_scaling_factor)

    # Create DataFrame with original data and new columns
    df_original_data = pd.DataFrame(pca_vectors, columns=[f'PC{i+1}' for i in range(pca_vectors.shape[1])])
    df_original_data['Frame'] = frame_numbers
    df_original_data['Cluster'] = all_clusters

    # Save DataFrame to clusters.csv
    df_original_data.to_csv('clusters.csv', index=False)

    # Create 3D scatter plot using Plotly
    fig = px.scatter_3d(df_original_data, x='PC1', y='PC2', z='PC3', color='Cluster', opacity=0.3,
                         color_discrete_sequence=['black'],
                         color_continuous_scale=['black', 'blue', 'green', 'yellow', 'orange', 'red'],
                         size=adjusted_scaled_point_size,
                         size_max=max_distance * size_scaling_factor, title='3D Clustering of PCA Vectors')

    # Add scatter plot for the top-ranked frames in each GMM cluster
    for cluster_num in np.unique(gmm_clusters):
        cluster_indices = np.where(gmm_clusters == cluster_num)
        cluster_points = pca_vectors[cluster_indices]

        density_based_point, _ = calculate_density_based_point(cluster_points)
        closest_indices = find_closest_frames(cluster_points, density_based_point)

        if len(closest_indices) > 0:
            df_closest_frames = pd.DataFrame({'PC1': pca_vectors[cluster_indices[0][closest_indices], 0],
                                              'PC2': pca_vectors[cluster_indices[0][closest_indices], 1],
                                              'PC3': pca_vectors[cluster_indices[0][closest_indices], 2]})

            trace_closest_frames = go.Scatter3d(x=df_closest_frames['PC1'],
                                                y=df_closest_frames['PC2'],
                                                z=df_closest_frames['PC3'],
                                                mode='markers',
                                                marker=dict(size=10, color='white', symbol='circle', line=dict(color='black', width=10)),
                                                name=f'Frames {cluster_num}')
            fig.add_trace(trace_closest_frames)

    # Add scatter plot for DBSCAN outliers
    df_outliers = pd.DataFrame({'PC1': pca_vectors[clusters == -1, 0],
                                'PC2': pca_vectors[clusters == -1, 1],
                                'PC3': pca_vectors[clusters == -1, 2],
                                'Cluster': [-1] * num_dbscan_outliers})

    trace_outliers = go.Scatter3d(x=df_outliers['PC1'], y=df_outliers['PC2'], z=df_outliers['PC3'],
                                 mode='markers',
                                 marker=dict(size=3, color='black'))
    fig.add_trace(trace_outliers)

    # Update plot layout
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.update_layout(showlegend=False)
    fig.update_layout(scene=dict(aspectmode='cube'))
    fig.show()

    # Plot density distributions using Seaborn and Matplotlib
    plot_density_distributions(pca_vectors, gmm_clusters)
