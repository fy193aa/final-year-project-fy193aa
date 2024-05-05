import os
import time
import tracemalloc
import csv
from PIL import Image
import numpy as np
from sklearn.decomposition import randomized_svd
from sklearn.cluster import KMeans
import skimage.metrics

def save_compressed_representation(compressed_rep_path, cluster_centers, X, residuals):
    """Save the compressed representation to an NPZ file."""
    np.savez_compressed(
        compressed_rep_path,
        cluster_centers=cluster_centers,
        X=X,
        residuals=residuals
    )

def dimensionality_reduction_k_means(image_array, k, epsilon):
    """Perform dimensionality reduction using k-means and SVD."""
    m = k + int(72 * k / epsilon**2) - 1
    shape0 = image_array.shape[0]
    shape1 = image_array.shape[1]
    n_features = min(shape0, shape1)
    m = min(m, n_features)
    
    U, Sigma, Vt = np.linalg.svd(image_array, full_matrices=True)
    A_m = np.dot(U[:, :m], np.dot(np.diag(Sigma[:m]), Vt[:m, :]))
    
    kmeans = KMeans(n_clusters=m, n_init=10, max_iter=300)
    kmeans.fit(A_m)
    
    return (kmeans.cluster_centers_,)

def compress_and_save(image_path, compressed_rep_path, K, epsilon):
    tracemalloc.start()
    compress_start_time = time.time()
    
    original_image = Image.open(image_path).convert("L")
    image_array = np.array(original_image)
    cluster_centers_tuple = dimensionality_reduction_k_means(image_array, K, epsilon)
    cluster_centers = cluster_centers_tuple[0]
    X = np.linalg.lstsq(cluster_centers.T, image_array.T, rcond=None)[0]
    residuals = image_array.T - np.dot(cluster_centers.T, X)
    
    save_compressed_representation(compressed_rep_path, cluster_centers, X, residuals)
    
    compress_time = time.time() - compress_start_time
    _, compress_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return compress_time, compress_peak / 1024**2  # Return compression time and peak memory

def decompress(compressed_rep_path):
    tracemalloc.start()
    decompress_start_time = time.time()
    
    data = np.load(compressed_rep_path)
    cluster_centers = data["cluster_centers"]
    X = data["X"]
    residuals = data["residuals"]
    new_pixels = np.dot(cluster_centers.T, X) + residuals
    new_pixels = new_pixels.T.clip(0, 255).astype("uint8")
    
    decompress_time = time.time() - decompress_start_time
    _, decompress_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return new_pixels, decompress_time, decompress_peak / 1024**2  # Return image, decompress time and peak memory

def main():
    image_path = "/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/gray.bmp"
    results_file = "/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/gray_results.csv"
    ks = [2, 5, 10, 32, 64, 128]
    epsilons = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]

    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["K", "Epsilon", "Compression Time", "Compression Memory Usage", "Decompression Time", "Decompression Memory Usage", "Original Size (KB)", "Compressed Size (KB)", "Compression Ratio"])

        for K in ks:
            for epsilon in epsilons:
                compressed_rep_path = f"/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/data={K}_clusters={epsilon}.npz"
                decompressed_image_path = f"/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/comp_k={K}_clusters={epsilon}.bmp"

                compress_time, compress_memory = compress_and_save(image_path, compressed_rep_path, K, epsilon)
                image_array_reconstructed, decompress_time, decompress_memory = decompress(compressed_rep_path)
                original_size_kb = os.path.getsize(image_path) / 1024
                compressed_size_kb = os.path.getsize(compressed_rep_path) / 1024
                compression_ratio = compressed_size_kb / original_size_kb

                decompressed_image = Image.fromarray(image_array_reconstructed, mode="L")
                decompressed_image.save(decompressed_image_path)

                writer.writerow([K, epsilon, f"{compress_time:.2f}", f"{compress_memory:.2f}", f"{decompress_time:.2f}", f"{decompress_memory:.2f}", f"{original_size_kb:.2f}", f"{compressed_size_kb:.2f}", f"{compression_ratio:.2f}"])

if __name__ == "__main__":
    main()
