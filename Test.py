import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def grayscale_conversion(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def neutrosophic_falsity(mean_image):
    falsity_image = 1 - mean_image
    return falsity_image

def neutrosophic_indeterminacy(mean_image, std_image, diff_image):
    indeterminacy_image = (std_image * diff_image) / np.maximum(std_image, diff_image)
    return indeterminacy_image

def kmeans_clustering(data, k, max_iterations=10):
    # Initialize centroids randomly
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for _ in range(max_iterations):
        # Calculate distances from data points to centroids
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=-1)

        # Assign each data point to the cluster with the nearest centroid
        labels = np.argmin(distances, axis=-1)

        # Update centroids based on the mean of data points in each cluster
        for i in range(k):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = np.mean(cluster_points, axis=0)

    return labels

def neutrosophic_clustering(truth_image, falsity_image, indeterminacy_image, k):
    neutrosophic_image = np.stack([truth_image, falsity_image, indeterminacy_image], axis=-1)
    neutrosophic_image_reshaped = neutrosophic_image.reshape((-1, 3))

    cluster_assignments = kmeans_clustering(neutrosophic_image_reshaped, k)

    cluster_assignments = cluster_assignments.reshape(truth_image.shape)

    return cluster_assignments

def refine_clusters(cluster_assignments, min_cluster_size):
    refined_cluster_assignments = np.copy(cluster_assignments)

    for cluster_index in range(np.max(cluster_assignments) + 1):
        cluster_pixels = np.where(cluster_assignments == cluster_index)

        if len(cluster_pixels[0]) < min_cluster_size:
            neighboring_clusters = []
            neighboring_cluster_distances = []

            for pixel in zip(*cluster_pixels):
                neighbors = [(pixel[0] + i, pixel[1] + j) for i in range(-1, 2) for j in range(-1, 2)]
                neighbors = [(x, y) for x, y in neighbors if 0 <= x < cluster_assignments.shape[0] and 0 <= y < cluster_assignments.shape[1] and cluster_assignments[x, y] != cluster_index]
                neighboring_clusters.extend(cluster_assignments[x, y] for x, y in neighbors)

                # Calculate average distances to neighboring clusters
                for neighbor_cluster in neighboring_clusters:
                    neighbor_cluster_pixels = np.where(cluster_assignments == neighbor_cluster)
                    distances = np.linalg.norm(pixel - neighbor_cluster_pixels, axis=1)
                    average_distance = np.mean(distances)
                    neighboring_cluster_distances.append(average_distance)

            # Merge with the neighboring cluster with the smallest average distance
            most_overlapping_cluster = np.argmin(neighboring_cluster_distances)
            refined_cluster_assignments[cluster_pixels] = neighboring_clusters[most_overlapping_cluster]

    return refined_cluster_assignments

def segment_image(refined_cluster_assignments):
    # Initialize an empty segmented image with 3 channels (BGR)
    segmented_image = np.zeros((refined_cluster_assignments.shape[0], refined_cluster_assignments.shape[1], 3), dtype=np.uint8)

    # Assign color values to each segment based on cluster assignments
    cluster_colors = np.zeros((np.max(refined_cluster_assignments) + 1, 3), dtype=np.uint8)
    for cluster_index in range(np.max(refined_cluster_assignments) + 1):
        # Generate a unique color for each cluster
        cluster_colors[cluster_index] = np.random.randint(0, 255, size=(3,))

    for cluster_index, cluster_color in enumerate(cluster_colors):
        cluster_pixels = np.where(refined_cluster_assignments == cluster_index)
        segmented_image[cluster_pixels] = cluster_color

    # Check if there are any unassigned pixels
    unassigned_pixels = np.where(segmented_image == 0)
    if unassigned_pixels[0].size > 0:
        # Assign a default color to unassigned pixels
        default_color = np.array([255, 255, 255])  # White
        segmented_image[unassigned_pixels] = default_color

    return segmented_image


def image_segmentation(image_path, k, min_cluster_size):
    try:
        # Open the image using Pillow
        image = Image.open(image_path)

        # Convert the image to RGB format if it's not already
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert the image to a NumPy array
        image_np = np.array(image)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        mean_image = gray_image / 255.0

        # Calculate falsity and indeterminacy values
        falsity_image = neutrosophic_falsity(mean_image)

        std_image = np.std(image_np, axis=-1) / 255.0
        diff_image = np.abs(mean_image - std_image)
        indeterminacy_image = (std_image * diff_image) / (np.maximum(std_image, diff_image) + 1e-8)

        # Perform neutrosophic clustering
        cluster_assignments = neutrosophic_clustering(mean_image, falsity_image, indeterminacy_image, k)

        # Refine clusters based on minimum cluster size
        refined_cluster_assignments = refine_clusters(cluster_assignments, min_cluster_size)

        # Segment the image based on refined cluster assignments
        segmented_image = segment_image(refined_cluster_assignments)

        # Check the number of channels in the segmented image
        if segmented_image.shape[2] <= 1:
            # Image has no channels or only one channel (grayscale), skip grayscale conversion
            pass
        else:
            # Convert the segmented image to grayscale
            segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

        # Convert the segmented image to black and white
        segmented_image = cv2.threshold(segmented_image, 127, 255, cv2.THRESH_BINARY)[1]

        # Display the original and segmented images
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image_np)
        plt.title("Original Image")

        plt.subplot(1, 2, 2)
        plt.imshow(segmented_image, cmap="gray")
        plt.title("Segmented Image")

        plt.show()
    except FileNotFoundError as e:
        print(f"Error opening image file: {e}")


if __name__ == "__main__":
    image_path = "image.jpg"  # Path to the image to be segmented
    k = 3  # Number of clusters
    min_cluster_size = 50  # Minimum cluster size

    image_segmentation(image_path, k, min_cluster_size)


