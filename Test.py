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

def kmeans_plus_plus(data, k):
    # Initialize empty centroids and distance matrix
    centroids = []
    distances = np.zeros((data.shape[0], data.shape[1]))

    # Randomly select the first centroid
    centroid_index = np.random.randint(0, len(data))
    centroids.append(data[centroid_index])

    # Select subsequent centroids based on probability proportional to squared distances
    for _ in range(k - 1):
        for i in range(len(data)):
            squared_distances = np.sum((data[i] - centroids) ** 2, axis=1)
            distances[i] = np.min(squared_distances)

        # Calculate probabilities
        probabilities = distances / np.sum(distances)

        # Select the next centroid based on cumulative probabilities
        cumulative_probabilities = np.cumsum(probabilities)
        random_value = np.random.random()
        for index, cumulative_probability in enumerate(cumulative_probabilities):
            if cumulative_probability >= random_value:
                centroid_index = index
                break

        # Check if centroid index is within valid range
        if centroid_index >= len(data):
            centroid_index = len(data) - 1

        centroids.append(data[centroid_index])

    return np.array(centroids), np.argmin(distances, axis=-1)

def neutrosophic_clustering(truth_image, falsity_image, indeterminacy_image, k):
    neutrosophic_image = np.stack([truth_image, falsity_image, indeterminacy_image], axis=-1)
    neutrosophic_image_reshaped = neutrosophic_image.reshape((-1, 3))

    cluster_assignments = kmeans_plus_plus(neutrosophic_image_reshaped, k)[1]

    cluster_assignments = cluster_assignments.reshape(truth_image.shape)

    return cluster_assignments

def refine_clusters(cluster_assignments, min_cluster_size):
    refined_cluster_assignments = np.copy(cluster_assignments)

    for cluster_index in range(np.max(cluster_assignments) + 1):
        cluster_pixels = np.where(cluster_assignments == cluster_index)

        if len(cluster_pixels[0]) < min_cluster_size:
            neighboring_clusters = []
            for pixel in zip(*cluster_pixels):
                neighbors = [(pixel[0] + i, pixel[1] + j) for i in range(-1, 2) for j in range(-1, 2)]
                neighbors = [(x, y) for x, y in neighbors if 0 <= x < cluster_assignments.shape[0] and 0 <= y < cluster_assignments.shape[1] and cluster_assignments[x, y] != cluster_index]
                neighboring_clusters.extend(cluster_assignments[x, y] for x, y in neighbors)

            most_overlapping_cluster = np.argmax(np.bincount(neighboring_clusters))
            refined_cluster_assignments[cluster_pixels] = most_overlapping_cluster

    return refined_cluster_assignments

def segment_image(refined_cluster_assignments):
    # Initialize an empty segmented image with 3 channels (BGR)
    segmented_image = np.zeros((refined_cluster_assignments.shape[0], refined_cluster_assignments.shape[1], 3), dtype=np.uint8)

    # Assign color values to each segment based on cluster assignments
    for cluster_index in range(np.max(refined_cluster_assignments) + 1):
        cluster_pixels = np.where(refined_cluster_assignments == cluster_index)
        cluster_color = np.random.randint(0, 255, size=(3,))
        segmented_image[cluster_pixels] = cluster_color

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
    image_path = "image.jpg"
    k = 2
    min_cluster_size = 50

    image_segmentation(image_path, k, min_cluster_size)
