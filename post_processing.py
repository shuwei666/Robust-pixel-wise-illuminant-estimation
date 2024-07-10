"""
This is the core code from the 2024 Optics Express paper “Robust pixel-wise illuminant estimation algorithm for
images with a low bit-depth”.

For any questions, please contact the author at shuwei.yue@connect.polyu.hk(shuweiyue.com)

"""
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

EPS = 1e-8


class RobustPhysicalConstrainedPostProcessing:
    """
    RobustPhysicalConstrainedPostProcessing class to handle image processing tasks, including noise identification and removal.

    Steps:
    1. Identification of two major illuminants using KMeans.
    2. Identification of noise based on distance to line segments between illuminants.
    3. Noise removal by replacing noise pixels with regional mean color.
    4. Adjustment of illuminant if angular error between illuminants is below a threshold.
    """

    def __init__(self, sigma=0.2):
        """
        Initialize the RobustPhysicalConstrainedPostProcessing with sigma value.

        :param sigma: Threshold distance to identify noise.
        """
        self.sigma = sigma

    def _normalize_channels(self, img_array):

        r_values = img_array[:, :, 0]
        g_values = img_array[:, :, 1]
        b_values = img_array[:, :, 2]

        if np.any(g_values == 0):
            raise ValueError("G channel contains zero values, which may cause division by zero")

        normalized_r = r_values / (g_values + EPS)
        normalized_b = b_values / (g_values + EPS)
        return normalized_r, normalized_b

    def _remove_extreme_points(self, rb_points, percentile=10):

        r_threshold_high = np.percentile(rb_points[:, 0], 100 - percentile)
        r_threshold_low = np.percentile(rb_points[:, 0], percentile)
        b_threshold_high = np.percentile(rb_points[:, 1], 100 - percentile)
        b_threshold_low = np.percentile(rb_points[:, 1], percentile)

        mask = (rb_points[:, 0] > r_threshold_low) & (rb_points[:, 0] < r_threshold_high) & \
               (rb_points[:, 1] > b_threshold_low) & (rb_points[:, 1] < b_threshold_high)
        return rb_points[mask]

    def _cluster_rb_points(self, rb_points):
        """
        Perform KMeans clustering on the normalized R and B points.

        :return: Cluster centers.
        """
        filtered_rb_points = self._remove_extreme_points(rb_points)
        if filtered_rb_points.shape[0] < 2:
            raise ValueError("Not enough points for clustering after removing extremes")

        kmeans = KMeans(n_clusters=2, random_state=666).fit(filtered_rb_points)
        return kmeans.cluster_centers_

    def _sample_line_segment(self, clusters, num_samples=10):
        """
        Sample points along the line segment between two cluster centers.

        :param clusters: Cluster centers.
        :param num_samples: Number of samples to take along the line segment.
        :return: Sampled points along the line segment.
        """
        return np.linspace(clusters[0], clusters[1], num=num_samples)

    def _identify_noise(self, rb_points, line_samples):
        """
        Identify noise points based on their distance to the sampled line segments.

        :param line_samples: Sampled points along the line segment.
        :return: Indices of noise points and inlier points.
        """
        distances = np.sqrt(((rb_points[:, np.newaxis, :] - line_samples[np.newaxis, :, :]) ** 2).sum(axis=2))
        min_distances = np.min(distances, axis=1)
        noise_indices = np.where(min_distances > self.sigma)[0]
        inlier_indices = np.where(min_distances <= self.sigma)[0]
        return noise_indices, inlier_indices

    def _calculate_angular_error(self, cluster_centers):

        vector_a = np.array([cluster_centers[0][0], 1.0, cluster_centers[0][1]])
        vector_b = np.array([cluster_centers[1][0], 1.0, cluster_centers[1][1]])

        dot_product = np.dot(vector_a, vector_b)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        cos_theta = np.clip(dot_product / (norm_a * norm_b), -0.99999, 0.99999)
        theta = np.arccos(cos_theta)

        angle_error = np.degrees(theta)
        return angle_error

    def _replace_with_global_mean(self, img_array):
        """
        Replace all pixels in the image with the global mean color.

        :param img_array: Input image array.
        :return: Image array with all pixels replaced by the global mean color.
        """
        global_mean = img_array.mean(axis=(0, 1))
        return np.full_like(img_array, global_mean)

    def _replace_noise_with_region_mean(self, img_array, noise_indices):
        """
        Replace noise pixels with the mean color of the surrounding region.

        :param img_array: Original image array.
        :param noise_indices: Indices of noise pixels.
        :return: Image array with noise pixels replaced by regional mean colors.
        """
        height, width, _ = img_array.shape
        new_img_array = np.copy(img_array)
        noise_coords = np.column_stack(np.unravel_index(noise_indices, (height, width)))

        noise_mask = np.zeros((height, width), dtype=bool)
        noise_mask[noise_coords[:, 0], noise_coords[:, 1]] = True

        for i in range(2):
            for j in range(2):
                row_start, row_end = i * height // 2, (i + 1) * height // 2
                col_start, col_end = j * width // 2, (j + 1) * width // 2

                region_noise_mask = noise_mask[row_start:row_end, col_start:col_end]
                region_pixels = img_array[row_start:row_end, col_start:col_end]
                region_non_noise_pixels = region_pixels[~region_noise_mask]

                if region_non_noise_pixels.size == 0:
                    mean_color = np.array([0, 0, 0])
                else:
                    mean_color = np.mean(region_non_noise_pixels, axis=0)

                noise_y, noise_x = np.where(region_noise_mask)
                new_img_array[noise_y + row_start, noise_x + col_start] = mean_color

        return new_img_array

    def process_single_image(self, img_array):
        """
        Process a single image to remove prediction illu. map's noise.

        """
        normalized_r, normalized_b = self._normalize_channels(img_array)
        rb_points = np.column_stack((normalized_r.flatten(), normalized_b.flatten()))

        # Step 1: Identification of two major illuminants using KMeans
        clusters = self._cluster_rb_points(rb_points)

        # Step 2: Identification of noise
        line_samples = self._sample_line_segment(clusters)
        noise_indices, inlier_indices = self._identify_noise(rb_points, line_samples)

        # Step 3: Noise removal
        new_img_array = self._replace_noise_with_region_mean(img_array, noise_indices)

        # Step 4: Adjustment of illuminant
        if self._calculate_angular_error(clusters) < 3.0:
            new_img_array = self._replace_with_global_mean(new_img_array)

        return new_img_array

    def process_batch_images(self, img_arrays):
        """
        Process a batch of images to remove noise.

        :param img_arrays: List of input image arrays.
        :return: List of processed image arrays and additional data for plotting.
        """
        results = []
        for img_array in img_arrays:
            results.append(self.process_single_image(img_array))
        return results


if __name__ == "__main__":
    img_path = 'path_to_image.jpg'  # Add the correct image path here
    sigma = 0.2  # Adjust sigma as needed
    img = Image.open(img_path)
    img_array = np.array(img).astype(np.float32)

    processor = RobustPhysicalConstrainedPostProcessing(sigma)
    new_img_array = processor.process_single_image(img_array)
