#!/usr/bin/env python3
"""
Crosstalk Analyzer and Compensation Module

This module provides functionality to:
1. Analyze crosstalk patterns in pressure sensor data
2. Calculate crosstalk matrices
3. Apply compensation to reduce crosstalk effects
4. Dynamically update calibration based on new data
"""

import numpy as np
from scipy import ndimage
from scipy.stats import pearsonr
from scipy.ndimage import median_filter, gaussian_filter
import time
from collections import deque


class CrosstalkAnalyzer:
    """
    Analyzes and compensates for crosstalk in pressure sensor arrays
    """

    def __init__(self, sensor_shape=(14, 10), correlation_threshold=0.5):
        """
        Initialize the Crosstalk Analyzer

        Args:
            sensor_shape: Shape of the sensor array (default: 14x10)
            correlation_threshold: Threshold for detecting crosstalk (default: 0.5)
        """
        self.sensor_shape = sensor_shape
        self.correlation_threshold = correlation_threshold
        self.num_sensors = sensor_shape[0] * sensor_shape[1]

        # Storage for crosstalk matrices
        self.crosstalk_matrix = np.eye(self.num_sensors)  # Identity matrix initially
        self.compensation_matrix = np.eye(self.num_sensors)
        self.correlation_matrix = np.eye(self.num_sensors)

        # Buffer for dynamic updates
        self.update_buffer_size = 100
        self.update_buffer = deque(maxlen=self.update_buffer_size)

        # Performance tracking
        self.processing_times = deque(maxlen=100)

    def calculate_correlation_matrix(self, data, update_buffer=False):
        """
        Calculate correlation matrix for sensor data

        Args:
            data: Time series data, shape (num_frames, height, width)
            update_buffer: Whether to store in update buffer for dynamic calibration

        Returns:
            correlation_matrix: 140x140 correlation matrix
        """
        start_time = time.time()

        num_frames = data.shape[0]
        flattened_data = data.reshape(num_frames, -1)  # (num_frames, 140)

        # Calculate correlation matrix
        self.correlation_matrix = np.corrcoef(flattened_data.T)

        # Handle NaN values (constant sensors)
        self.correlation_matrix = np.nan_to_num(self.correlation_matrix, nan=0.0)

        if update_buffer:
            self.update_buffer.append(data)

        # Update processing time
        self.processing_times.append(time.time() - start_time)

        return self.correlation_matrix

    def calculate_crosstalk_matrix(self, correlation_matrix=None, threshold_factor=1.0):
        """
        Calculate crosstalk matrix based on correlations

        Args:
            correlation_matrix: Existing correlation matrix (if None, use stored)
            threshold_factor: Multiplier for correlation threshold

        Returns:
            crosstalk_matrix: Matrix representing crosstalk relationships
        """
        if correlation_matrix is None:
            correlation_matrix = self.correlation_matrix

        threshold = self.correlation_threshold * threshold_factor

        # Create crosstalk matrix (absolute values above threshold)
        self.crosstalk_matrix = np.abs(correlation_matrix)
        self.crosstalk_matrix[self.crosstalk_matrix <= threshold] = 0

        # Set diagonal to zero (no self-crosstalk)
        np.fill_diagonal(self.crosstalk_matrix, 0)

        return self.crosstalk_matrix

    def calculate_compensation_matrix(self, method='inverse', regularization=0.01):
        """
        Calculate compensation matrix to reduce crosstalk

        Args:
            method: 'inverse', 'pseudoinverse', or 'weighted'
            regularization: Regularization factor for numerical stability

        Returns:
            compensation_matrix: Matrix for compensating crosstalk
        """
        if method == 'inverse':
            # Simple inverse with regularization
            matrix_to_invert = self.crosstalk_matrix + np.eye(self.num_sensors) * regularization
            try:
                self.compensation_matrix = np.linalg.inv(matrix_to_invert)
            except np.linalg.LinAlgError:
                # Fallback to pseudoinverse if singular
                self.compensation_matrix = np.linalg.pinv(matrix_to_invert)

        elif method == 'pseudoinverse':
            # Moore-Penrose pseudoinverse
            self.compensation_matrix = np.linalg.pinv(self.crosstalk_matrix + regularization * np.eye(self.num_sensors))

        elif method == 'weighted':
            # Weighted compensation based on correlation strength
            weight_matrix = 1.0 / (self.crosstalk_matrix + 0.001)  # Avoid division by zero
            self.compensation_matrix = weight_matrix / np.sum(weight_matrix, axis=1, keepdims=True)

        # Normalize compensation matrix
        self.compensation_matrix = self.compensation_matrix / np.sum(np.abs(self.compensation_matrix), axis=1, keepdims=True)

        return self.compensation_matrix

    def apply_compensation(self, sensor_data):
        """
        Apply crosstalk compensation to sensor data

        Args:
            sensor_data: Flattened sensor data (1, 140) or (140,)

        Returns:
            compensated_data: Compensated sensor data
        """
        start_time = time.time()

        # Ensure correct shape
        if sensor_data.ndim == 1:
            sensor_data = sensor_data.reshape(1, -1)

        # Apply compensation
        compensated_data = np.dot(sensor_data, self.compensation_matrix.T)

        # Update processing time
        self.processing_times.append(time.time() - start_time)

        return compensated_data.flatten()

    def apply_adaptive_filtering(self, frame, activity_threshold=100, max_filter_size=3):
        """
        Apply adaptive spatial filtering based on local activity

        Args:
            frame: Single frame data (14, 10)
            activity_threshold: Threshold for adaptive filtering
            max_filter_size: Maximum filter size for high activity areas

        Returns:
            filtered_frame: Filtered frame
        """
        filtered_frame = frame.copy()

        # Find active regions (above threshold)
        active_mask = frame > activity_threshold

        # Apply different filtering based on local activity
        for i in range(self.sensor_shape[0]):
            for j in range(self.sensor_shape[1]):
                if active_mask[i, j]:
                    # Calculate local activity
                    neighbors = []
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            ni, nj = i + di, j + dj
                            if (0 <= ni < self.sensor_shape[0] and
                                0 <= nj < self.sensor_shape[1] and
                                (di != 0 or dj != 0)):
                                neighbors.append(frame[ni, nj])

                    if neighbors:
                        local_activity = np.mean(neighbors)

                        # Adaptive filter size based on local activity
                        if local_activity > activity_threshold * 2:
                            filter_size = 1  # No filtering for very high activity
                        elif local_activity > activity_threshold:
                            filter_size = min(2, max_filter_size)
                        else:
                            filter_size = min(3, max_filter_size)

                        # Apply filtering if filter size > 1
                        if filter_size > 1:
                            # Create a small region for filtering
                            region_min_i = max(0, i - filter_size//2)
                            region_max_i = min(self.sensor_shape[0], i + filter_size//2 + 1)
                            region_min_j = max(0, j - filter_size//2)
                            region_max_j = min(self.sensor_shape[1], j + filter_size//2 + 1)

                            # Extract region and filter
                            region = frame[region_min_i:region_max_i, region_min_j:region_max_j]
                            if region.size > 1:
                                filtered_region = median_filter(region, size=min(filter_size, min(region.shape)))
                                filtered_region = gaussian_filter(filtered_region, sigma=0.5)

                                # Apply filtered values
                                filtered_frame[region_min_i:region_max_i, region_min_j:region_max_j] = filtered_region

        return filtered_frame

    def update_calibration(self, new_data, learning_rate=0.1):
        """
        Dynamically update calibration based on new data

        Args:
            new_data: New sensor data (num_frames, 14, 10)
            learning_rate: Learning rate for updates (0-1)
        """
        if len(self.update_buffer) > 0:
            # Combine old and new data
            combined_data = np.concatenate(list(self.update_buffer) + [new_data], axis=0)

            # Update correlation matrix
            new_correlation = self.calculate_correlation_matrix(combined_data)

            # Update crosstalk matrix with learning rate
            self.crosstalk_matrix = (1 - learning_rate) * self.crosstalk_matrix + learning_rate * np.abs(new_correlation)
            np.fill_diagonal(self.crosstalk_matrix, 0)

            # Update compensation matrix
            self.calculate_compensation_matrix()

    def get_crosstalk_stats(self):
        """
        Get statistics about crosstalk

        Returns:
            stats: Dictionary with crosstalk statistics
        """
        # Count significant crosstalk pairs
        significant_crosstalk = np.sum(self.crosstalk_matrix > self.correlation_threshold)

        # Find strongest crosstalk
        max_crosstalk = np.max(self.crosstalk_matrix)

        # Calculate average crosstalk per sensor
        avg_crosstalk_per_sensor = np.mean(self.crosstalk_matrix, axis=1)

        stats = {
            'significant_crosstalk_pairs': significant_crosstalk,
            'max_crosstalk': max_crosstalk,
            'avg_crosstalk_per_sensor': np.mean(avg_crosstalk_per_sensor),
            'max_avg_crosstalk_sensor': np.argmax(avg_crosstalk_per_sensor),
            'processing_time_ms': np.mean(self.processing_times) * 1000 if self.processing_times else 0
        }

        return stats

    def visualize_crosstalk_matrix(self, ax=None):
        """
        Visualize the crosstalk matrix

        Args:
            ax: Matplotlib axis for plotting

        Returns:
            ax: Matplotlib axis with plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 8))

            # Plot crosstalk matrix
            im = ax.imshow(self.crosstalk_matrix, cmap='coolwarm', interpolation='nearest')
            ax.set_title('Crosstalk Matrix')
            ax.set_xlabel('Sensor Index')
            ax.set_ylabel('Sensor Index')

            # Add colorbar
            plt.colorbar(im, ax=ax)

            return ax

        except ImportError:
            print("Matplotlib/Seaborn not available for visualization")
            return None