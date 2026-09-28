#!/usr/bin/env python3
"""
Interactive Correlation Visualizer for Pressure Sensor Array

This script creates an interactive visualization of correlation matrix
for the 14x10 pressure sensor array. When hovering over a sensor point,
it shows the correlation between that sensor and all other sensors
using color coding.

Usage:
    python interactive_correlation_visualizer.py [--data DATA_NPY] [--config CONFIG_JSON] [--correlation CORRELATION_NPZ]

Example:
    python interactive_correlation_visualizer.py --data PalmPoints_and_FingerBend/For_correlation.npy
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import sys
import os
from pathlib import Path

# Add the parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from PalmPoints_and_FingerBend.crosstalk_analyzer import CrosstalkAnalyzer
    HAS_CROSSTALK_ANALYZER = True
except ImportError:
    HAS_CROSSTALK_ANALYZER = False
    print("Warning: CrosstalkAnalyzer not found. Will need precomputed correlation matrix.")


def load_palm_coordinates(config_path):
    """Load palm coordinates from configuration file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        coords = np.array(config["PlaneHandPressure_Display"]["PlaneCoords"])
        return coords
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except KeyError:
        print("Error: 'PlaneHandPressure_Display' or 'PlaneCoords' not found in config")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: Invalid JSON in config file")
        sys.exit(1)


def load_pressure_data(data_path):
    """Load pressure data from NPY file"""
    try:
        pressure_data = np.load(data_path)
        print(f"Pressure data shape: {pressure_data.shape}")
        return pressure_data
    except FileNotFoundError:
        print(f"Error: Pressure data file not found at {data_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading pressure data: {e}")
        sys.exit(1)


def calculate_correlation_matrix(pressure_data):
    """
    Calculate correlation matrix from pressure data

    Args:
        pressure_data: Time series pressure data, shape (num_frames, rows, cols)

    Returns:
        correlation_matrix: shape (num_sensors, num_sensors)
    """
    num_frames, rows, cols = pressure_data.shape
    num_sensors = rows * cols

    # Reshape to (num_frames, num_sensors)
    flattened_data = pressure_data.reshape(num_frames, -1)

    # Calculate correlation matrix
    correlation_matrix = np.corrcoef(flattened_data.T)

    # Handle NaN values (constant sensors)
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)

    return correlation_matrix


def load_correlation_matrix(corr_path):
    """Load precomputed correlation matrix from NPZ or NPY file"""
    if corr_path.endswith('.npz'):
        data = np.load(corr_path)
        # Try common keys
        for key in ['correlation_matrix', 'correlations', 'crosstalk_matrix']:
            if key in data:
                print(f"Loaded correlation matrix from key '{key}' in {corr_path}")
                return data[key]
        # If no matching key, assume first array
        first_key = list(data.keys())[0]
        print(f"Loaded array from key '{first_key}' in {corr_path}")
        return data[first_key]
    else:
        # Assume .npy file
        return np.load(corr_path)


class InteractiveCorrelationVisualizer:
    """
    Interactive visualization of correlation matrix
    """

    def __init__(self, coords, correlation_matrix, sensor_shape=(14, 10)):
        """
        Initialize visualizer

        Args:
            coords: Array of shape (num_sensors, 2) with sensor coordinates
            correlation_matrix: Array of shape (num_sensors, num_sensors) with correlations
            sensor_shape: Tuple (rows, cols) of sensor grid
        """
        self.coords = coords
        self.correlation_matrix = correlation_matrix
        self.sensor_shape = sensor_shape
        self.num_sensors = coords.shape[0]

        # Validate inputs
        assert self.num_sensors == correlation_matrix.shape[0] == correlation_matrix.shape[1], \
            f"Mismatch: coords has {self.num_sensors} sensors, correlation matrix is {correlation_matrix.shape}"

        # Figure and axes
        self.fig = None
        self.ax = None
        self.scatter = None
        self.colorbar = None
        self.title = None

        # Current hover state
        self.hovered_sensor = None
        self.default_colors = np.zeros(self.num_sensors)  # Default color (all zero)

        # Color map for correlations (-1 to 1)
        self.cmap = plt.cm.RdBu_r

        # Calculate axis limits with padding
        self.x_min, self.x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        self.y_min, self.y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
        self.padding_x = (self.x_max - self.x_min) * 0.1
        self.padding_y = (self.y_max - self.y_min) * 0.1

        print(f"Initialized visualizer with {self.num_sensors} sensors")
        print(f"Correlation matrix range: [{correlation_matrix.min():.3f}, {correlation_matrix.max():.3f}]")

    def create_plot(self):
        """Create the initial plot with all sensors"""
        self.fig, self.ax = plt.subplots(figsize=(12, 10))

        # Plot sensors with default color (gray)
        self.scatter = self.ax.scatter(
            self.coords[:, 0],
            -self.coords[:, 1],  # Negative to flip y-axis (matching other visualizations)
            c=self.default_colors,
            cmap=self.cmap,
            s=300,
            alpha=0.8,
            edgecolors='black',
            linewidth=0.5,
            vmin=-1,
            vmax=1,
            picker=True,  # Enable picking for hover events
            pickradius=10  # Tolerance for picking
        )

        # Add sensor indices as text (optional, can be toggled)
        for i, (x, y) in enumerate(self.coords):
            self.ax.text(x, -y, str(i), ha='center', va='center',
                        fontsize=6, fontweight='bold', alpha=0.5)

        # Set up axes
        self.ax.set_xlim(self.x_min - self.padding_x, self.x_max + self.padding_x)
        self.ax.set_ylim(-self.y_max - self.padding_y, -self.y_min + self.padding_y)
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)

        # Add colorbar
        self.colorbar = self.fig.colorbar(self.scatter, ax=self.ax, orientation='vertical')
        self.colorbar.set_label('Correlation with hovered sensor')

        # Initial title
        self.title = self.ax.set_title('Interactive Correlation Visualizer - Hover over a sensor point')

        # Connect events
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)

        print("Plot created. Hover over sensor points to see correlations.")

    def on_hover(self, event):
        """Handle mouse motion events to highlight nearest sensor"""
        if event.inaxes != self.ax:
            return

        # Find closest sensor point
        distances = np.sqrt((self.coords[:, 0] - event.xdata)**2 +
                           ((-self.coords[:, 1]) - event.ydata)**2)
        closest_idx = np.argmin(distances)
        min_distance = distances[closest_idx]

        # Only update if we're close enough to a point
        threshold = (self.x_max - self.x_min) * 0.02  # 2% of x-range
        if min_distance < threshold:
            if self.hovered_sensor != closest_idx:
                self.hovered_sensor = closest_idx
                self.update_colors()
        else:
            if self.hovered_sensor is not None:
                self.hovered_sensor = None
                self.update_colors()

    def on_pick(self, event):
        """Handle pick events (mouse click on a point)"""
        if event.artist != self.scatter:
            return

        # Get the index of the picked point
        ind = event.ind[0]
        self.hovered_sensor = ind
        self.update_colors()

    def update_colors(self):
        """Update scatter plot colors based on hovered sensor"""
        if self.hovered_sensor is None:
            # Reset to default (all gray)
            colors = np.zeros(self.num_sensors)
            self.title.set_text('Interactive Correlation Visualizer - Hover over a sensor point')
        else:
            # Get correlations with hovered sensor
            colors = self.correlation_matrix[self.hovered_sensor, :].copy()
            # Set self-correlation to 1 for highlighting
            colors[self.hovered_sensor] = 1.0

            # Update title
            row = self.hovered_sensor // self.sensor_shape[1]
            col = self.hovered_sensor % self.sensor_shape[1]
            self.title.set_text(f'Sensor {self.hovered_sensor} (row={row}, col={col}) - Correlations with all sensors')

        # Update scatter plot colors
        self.scatter.set_array(colors)
        self.fig.canvas.draw_idle()

    def show(self):
        """Show the interactive plot"""
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Interactive visualization of pressure sensor correlations'
    )
    parser.add_argument('--data',
                        default='For_correlation.npy',
                        help='Path to pressure data NPY file (default: PalmPoints_and_FingerBend/For_correlation.npy)')
    parser.add_argument('--config',
                        default='PalmPointsConfig.json',
                        help='Path to palm coordinates config JSON (default: PalmPoints_and_FingerBend/PalmPointsConfig.json)')
    parser.add_argument('--correlation',
                        help='Path to precomputed correlation matrix (NPZ or NPY file). If not provided, will calculate from data.')
    parser.add_argument('--sensor-shape', default='14,10',
                        help='Sensor grid shape as rows,cols (default: 14,10)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display interactive window (useful for saving)')
    parser.add_argument('--output',
                        help='Save screenshot to this file (requires --no-show)')

    args = parser.parse_args()

    # Parse sensor shape
    try:
        sensor_shape = tuple(map(int, args.sensor_shape.split(',')))
        if len(sensor_shape) != 2:
            raise ValueError
    except ValueError:
        print(f"Error: Invalid sensor shape '{args.sensor_shape}'. Use rows,cols (e.g., 14,10)")
        sys.exit(1)

    print("Loading palm coordinates...")
    coords = load_palm_coordinates(args.config)
    print(f"Loaded {coords.shape[0]} sensor coordinates")

    # Load or calculate correlation matrix
    if args.correlation:
        print(f"Loading precomputed correlation matrix from {args.correlation}...")
        correlation_matrix = load_correlation_matrix(args.correlation)
    else:
        if not HAS_CROSSTALK_ANALYZER:
            print("Error: Need pressure data to calculate correlation matrix.")
            print("Please provide --data argument or install CrosstalkAnalyzer.")
            sys.exit(1)

        print(f"Loading pressure data from {args.data}...")
        pressure_data = load_pressure_data(args.data)

        print("Calculating correlation matrix...")
        correlation_matrix = calculate_correlation_matrix(pressure_data)

    print(f"Correlation matrix shape: {correlation_matrix.shape}")

    # Create visualizer
    visualizer = InteractiveCorrelationVisualizer(coords, correlation_matrix, sensor_shape)
    visualizer.create_plot()

    if args.no_show:
        if args.output:
            print(f"Saving screenshot to {args.output}...")
            visualizer.fig.savefig(args.output, dpi=150, bbox_inches='tight')
            print("Saved.")
        else:
            print("No display requested (--no-show) and no output file specified.")
        # Close figure to free memory
        plt.close(visualizer.fig)
    else:
        # Show interactive plot
        if args.output:
            print("Warning: --output ignored when displaying interactive window. Use --no-show to save.")
        visualizer.show()


if __name__ == "__main__":
    main()