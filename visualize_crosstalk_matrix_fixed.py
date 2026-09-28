#!/usr/bin/env python3
"""
Visualize crosstalk matrix as scatter plot using palm coordinates
Fixed version with optimizations and error handling.

This script loads the pressure data sequence, calculates the crosstalk matrix,
and displays it as a scatter plot where:
- X, Y positions are from PalmPointsConfig.json
- Point colors show the strength of crosstalk
- Point sizes can be adjusted based on correlation strength
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import json
import argparse
import sys
import os
import warnings
from pathlib import Path
from PalmPoints_and_FingerBend.palm_points_class import PalmPointsClass
from PalmPoints_and_FingerBend.crosstalk_analyzer import CrosstalkAnalyzer

# Suppress runtime warnings from numpy divide (zero variance sensors)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered')


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


def calculate_crosstalk_matrix(pressure_data):
    """
    Calculate crosstalk matrix efficiently using numpy corrcoef.

    Args:
        pressure_data: Time series pressure data, shape (num_frames, rows, cols)

    Returns:
        crosstalk_matrix: shape (num_sensors, num_sensors) with absolute correlations
    """
    num_frames, rows, cols = pressure_data.shape
    num_sensors = rows * cols

    # Reshape to (num_sensors, num_frames)
    sensor_data = pressure_data.reshape(num_frames, rows * cols).T  # shape (num_sensors, num_frames)

    # Calculate correlation matrix using numpy (much faster)
    correlation_matrix = np.corrcoef(sensor_data)  # shape (num_sensors, num_sensors)

    # Replace NaN with 0 (in case of constant signals) and ensure diagonal is 1
    np.fill_diagonal(correlation_matrix, 1.0)
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)

    # Create crosstalk matrix (absolute values)
    crosstalk_matrix = np.abs(correlation_matrix)

    return crosstalk_matrix


def calculate_crosstalk_scatter_plot(pressure_data, coords):
    """
    Calculate crosstalk matrix and create scatter plot visualization

    Args:
        pressure_data: Time series pressure data, shape (num_frames, 14, 10)
        coords: Palm coordinates for each sensor, shape (140, 2)

    Returns:
        fig, ax1, ax2: matplotlib figure and axes with the plot
        crosstalk_matrix: calculated crosstalk matrix
    """
    num_frames, rows, cols = pressure_data.shape
    num_sensors = rows * cols

    # Calculate correlation matrix efficiently
    print("Calculating correlation matrix...")
    crosstalk_matrix = calculate_crosstalk_matrix(pressure_data)
    print("Correlation matrix calculated.")

    # Create figure with subplots - adjust size for better aspect ratio
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Calculate axis limits with padding
    x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
    y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
    padding_x = (x_max - x_min) * 0.1
    padding_y = (y_max - y_min) * 0.1

    # Note: y-coordinates are flipped (-coords[:, 1]) in plots
    y_flipped_min = -y_max
    y_flipped_max = -y_min

    # Plot 1: Raw crosstalk values for each sensor
    # Use the correlation strength as color intensity
    # Exclude self-correlation (diagonal) when computing max crosstalk
    crosstalk_no_diag = crosstalk_matrix.copy()
    np.fill_diagonal(crosstalk_no_diag, 0)
    correlation_values = np.max(crosstalk_no_diag, axis=1)  # Max crosstalk for each sensor (excluding self)
    print(f"Correlation values range: {correlation_values.min():.3f} to {correlation_values.max():.3f}")

    scatter1 = ax1.scatter(
        coords[:, 0],  # x coordinates
        -coords[:, 1],  # y coordinates (negative to flip)
        c=correlation_values,  # color based on max crosstalk
        cmap='Reds',
        s=400,  # increased point size for larger figure
        alpha=0.8,
        edgecolors='black',
        linewidth=0.5
    )

    # Add colorbar
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Maximum Crosstalk Value', fontsize=14)

    # Add sensor indices as labels (skip for performance)
    # for i, (x, y) in enumerate(coords):
    #     ax1.text(x, -y, str(i), ha='center', va='center',
    #             fontsize=8, fontweight='bold')

    ax1.set_title('Maximum Crosstalk Strength per Sensor', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    # Set axis limits with padding
    ax1.set_xlim(x_min - padding_x, x_max + padding_x)
    ax1.set_ylim(y_flipped_min - padding_y, y_flipped_max + padding_y)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Plot 2: Pairwise crosstalk network
    # Show connections between sensors with high crosstalk
    threshold = 0.7  # Increased threshold to reduce clutter in network plot
    high_corr_pairs = np.where((crosstalk_matrix > threshold) & (crosstalk_matrix < 1.0))

    # Plot all sensors first
    ax2.scatter(
        coords[:, 0],
        -coords[:, 1],
        c='lightgray',
        s=300,  # increased point size for larger figure
        alpha=0.5,
        edgecolors='black',
        linewidth=0.5
    )

    # Draw connections for high crosstalk pairs
    for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
        if i < j:  # Avoid duplicate lines
            x_coords = [coords[i, 0], coords[j, 0]]
            y_coords = [-coords[i, 1], -coords[j, 1]]

            # Line thickness and color based on correlation strength
            linewidth = crosstalk_matrix[i, j] * 5  # increased for better visibility
            alpha = min(crosstalk_matrix[i, j], 1.0)

            ax2.plot(x_coords, y_coords, 'r-',
                    linewidth=linewidth, alpha=alpha)

    # Highlight the pressed point (11,7) - sensor 117
    pressed_sensor = 117  # 11*10 + 7 = 117
    ax2.scatter(
        coords[pressed_sensor, 0],
        -coords[pressed_sensor, 1],
        c='blue',
        s=700,  # increased size for prominence
        marker='*',
        edgecolors='darkblue',
        linewidth=2,
        label=f'Pressed point (11,7): {pressed_sensor}'
    )

    # Add legend and labels
    ax2.set_title(f'Crosstalk Network (threshold > {threshold})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    # Set axis limits with padding (same as ax1)
    ax2.set_xlim(x_min - padding_x, x_max + padding_x)
    ax2.set_ylim(y_flipped_min - padding_y, y_flipped_max + padding_y)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.legend(fontsize=12)

    plt.tight_layout(pad=3.0)

    return fig, ax1, ax2, crosstalk_matrix


def create_detailed_crosstalk_analysis(pressure_data, coords, press_point=(11, 7)):
    """
    Create detailed analysis of crosstalk for the pressed point
    """
    pressed_row, pressed_col = press_point
    pressed_idx = pressed_row * 10 + pressed_col

    # Calculate correlations for the pressed point efficiently
    print(f"Calculating correlations for pressed point ({pressed_row},{pressed_col})...")

    # Reshape pressure data to (num_frames, num_sensors)
    num_frames, rows, cols = pressure_data.shape
    sensor_data = pressure_data.reshape(num_frames, rows * cols)  # (num_frames, 140)

    # Get pressed point time series
    pressed_series = sensor_data[:, pressed_idx]

    # Compute correlations with all sensors using numpy corrcoef
    # corrcoef returns matrix (2, 2) for each pair; we can compute all at once
    # by computing full correlation matrix and extract row for pressed_idx
    full_corr = np.corrcoef(sensor_data.T)  # (140, 140)
    np.fill_diagonal(full_corr, 1.0)
    full_corr = np.nan_to_num(full_corr, nan=0.0)
    correlations = full_corr[pressed_idx, :].copy()

    print(f"Correlation calculation complete.")

    # Create figure with larger size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Calculate axis limits with padding (same as in calculate_crosstalk_scatter_plot)
    x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
    y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
    padding_x = (x_max - x_min) * 0.1
    padding_y = (y_max - y_min) * 0.1
    y_flipped_min = -y_max
    y_flipped_max = -y_min

    # Plot 1: All sensors colored by correlation with pressed point
    scatter_colors = correlations.copy()
    scatter_colors[pressed_idx] = 1.0  # Highlight the pressed point

    scatter1 = ax1.scatter(
        coords[:, 0],
        -coords[:, 1],
        c=scatter_colors,
        cmap='RdBu_r',
        s=400,  # increased point size for larger figure
        alpha=0.8,
        edgecolors='black',
        linewidth=0.5,
        vmin=-1,
        vmax=1
    )

    # Colorbar
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Correlation with Pressed Point', fontsize=14)

    # Add labels (skip for performance)
    # for i, (x, y) in enumerate(coords):
    #     color = 'white' if abs(correlations[i]) > 0.5 else 'black'
    #     ax1.text(x, -y, str(i), ha='center', va='center',
    #             fontsize=8, fontweight='bold', color=color)

    # Highlight pressed point
    ax1.scatter(
        coords[pressed_idx, 0],
        -coords[pressed_idx, 1],
        c='yellow',
        s=1000,  # increased size for prominence
        marker='*',
        edgecolors='orange',
        linewidth=3
    )

    ax1.set_title(f'Crosstalk from Pressed Point ({pressed_row},{pressed_col})',
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    # Set axis limits with padding
    ax1.set_xlim(x_min - padding_x, x_max + padding_x)
    ax1.set_ylim(y_flipped_min - padding_y, y_flipped_max + padding_y)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Plot 2: Histogram of correlation values
    ax2.hist(correlations, bins=40, alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.2)
    ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2,
               label='High correlation threshold (0.5)')
    ax2.axvline(x=-0.5, color='red', linestyle='--', linewidth=2)

    ax2.set_xlabel('Correlation Coefficient')
    ax2.set_ylabel('Number of Sensors')
    ax2.set_title('Distribution of Correlations with Pressed Point',
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)

    # Add statistics
    high_corr = np.sum(np.abs(correlations) > 0.5) - 1  # Exclude self
    max_corr = np.max(np.abs(correlations[correlations != 1.0]))

    stats_text = f'Sensors with |corr| > 0.5: {high_corr}\nMax correlation: {max_corr:.3f}'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout(pad=3.0)

    return fig, ax1, ax2, correlations


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


def main():
    """Main function to run the visualization"""
    parser = argparse.ArgumentParser(description='Visualize crosstalk matrix from pressure sensor data')
    parser.add_argument('--data',
                        # default='PalmPoints_and_FingerBend/For_correlation.npy',
                        default='PalmPoints_and_FingerBend/For_correlation.npy',
                        help='Path to pressure data NPY file (default: PalmPoints_and_FingerBend/For_correlation.npy)')
    parser.add_argument('--config', default='PalmPoints_and_FingerBend/PalmPointsConfig.json',
                        help='Path to palm coordinates config JSON (default: PalmPoints_and_FingerBend/PalmPointsConfig.json)')
    parser.add_argument('--output', default='crosstalk_analysis',
                        help='Output file prefix (default: crosstalk_analysis)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display interactive plots (just save files)')
    parser.add_argument('--press-point', default='11,7',
                        help='Pressed point as row,col (default: 11,7)')

    args = parser.parse_args()

    print("Starting crosstalk matrix visualization...")

    # Load pressure data
    print(f"Loading pressure data from {args.data}...")
    pressure_data = load_pressure_data(args.data)

    # Load palm coordinates
    print(f"Loading palm coordinates from {args.config}...")
    coords = load_palm_coordinates(args.config)
    print(f"Coordinates shape: {coords.shape}")

    # Parse pressed point
    try:
        press_point = tuple(map(int, args.press_point.split(',')))
        if len(press_point) != 2:
            raise ValueError
        if press_point[0] < 0 or press_point[0] >= 14 or press_point[1] < 0 or press_point[1] >= 10:
            print(f"Warning: Pressed point {press_point} may be outside sensor grid (14x10)")
    except ValueError:
        print(f"Error: Invalid press point format '{args.press_point}'. Use row,col (e.g., 11,7)")
        sys.exit(1)

    # Create main crosstalk visualization
    print("Creating main crosstalk visualization...")
    fig1, ax1, ax2, crosstalk_matrix = calculate_crosstalk_scatter_plot(pressure_data, coords)

    # Create detailed analysis for pressed point
    print(f"Creating detailed analysis for pressed point {press_point}...")
    fig2, ax1_det, ax2_det, correlations = create_detailed_crosstalk_analysis(pressure_data, coords, press_point)

    # Save figures
    print("Saving visualizations...")
    fig1.savefig(f'{args.output}_matrix_visualization.png', dpi=200, bbox_inches='tight')
    fig2.savefig(f'{args.output}_detailed_analysis.png', dpi=200, bbox_inches='tight')

    # Save data
    np.savez(f'{args.output}_results.npz',
             crosstalk_matrix=crosstalk_matrix,
             correlations=correlations,
             coordinates=coords,
             press_point=press_point)

    print("\nVisualization complete!")
    print("Saved files:")
    print(f"- {args.output}_matrix_visualization.png: Overall crosstalk pattern")
    print(f"- {args.output}_detailed_analysis.png: Analysis of pressed point {press_point}")
    print(f"- {args.output}_results.npz: Raw data for further analysis")

    # Display statistics
    print(f"\nCrosstalk Statistics:")
    print(f"- Maximum crosstalk value: {np.max(crosstalk_matrix):.3f}")
    print(f"- Average crosstalk: {np.mean(crosstalk_matrix):.3f}")
    print(f"- Sensors with high crosstalk (>0.5): {np.sum(crosstalk_matrix > 0.5)}")

    # Find sensors with most crosstalk
    max_crosstalk_per_sensor = np.max(crosstalk_matrix, axis=1)
    top_crosstalk_sensors = np.argsort(max_crosstalk_per_sensor)[-5:]
    print(f"\nTop 5 sensors with most crosstalk:")
    for sensor in top_crosstalk_sensors:
        row, col = sensor // 10, sensor % 10
        print(f"  Sensor {sensor} ({row},{col}): {max_crosstalk_per_sensor[sensor]:.3f}")

    # Show plots if not disabled
    if not args.no_show:
        print("Displaying interactive plots... Close plot windows to continue.")
        plt.show()


if __name__ == "__main__":
    main()