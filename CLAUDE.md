# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a real-time hand gesture recognition system that combines:
- **Pressure sensor array**: 14×10 grid (140 sensors) for palm pressure mapping
- **IMU sensors**: 3× MPU6050 for arm/hand orientation tracking
- **Blender integration**: 3D visualization and animation
- **CNN-based gesture classification**: 5 gesture classes
- **Advanced crosstalk mitigation**: For pressure sensor signal processing

The system performs real-time sensor fusion, crosstalk analysis/compensation, and provides multiple visualization modes (matplotlib, Blender).

## Dependencies

Core Python packages:
- `numpy`, `pandas`, `scipy` - Data processing
- `matplotlib`, `seaborn` - Visualization
- `ahrs` - IMU attitude estimation (Madgwick filter)
- `torch`, `scikit-learn` - Gesture recognition CNN
- `pyserial` - Serial communication with sensors
- `bpy` - Blender Python API (comes with Blender installation)

Install with:
```bash
pip install numpy pandas scipy matplotlib seaborn ahrs torch scikit-learn pyserial
```

Hardware dependencies:
- Pressure sensor array (14×10 grid) with serial interface
- 3× MPU6050 IMU sensors
- Serial port (default: COM4, 460800 baud)
- Blender 4.2+ for visualization

## Common Commands

### Data Collection & Real-time Visualization
```bash
# Collect gesture data with live pressure map
python Functions/Collect_Gesture.py

# Run Blender drawing application (requires Blender with bpy)
# Run from within Blender's Scripting workspace (bpy module not in system Python)
# Ensure the modules/ directory is in Blender's Python path
```

### Crosstalk Analysis & Mitigation
```bash
# Comprehensive crosstalk analysis
python analyze_pressure_crosstalk.py

# Demo complete crosstalk solution
python demo_crosstalk_solution.py

# Test crosstalk mitigation
python test_crosstalk_solution.py

# Visualize crosstalk matrix
python visualize_crosstalk_matrix.py
```

### IMU Processing
```bash
# Process IMU data with AHRS algorithms
python IMUProcessing/ahrs_imu.py

# Test IMU data collection
python IMUProcessing/imu_threading.py
```

### Gesture Recognition
```bash
# Train/evaluate gesture CNN
python Functions/identify_gesture_CNN.py

# Visualize pressure data
python Functions/illustrate_pressure_data.py
```

### Additional Visualization
```bash
# Pressure heatmap with gesture labels
python Functions/PressureHeat_GestureLabels.py

# Row/column pressure line plots
python Functions/PressureHeat_RowColsLines.py

# General pressure data illustration
python Functions/illustrate_pressure_data.py
```

### Testing & Debugging
```bash
# Run various test_on_dataset scripts
python general_test.py
python test_coordinate_plot.py
python PalmPoints_and_FingerBend/pp_test.py
python debug_coordinates.py
```

## Architecture Overview

### Core Components

1. **`ThreadingCollect`** (`ThreadingCollect_class.py`)
   - Multi-threaded data collection from serial port
   - Parses combined pressure frames and IMU packets
   - Thread-safe buffers with locks

2. **`IMUData`** (`IMUProcessing/MPU6050_utils.py`)
   - IMU sensor management and calibration
   - Quaternion conversion for orientation
   - Bias compensation

3. **`PalmPointsClass`** (`PalmPoints_and_FingerBend/palm_points_class.py`)
   - Pressure sensor array management
   - Integrates `RealTimePressureDenoiser` for crosstalk mitigation
   - Finger bending detection via change rate analysis

4. **`RealTimePressureDenoiser`** (`PalmPoints_and_FingerBend/palm_points_class.py`)
   - Real-time pressure denoising with adaptive filtering
   - Crosstalk compensation via `CrosstalkAnalyzer`
   - Time-domain smoothing with configurable window

5. **`CrosstalkAnalyzer`** (`PalmPoints_and_FingerBend/crosstalk_analyzer.py`)
   - Calculates 140×140 correlation matrix
   - Computes crosstalk and compensation matrices
   - Supports dynamic calibration updates

6. **`SmallCNN`** (`Functions/identify_gesture_CNN.py`)
   - Convolutional Neural Network for gesture classification
   - Trained on pressure patterns from `Hand_Gesture/` directories

### Data Flow

1. **Serial Acquisition**: `ThreadingCollect` reads from COM4, parses:
   - Pressure frames: `FRAME_START` → 14 rows → `FRAME_END`
   - IMU packets: `X1A{acc}G{gyro}` format per sensor

2. **Pressure Processing Pipeline**:
   - Raw pressure (14×10) → `RealTimePressureDenoiser`
   - Crosstalk analysis/compensation via correlation matrices
   - Adaptive spatial filtering based on local activity
   - Time-domain smoothing (5-frame window)

3. **IMU Processing Pipeline**:
   - Raw accel/gyro → bias compensation → quaternion conversion
   - Madgwick filter for attitude estimation

4. **Gesture Recognition**:
   - Processed pressure patterns → `SmallCNN` → gesture classification
   - 5 gesture classes stored in `Hand_Gesture/{gesture_name}/` directories

5. **Visualization**:
   - Real-time: `Collect_Gesture.py` (matplotlib heatmaps)
   - 3D: `BlenderDrawingApp.py` (Blender hand mesh animation)

### Configuration Files

- `IMUProcessing/config.json`: IMU calibration parameters, serial settings
- `PalmPoints_and_FingerBend/PalmPointsConfig.json`: Pressure sensor coordinates, zero bias, Blender mesh IDs

## Hardware Setup Notes

1. **Serial Connection**:
   - Pressure array and IMUs share COM4 at 460800 baud
   - Data format: interleaved pressure frames and IMU packets

2. **Calibration Requirements**:
   - IMU zero bias calibration via `ThreadingCollect.init_Gyro_calibration()`
   - Pressure sensor zero bias in `PalmPointsConfig.json`
   - Crosstalk calibration using known pressure patterns

3. **Blender Integration**:
   - Requires Blender 4.2+ with Python API
   - Hand mesh must have specific bone structure matching `PalmPointsConfig.json`
   - Run Blender scripts from within Blender's Python environment
   - **Directory layout**: Place the `modules/` directory alongside your `.blend` file; scripts expect this structure

## Data Organization

- `DataCSV/`: Raw IMU data collected from sensors
- `Functions/Hand_Gesture/`: Gesture datasets organized by gesture name
  - Subdirectories: `ideal_gesture_1/`, `ideal_gesture_2/`, etc.
  - Each contains timestamped CSV files with pressure frames
- `PressureDataCalibration.csv`: Calibration data for pressure sensors
- `PalmPoints_and_FingerBend/pressure_data_sequence.npy`: Time-series pressure data for crosstalk analysis
- `*.npz` files: Preprocessed datasets and trained models for CNN

## Development Patterns

- **Real-time Processing**: Use `deque` for frame buffers, threading locks for data safety
- **Crosstalk Analysis**: Correlation matrices reveal row-column coupling patterns
- **Adaptive Filtering**: Filter size adjusts based on local pressure activity
- **Modular Design**: Each sensor modality has separate processing classes
- **Configuration-driven**: All hardware parameters in JSON configs

## Key File References

- Main entry: `Functions/Collect_Gesture.py`
- Core pressure processing: `PalmPoints_and_FingerBend/palm_points_class.py`
- Crosstalk analysis: `analyze_pressure_crosstalk.py`, `crosstalk_analyzer.py`
- IMU utilities: `IMUProcessing/MPU6050_utils.py`
- Data collection: `ThreadingCollect_class.py`
- Blender integration: `Functions/BlenderDrawingApp.py`
- Gesture CNN: `Functions/identify_gesture_CNN.py`