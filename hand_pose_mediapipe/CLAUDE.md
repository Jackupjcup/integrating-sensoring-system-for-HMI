# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a hand pose estimation and angle prediction system that combines:
- **MediaPipe Hand Tracking**: Real-time hand landmark detection from camera input
- **Finger angle calculation**: Computes 14 joint angles from hand landmarks
- **Pressure sensor integration**: Optional 14×10 pressure grid data collection
- **CNN regression model**: Predicts finger angles from pressure sensor data
- **Real-time visualization**: Live camera feed with angle overlays and predictions

The system collects synchronized hand angle and pressure data, trains a CNN regression model, and provides real-time angle predictions from pressure input.

## Dependencies

Core Python packages:
- `mediapipe` - Hand landmark detection
- `opencv-python` - Camera capture and video processing
- `numpy`, `pandas` - Data processing
- `torch`, `scikit-learn` - CNN regression model training
- `matplotlib` - Visualization and plotting

Install with:
```bash
pip install mediapipe opencv-python numpy pandas torch scikit-learn matplotlib
```

Hardware dependencies (optional):
- Camera for hand tracking
- Pressure sensor array (14×10 grid) for pressure data collection
- Serial interface for pressure sensor communication (COM port)

## Common Commands

### Data Collection
```bash
# Collect hand angles and pressure data (requires camera and pressure sensors)
python dataset_collector.py

# Collect only hand angles from camera
python bend_angle_collect.py

# Collect hand angles with real-time visualization
python realtime_handpose_cv.py
```

### Dataset Assembly
```bash
# Assemble individual session datasets into total_dataset.csv
python assamble_dataset.py

# Analyze angles distribution in the dataset
python analyze_angles_histogram.py
```

### Model Training
```bash
# Train CNN regression model to predict 14 angles from pressure data
python train_cnn_regression.py

# The model is defined in smallCNN_regression.py
```

### Real-time Prediction
```bash
# Run real-time pressure-to-angle prediction (requires trained model)
python realtime_pressure_angle_prediction_single_frame_copy.py
```

## Architecture Overview

### Data Flow

1. **Data Collection Phase**:
   - Camera captures hand images → MediaPipe detects 21 hand landmarks
   - Landmarks converted to 14 joint angles using geometric calculations (`calculate_finger_angles_mediapipe`)
   - Optional pressure sensor array (14×10 grid) provides simultaneous pressure readings
   - `HandAngleFilter` class applies median and exponential moving average filtering to smooth angles
   - Data saved to session directories in `record/` with timestamps

2. **Dataset Assembly**:
   - Individual session CSV files contain `pressure` (14×10 array) and `angles` (14 values) columns
   - `assamble_dataset.py` combines all sessions into `record/total_dataset.csv`
   - Pressure values are strings representing numpy arrays; angles are normalized [0,1]

3. **Model Training**:
   - `PressureAngleDataset` class parses CSV and applies pressure normalization
   - CNN architecture (`SmallCNN_regression`): 2 convolutional layers + 2 fully connected
   - Input: 1×14×10 pressure grid → Output: 14 continuous angle values
   - Training includes pressure normalization using training set statistics
   - Comprehensive evaluation: R², MAE, RMSE, MAPE, residual plots, 2×7 prediction grids

4. **Real-time Prediction**:
   - Load trained CNN model from `train/best_cnn_regression_model.pth`
   - Process live pressure data → predict angles → visualize results
   - `realtime_pressure_angle_prediction.py` integrates with parent system's sensor classes

### Key Components

1. **`FingerAngleCollector`** (`bend_angle_collect.py`)
   - Hand landmark detection using MediaPipe
   - Angle calculation from 21 landmarks to 14 joint angles
   - Session management with timestamped directories
   - Video recording synchronized with angle data

2. **`FingerAngleAndPressureSensorCollector`** (`AllData_collect_2.py`)
   - Extended collector with pressure sensor integration
   - Serial communication with pressure sensor array
   - Synchronized pressure and angle data collection

3. **`HandAngleFilter`** (`hand_angle_filter.py`)
   - Real-time angle filtering with median and exponential moving average
   - Velocity limiting to prevent unrealistic angle changes
   - Per-joint buffer management for smoothing

4. **`SmallCNN_regression`** (`smallCNN_regression.py`)
   - CNN architecture: Conv2d(1,16)→ReLU→MaxPool→Conv2d(16,32)→ReLU→MaxPool→Linear→ReLU→Dropout→Linear(14)
   - Input shape: (batch_size, 1, 14, 10)
   - Output shape: (batch_size, 14)

5. **`PressureAngleDataset`** (`train_cnn_regression.py`)
   - Custom PyTorch Dataset for pressure-angle pairs
   - Parses string representations of numpy arrays
   - Applies Z-score normalization to pressure data using training set statistics

6. **Data Assembly Pipeline** (`assamble_dataset.py`)
   - Scans `record/session_*/dataset.csv` files
   - Combines into single `total_dataset.csv`
   - Handles pressure array string formatting

### Configuration Files

- `mediapipe_config.json`: MediaPipe model configuration
  ```json
  {
    "model_asset_path": "hand_landmarker.task",
    "num_hands": 2,
    "min_hand_detection_confidence": 0.5,
    "min_hand_presence_confidence": 0.5,
    "min_tracking_confidence": 0.5
  }
  ```

## Data Organization

- `record/`: Root directory for all collected data
  - `session_YYYYMMDD_HHMMSS/`: Individual recording sessions
    - `angles_data.csv`: Hand angle measurements
    - `pressure_data.csv`: Pressure sensor readings (if available)
    - `dataset.csv`: Combined pressure and angles for training
    - `video.avi`: Recorded video (if enabled)
    - `session_info.json`: Session metadata
  - `total_dataset.csv`: Combined dataset for all sessions
  - `pressure_mean_std.txt`: Normalization statistics

- `train/`: Model training outputs
  - `best_cnn_regression_model.pth`: Best model checkpoint
  - `final_cnn_regression_model.pth`: Final trained model
  - `training_loss_curve.png`: Training/validation loss plot
  - `evaluation_report.txt/json`: Model evaluation metrics
  - `residual_plot.png`: Prediction residuals visualization
  - `prediction_vs_truth_all_angles.png`: 2×7 grid of predictions vs truth

## Angle Calculation Details

The system calculates 14 finger joint angles from MediaPipe's 21 hand landmarks:
- Thumb: CMC-MCP, MCP-IP
- Index finger: Wrist-MCP, MCP-PIP, PIP-DIP
- Middle finger: Wrist-MCP, MCP-PIP, PIP-DIP
- Ring finger: Wrist-MCP, MCP-PIP, PIP-DIP
- Pinky finger: Wrist-MCP, MCP-PIP, PIP-DIP

Angles are normalized to [0,1] range where 0 = minimum flexion, 1 = maximum flexion.

## Model Evaluation Metrics

The training pipeline generates comprehensive evaluation reports:
- **R² Score**: Coefficient of determination for each angle
- **MAE**: Mean Absolute Error (degrees)
- **RMSE**: Root Mean Square Error
- **MAPE**: Mean Absolute Percentage Error (handles zero values)
- **Residual plots**: Prediction error analysis
- **Prediction vs Truth**: 2×7 grid visualization of all 14 angles

## Integration with Parent System

The `realtime_pressure_angle_prediction.py` script imports components from the parent directory system:
- `ThreadingCollect_class` - Multi-threaded serial data collection
- `IMUProcessing.MPU6050_utils` - IMU sensor utilities
- `PalmPoints_and_FingerBend.palm_points_class` - Pressure sensor processing

This enables integration with the larger hand gesture recognition system that combines pressure sensors, IMUs, and Blender visualization. However, the core MediaPipe-based angle collection and CNN regression can operate independently.

## Development Patterns

- **String array storage**: Pressure and angles stored as string representations of numpy arrays in CSV
- **Session-based organization**: Each recording creates timestamped directory
- **Training-test split**: 70% train, 15% validation, 15% test with consistent random seed
- **Pressure normalization**: Z-score normalization using training set statistics only
- **Real-time processing**: Frame-by-frame processing with OpenCV video capture

## Key File References

- Data collection: `bend_angle_collect.py`, `AllData_collect.py`, `AllData_collect_2.py`, `dataset_collector.py`
- Real-time visualization: `realtime_handpose_cv.py`
- Angle filtering: `hand_angle_filter.py`
- Dataset assembly: `assamble_dataset.py`
- Model definition: `smallCNN_regression.py`
- Model training: `train_cnn_regression.py`
- Real-time prediction: `realtime_pressure_angle_prediction.py`
- Analysis tools: `analyze_angles_histogram.py`
- Configuration: `mediapipe_config.json`
- Documentation: `README_collector.md`