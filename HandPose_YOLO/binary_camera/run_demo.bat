@echo off
echo ========================================
echo 双摄像头实时姿态估计演示
echo ========================================
echo.
echo 请确保:
echo 1. 已完成摄像头标定 (运行 run_calibration.bat)
echo 2. 两个USB摄像头已连接
echo 3. YOLO模型文件存在 (yolo26n-pose.pt)
echo.
echo 按任意键开始实时演示...
pause >nul

python main_demo.py

echo.
echo ========================================
echo 演示结束
echo ========================================
pause