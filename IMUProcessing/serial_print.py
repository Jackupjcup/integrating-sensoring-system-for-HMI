"""
串口数据打印监视器
功能：
1. 实时显示原始数据流和解析后的矩阵数据
2. 支持压力帧 + IMU 帧双通道帧率检测
3. 支持串口发送功能（输入框 + 发送按钮）
"""

import sys
import re
import time
import numpy as np
import serial
import serial.tools.list_ports
from collections import deque
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QComboBox, QLabel,
                             QGroupBox, QTextEdit, QSplitter, QLineEdit, QSpinBox, QGridLayout)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QTextCursor


# ================= 配置区域 =================
ROWS = 14
COLS = 10
BAUDRATE = 460800
MAX_DISPLAY_LINES = 500
DEFAULT_IMU_PREFIXES = ["X1", "Y1", "Z1", "X2", "Y2", "Z2", "X3", "Y3", "Z3"]
# =============================


class FPSTracker:
    """简单帧率检测器"""
    def __init__(self, window_size=30):
        self.timestamps = deque(maxlen=window_size)
        self.total_frames = 0
        self.start_time = None
        self.last_time = None

    def start(self):
        self.start_time = time.time()
        self.last_time = time.time()
        self.timestamps.clear()
        self.total_frames = 0

    def record(self):
        now = time.time()
        self.total_frames += 1
        self.timestamps.append(now)
        self.last_time = now
        return self._calc_fps()

    def _calc_fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        span = self.timestamps[-1] - self.timestamps[0]
        if span <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / span


class IMUParser:
    """IMU 数据解析器"""
    def __init__(self, prefixes=None):
        self.prefixes = prefixes or DEFAULT_IMU_PREFIXES
        self.buffer = []
        self.collected = []

    def feed(self, line):
        if not line:
            return False, None
        for prefix in self.prefixes:
            if line.startswith(prefix):
                parsed = self._parse_line(prefix, line)
                if parsed:
                    self.buffer.append((prefix, parsed))
                    self.collected.append(prefix)
                    if self.collected == self.prefixes:
                        data = dict(self.buffer)
                        self.buffer.clear()
                        self.collected.clear()
                        return True, data
                break
        return False, None

    def _parse_line(self, prefix, line):
        rest = line[len(prefix):]
        match = re.match(r'A(-?\d+)G(-?\d+)', rest)
        if match:
            return {"A": int(match.group(1)), "G": int(match.group(2))}
        return None

    def reset(self):
        self.buffer.clear()
        self.collected.clear()


class SerialReader(QThread):
    """串口读取线程 - 双通道解析"""
    line_received = pyqtSignal(str)              # 原始行
    pressure_frame = pyqtSignal(np.ndarray)      # 压力帧
    imu_frame = pyqtSignal(dict)                 # IMU 帧
    connection_status = pyqtSignal(str)

    def __init__(self, port_name, frame_start="FRAME_START", frame_end="FRAME_END",
                 expected_rows=ROWS, expected_cols=COLS, imu_prefixes=None):
        super().__init__()
        self.port_name = port_name
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.expected_rows = expected_rows
        self.expected_cols = expected_cols
        self.imu_prefixes = imu_prefixes or DEFAULT_IMU_PREFIXES
        self.is_running = True
        self.serial_port = None  # 保存串口对象供发送使用

    def run(self):
        try:
            self.serial_port = serial.Serial(self.port_name, BAUDRATE, timeout=1)
            self.connection_status.emit(
                f"已连接: {self.port_name} | 压力 {self.expected_rows}x{self.expected_cols} | IMU {len(self.imu_prefixes)}通道")

            temp_matrix = []
            is_capturing = False
            imu_parser = IMUParser(self.imu_prefixes)

            while self.is_running:
                try:
                    line_bytes = self.serial_port.readline()
                    line = line_bytes.decode('utf-8', errors='ignore').strip()
                except Exception as e:
                    self.line_received.emit(f"[解码错误] {e}")
                    continue

                if not line:
                    continue

                self.line_received.emit(line)

                # IMU 解析
                imu_ok, imu_data = imu_parser.feed(line)
                if imu_ok and imu_data:
                    self.imu_frame.emit(imu_data)
                    continue

                # 压力帧解析
                if line == self.frame_start:
                    temp_matrix = []
                    is_capturing = True

                elif line == self.frame_end:
                    is_capturing = False
                    if len(temp_matrix) == self.expected_rows:
                        try:
                            final_data = np.array(temp_matrix, dtype=float)
                            self.pressure_frame.emit(final_data)
                        except Exception as e:
                            self.line_received.emit(f"[解析错误] {e}")

                elif is_capturing:
                    try:
                        vals = [int(x) for x in line.split() if x.isdigit()]
                        if len(vals) == self.expected_cols:
                            temp_matrix.append(vals)
                    except ValueError:
                        pass

            self.serial_port.close()
            self.connection_status.emit("串口已关闭")
        except Exception as e:
            self.connection_status.emit(f"[连接失败] {e}")

    def send(self, text):
        """发送文本到串口"""
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.write((text + "\n").encode('utf-8'))
                return True
            except Exception as e:
                print(f"发送失败: {e}")
                return False
        return False

    def stop(self):
        self.is_running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("串口监视器 - 双通道帧率 + 发送功能")
        self.resize(1100, 800)

        self.thread = None
        self.pressure_fps = FPSTracker()
        self.imu_fps = FPSTracker()

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout_main = QHBoxLayout(main_widget)

        # ===== 左侧控制面板 =====
        panel = QGroupBox("控制")
        panel.setFixedWidth(300)
        layout_panel = QVBoxLayout(panel)

        # 端口
        layout_panel.addWidget(QLabel("串口:"))
        self.combo_ports = QComboBox()
        self.refresh_ports()
        layout_panel.addWidget(self.combo_ports)

        btn_h = QHBoxLayout()
        btn_refresh = QPushButton("刷新端口")
        btn_refresh.clicked.connect(self.refresh_ports)
        btn_h.addWidget(btn_refresh)
        btn_clear = QPushButton("清空显示")
        btn_clear.clicked.connect(self.clear_display)
        btn_h.addWidget(btn_clear)
        layout_panel.addLayout(btn_h)

        layout_panel.addSpacing(8)

        # 压力帧设置
        p_group = QGroupBox("压力帧设置")
        p_layout = QGridLayout(p_group)
        p_layout.addWidget(QLabel("帧开始:"), 0, 0)
        self.edit_frame_start = QLineEdit("FRAME_START")
        p_layout.addWidget(self.edit_frame_start, 0, 1)
        p_layout.addWidget(QLabel("帧结束:"), 1, 0)
        self.edit_frame_end = QLineEdit("FRAME_END")
        p_layout.addWidget(self.edit_frame_end, 1, 1)
        p_layout.addWidget(QLabel("行数:"), 2, 0)
        self.spin_rows = QSpinBox()
        self.spin_rows.setRange(1, 100)
        self.spin_rows.setValue(ROWS)
        p_layout.addWidget(self.spin_rows, 2, 1)
        p_layout.addWidget(QLabel("列数:"), 3, 0)
        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(1, 100)
        self.spin_cols.setValue(COLS)
        p_layout.addWidget(self.spin_cols, 3, 1)
        layout_panel.addWidget(p_group)

        # IMU 设置
        imu_group = QGroupBox("IMU 帧设置")
        imu_layout = QVBoxLayout(imu_group)
        imu_layout.addWidget(QLabel("IMU前缀(逗号分隔):"))
        self.edit_imu_prefixes = QLineEdit(",".join(DEFAULT_IMU_PREFIXES))
        imu_layout.addWidget(self.edit_imu_prefixes)
        layout_panel.addWidget(imu_group)

        layout_panel.addSpacing(8)

        # 连接按钮
        self.btn_connect = QPushButton("连接")
        self.btn_connect.setCheckable(True)
        self.btn_connect.clicked.connect(self.toggle_connection)
        self.btn_connect.setStyleSheet(
            "background-color: #4CAF50; color: white; font-size: 14px; padding: 10px;")
        layout_panel.addWidget(self.btn_connect)

        layout_panel.addSpacing(8)

        # 帧率显示
        fps_group = QGroupBox("帧率统计")
        fps_layout = QGridLayout(fps_group)
        fps_layout.addWidget(QLabel("压力帧:"), 0, 0)
        self.lbl_pressure_fps = QLabel("0 FPS")
        self.lbl_pressure_fps.setStyleSheet("font-weight: bold; color: #2196F3;")
        fps_layout.addWidget(self.lbl_pressure_fps, 0, 1)
        fps_layout.addWidget(QLabel("IMU帧:"), 1, 0)
        self.lbl_imu_fps = QLabel("0 FPS")
        self.lbl_imu_fps.setStyleSheet("font-weight: bold; color: #9C27B0;")
        fps_layout.addWidget(self.lbl_imu_fps, 1, 1)
        layout_panel.addWidget(fps_group)

        # === 串口发送区域 ===
        send_group = QGroupBox("串口发送")
        send_layout = QVBoxLayout(send_group)

        send_h = QHBoxLayout()
        self.edit_send = QLineEdit()
        self.edit_send.setPlaceholderText("输入要发送的内容...")
        self.edit_send.returnPressed.connect(self.send_text)
        send_h.addWidget(self.edit_send)

        self.btn_send = QPushButton("发送")
        self.btn_send.clicked.connect(self.send_text)
        self.btn_send.setEnabled(False)
        send_h.addWidget(self.btn_send)
        send_layout.addLayout(send_h)

        # 快捷命令按钮
        quick_h = QHBoxLayout()
        for cmd in ["START", "STOP", "RESET"]:
            btn = QPushButton(cmd)
            btn.clicked.connect(lambda checked, c=cmd: self.send_quick(c))
            btn.setEnabled(False)
            setattr(self, f"btn_quick_{cmd.lower()}", btn)
            quick_h.addWidget(btn)
        send_layout.addLayout(quick_h)

        layout_panel.addWidget(send_group)

        layout_panel.addStretch()

        self.status_label = QLabel("状态: 未连接")
        layout_panel.addWidget(self.status_label)

        layout_main.addWidget(panel)

        # ===== 右侧显示区域 =====
        splitter = QSplitter(Qt.Orientation.Vertical)

        # 原始数据
        raw_group = QGroupBox("原始数据流")
        raw_layout = QVBoxLayout(raw_group)
        self.raw_text = QTextEdit()
        self.raw_text.setFont(QFont("Consolas", 10))
        self.raw_text.setReadOnly(True)
        self.raw_text.setStyleSheet("background-color: #1e1e1e; color: #00ff00;")
        raw_layout.addWidget(self.raw_text)
        splitter.addWidget(raw_group)

        # 解析结果
        self.matrix_group = QGroupBox(f"压力矩阵 ({ROWS}x{COLS})")
        matrix_layout = QVBoxLayout(self.matrix_group)
        self.matrix_text = QTextEdit()
        self.matrix_text.setFont(QFont("Consolas", 10))
        self.matrix_text.setReadOnly(True)
        self.matrix_text.setStyleSheet("background-color: #f5f5f5; color: #333;")
        matrix_layout.addWidget(self.matrix_text)
        splitter.addWidget(self.matrix_group)

        # IMU 数据
        imu_display_group = QGroupBox("IMU 数据")
        imu_display_layout = QVBoxLayout(imu_display_group)
        self.imu_text = QTextEdit()
        self.imu_text.setFont(QFont("Consolas", 10))
        self.imu_text.setReadOnly(True)
        self.imu_text.setStyleSheet("background-color: #fff3e0; color: #333;")
        self.imu_text.setMaximumHeight(150)
        imu_display_layout.addWidget(self.imu_text)
        splitter.addWidget(imu_display_group)

        splitter.setSizes([350, 250, 150])
        layout_main.addWidget(splitter)

    def _parse_imu_prefixes(self):
        text = self.edit_imu_prefixes.text().strip()
        if not text:
            return DEFAULT_IMU_PREFIXES
        prefixes = [p.strip() for p in text.split(",") if p.strip()]
        return prefixes if prefixes else DEFAULT_IMU_PREFIXES

    def refresh_ports(self):
        self.combo_ports.clear()
        ports = serial.tools.list_ports.comports()
        for p in ports:
            self.combo_ports.addItem(p.device)
        if self.combo_ports.count() == 0:
            self.combo_ports.addItem("无可用端口")

    def clear_display(self):
        self.raw_text.clear()
        self.matrix_text.clear()
        self.imu_text.clear()
        self.lbl_pressure_fps.setText("0 FPS")
        self.lbl_imu_fps.setText("0 FPS")

    def toggle_connection(self):
        if self.btn_connect.isChecked():
            port = self.combo_ports.currentText()
            if not port or "无可用" in port:
                self.btn_connect.setChecked(False)
                return

            frame_start = self.edit_frame_start.text().strip()
            frame_end = self.edit_frame_end.text().strip()
            rows = self.spin_rows.value()
            cols = self.spin_cols.value()
            imu_prefixes = self._parse_imu_prefixes()

            self.matrix_group.setTitle(f"压力矩阵 ({rows}x{cols})")

            self.pressure_fps.start()
            self.imu_fps.start()

            self.thread = SerialReader(
                port,
                frame_start=frame_start,
                frame_end=frame_end,
                expected_rows=rows,
                expected_cols=cols,
                imu_prefixes=imu_prefixes
            )
            self.thread.line_received.connect(self.on_line_received)
            self.thread.pressure_frame.connect(self.on_pressure_frame)
            self.thread.imu_frame.connect(self.on_imu_frame)
            self.thread.connection_status.connect(lambda s: self.status_label.setText(f"状态: {s}"))
            self.thread.start()

            self.btn_connect.setText("断开")
            self.btn_connect.setStyleSheet(
                "background-color: #f44336; color: white; font-size: 14px; padding: 10px;")

            # 启用发送按钮
            self.btn_send.setEnabled(True)
            self.btn_quick_start.setEnabled(True)
            self.btn_quick_stop.setEnabled(True)
            self.btn_quick_reset.setEnabled(True)
        else:
            if self.thread:
                self.thread.stop()
                self.thread.wait()
                self.thread = None

            self.btn_connect.setText("连接")
            self.btn_connect.setStyleSheet(
                "background-color: #4CAF50; color: white; font-size: 14px; padding: 10px;")
            self.status_label.setText("状态: 已断开")

            # 禁用发送按钮
            self.btn_send.setEnabled(False)
            self.btn_quick_start.setEnabled(False)
            self.btn_quick_stop.setEnabled(False)
            self.btn_quick_reset.setEnabled(False)

    def send_text(self):
        text = self.edit_send.text().strip()
        if text and self.thread:
            if self.thread.send(text):
                self.raw_text.append(f'<span style="color:#2196F3;">[发送] {text}</span>')
                self.edit_send.clear()
            else:
                self.raw_text.append('<span style="color:#f44336;">[发送失败]</span>')

    def send_quick(self, cmd):
        if self.thread:
            if self.thread.send(cmd):
                self.raw_text.append(f'<span style="color:#2196F3;">[发送] {cmd}</span>')
            else:
                self.raw_text.append('<span style="color:#f44336;">[发送失败]</span>')

    def on_line_received(self, line):
        frame_start = self.edit_frame_start.text().strip()
        frame_end = self.edit_frame_end.text().strip()
        imu_prefixes = self._parse_imu_prefixes()

        # 高亮特殊行
        if line == frame_start or line == frame_end:
            display = f'<span style="color:#ff9800;">{line}</span>'
        elif any(line.startswith(p) for p in imu_prefixes):
            display = f'<span style="color:#9C27B0;">{line}</span>'
        else:
            display = f'<span style="color:#00ff00;">{line}</span>'

        self.raw_text.append(display)

        if self.raw_text.document().lineCount() > MAX_DISPLAY_LINES:
            cursor = self.raw_text.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            cursor.movePosition(QTextCursor.MoveOperation.Down, QTextCursor.MoveMode.KeepAnchor, 100)
            cursor.removeSelectedText()

        self.raw_text.moveCursor(QTextCursor.MoveOperation.End)

    def on_pressure_frame(self, matrix):
        fps = self.pressure_fps.record()
        self.lbl_pressure_fps.setText(f"{fps:.1f} FPS")

        header = f"=== 压力帧 | FPS: {fps:.1f} ===\n"
        self.matrix_text.setText(header + self.format_matrix(matrix))

    def on_imu_frame(self, imu_data):
        fps = self.imu_fps.record()
        self.lbl_imu_fps.setText(f"{fps:.1f} FPS")

        lines = [f"=== IMU 帧 | FPS: {fps:.1f} ==="]
        for prefix in self._parse_imu_prefixes():
            if prefix in imu_data:
                a = imu_data[prefix]["A"]
                g = imu_data[prefix]["G"]
                lines.append(f"{prefix}: A={a:+6d}  G={g:+5d}")
        self.imu_text.setText("\n".join(lines))

    def format_matrix(self, matrix):
        lines = []
        for r, row in enumerate(matrix):
            row_str = " ".join(f"{int(v):5d}" for v in row)
            lines.append(f"行{r:2d}: {row_str}")
        lines.append("\n--- 统计 ---")
        lines.append(f"最大值: {int(matrix.max())}")
        lines.append(f"最小值: {int(matrix.min())}")
        lines.append(f"平均值: {matrix.mean():.1f}")
        lines.append(f"总和:   {int(matrix.sum())}")
        return "\n".join(lines)

    def closeEvent(self, event):
        if self.thread:
            self.thread.stop()
            self.thread.wait()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
