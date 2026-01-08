#!/usr/bin/env python3
"""
Hospital Delivery Robot Control Interface
Professional Medical-Grade UI with ROS2 Integration
Dark Theme - 24/7 Operation Optimized
"""

import sys
import threading
import time
from datetime import datetime
from collections import deque
from enum import Enum

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QListWidget,
    QTableWidget, QTableWidgetItem, QTextEdit, QProgressBar,
    QFrame, QSplitter, QGroupBox, QMessageBox, QListWidgetItem,
    QSizePolicy, QSpacerItem, QHeaderView, QAbstractItemView
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QSize
from PyQt6.QtGui import QFont, QColor, QPixmap, QImage, QPalette, QIcon

# ROS2 Imports
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, BatteryState
from std_msgs.msg import String
from action_msgs.msg import GoalStatus

# Custom Action Interface
from moma_interfaces.action import RunDelivery

import cv2
import numpy as np
from cv_bridge import CvBridge

# =============================================================================
# CONSTANTS
# =============================================================================

ITEM_TYPES = {
    "Blood Sample (Emergency)": {"icon": "🩸", "priority": "HIGH", "speed": "Slow", "dest_hint": "Clinical Lab"},
    "General Medicine": {"icon": "💊", "priority": "NORMAL", "speed": "Normal", "dest_hint": "Ward 10x"},
    "Narcotics (Secure)": {"icon": "🔒", "priority": "CRITICAL", "speed": "Fast", "dest_hint": "Doctor's Office"},
    "Surgical Kit": {"icon": "✂️", "priority": "NORMAL", "speed": "Normal", "dest_hint": "Operating Room"},
    "Documents/Chart": {"icon": "📄", "priority": "LOW", "speed": "Max", "dest_hint": "Doctor's Office"},
}

LOCATIONS = [
    "Nurse Station A (Base)", "Ward 101", "Ward 102", "Ward 103", 
    "Ward 104", "Ward 105", "Main Pharmacy (Central)", "Sub Pharmacy",
    "Clinical Lab (Zone C)", "Central Supply", "Doctor's Office"
]

PATIENTS = {
    "PT-2401": {"name": "김철수", "ward": "Ward 101", "condition": "Stable"},
    "PT-2402": {"name": "이영희", "ward": "Ward 102", "condition": "Post-Op"},
    "PT-2403": {"name": "박지성", "ward": "Ward 105", "condition": "Critical"},
    "PT-2404": {"name": "최민아", "ward": "Ward 102", "condition": "Check-up"},
}

# Task Modes for Step Execution
TASK_MODES = {
    "ALL": "Full Sequence",
    "NAV_PICKUP_CONT": "1. Navigate to Pickup →",
    "DOCK_PICKUP_CONT": "2. Dock at Pickup →",
    "PICK_CONT": "3. Scan & Pick →",
    "NAV_DROPOFF_CONT": "4. Navigate to Dropoff →",
    "DOCK_DROPOFF_CONT": "5. Dock at Dropoff →",
    "PLACE": "6. Place Item",
    "HOME": "Return Home",
}

# State to Progress mapping
STATE_PROGRESS = {
    "IDLE": 0,
    "NAVIGATING TO PICKUP": 15,
    "DOCKING AT PICKUP": 30,
    "SCANNING & PICKING": 45,
    "NAVIGATING TO DROPOFF": 60,
    "DOCKING AT DROPOFF": 75,
    "PLACING": 90,
    "RETURNING HOME": 95,
    "COMPLETED": 100,
    "ERROR": 0,
}

# Camera overlay text mapping
STATE_OVERLAY = {
    "IDLE": "STANDBY",
    "NAVIGATING TO PICKUP": "NAV2: Moving to Pickup",
    "DOCKING AT PICKUP": "VISION: Precision Docking",
    "SCANNING & PICKING": "ARM: Picking Item",
    "NAVIGATING TO DROPOFF": "NAV2: Delivering...",
    "DOCKING AT DROPOFF": "VISION: Precision Docking",
    "PLACING": "ARM: Placing Item",
    "RETURNING HOME": "NAV2: Return to Base",
    "COMPLETED": "JOB DONE ✓",
    "CANCELED": "CANCELED",
    "ERROR": "ERROR",
}

# Camera selection per state
STATE_CAMERA = {
    "IDLE": "front",
    "NAVIGATING TO PICKUP": "front",
    "DOCKING AT PICKUP": "front",  # April tag uses front camera
    "SCANNING & PICKING": "auto",  # Based on work_side
    "NAVIGATING TO DROPOFF": "front",
    "DOCKING AT DROPOFF": "front",
    "PLACING": "auto",
    "RETURNING HOME": "front",
}

# =============================================================================
# STYLE SHEETS
# =============================================================================

DARK_STYLE = """
QMainWindow {
    background-color: #0d1117;
}
QWidget {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: 'Segoe UI', 'Malgun Gothic', sans-serif;
}
QGroupBox {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 10px;
    font-weight: bold;
    font-size: 13px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
    color: #58a6ff;
}
QLabel {
    color: #c9d1d9;
}
QComboBox {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 8px 12px;
    min-height: 20px;
    color: #c9d1d9;
}
QComboBox:hover {
    border-color: #58a6ff;
}
QComboBox::drop-down {
    border: none;
    width: 30px;
}
QComboBox QAbstractItemView {
    background-color: #21262d;
    border: 1px solid #30363d;
    selection-background-color: #388bfd;
}
QPushButton {
    background-color: #238636;
    border: none;
    border-radius: 6px;
    padding: 10px 16px;
    color: white;
    font-weight: bold;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #2ea043;
}
QPushButton:pressed {
    background-color: #1a7f37;
}
QPushButton:disabled {
    background-color: #21262d;
    color: #484f58;
}
QPushButton#emergency {
    background-color: #da3633;
    font-size: 16px;
    padding: 15px;
}
QPushButton#emergency:hover {
    background-color: #f85149;
}
QPushButton#stepBtn {
    background-color: #1f6feb;
    padding: 8px 12px;
    font-size: 11px;
}
QPushButton#stepBtn:hover {
    background-color: #388bfd;
}
QPushButton#deleteBtn {
    background-color: #6e7681;
}
QPushButton#deleteBtn:hover {
    background-color: #8b949e;
}
QListWidget {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 4px;
}
QListWidget::item {
    padding: 8px;
    border-radius: 4px;
    margin: 2px;
}
QListWidget::item:selected {
    background-color: #388bfd;
}
QListWidget::item:hover {
    background-color: #21262d;
}
QTableWidget {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    gridline-color: #30363d;
}
QTableWidget::item {
    padding: 8px;
}
QTableWidget::item:selected {
    background-color: #388bfd;
}
QHeaderView::section {
    background-color: #21262d;
    color: #8b949e;
    padding: 8px;
    border: none;
    border-bottom: 1px solid #30363d;
    font-weight: bold;
}
QTextEdit {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 8px;
    font-family: 'Consolas', 'D2Coding', monospace;
    font-size: 11px;
    color: #8b949e;
}
QProgressBar {
    background-color: #21262d;
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #238636;
    border-radius: 4px;
}
QFrame#statusCard {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px;
}
QFrame#cameraFrame {
    background-color: #000000;
    border: 2px solid #30363d;
    border-radius: 8px;
}
QLabel#stateLabel {
    font-size: 18px;
    font-weight: bold;
    color: #58a6ff;
}
QLabel#batteryLabel {
    font-size: 14px;
}
QLabel#itemInfo {
    color: #8b949e;
    font-size: 12px;
}
"""

# =============================================================================
# ROS2 WORKER NODE
# =============================================================================

class ROS2Signals(QObject):
    """Qt Signals for ROS2 callbacks"""
    log_signal = pyqtSignal(str)
    state_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    battery_signal = pyqtSignal(int)
    cam_overlay_signal = pyqtSignal(str)
    task_finished_signal = pyqtSignal(bool, str)
    front_image_signal = pyqtSignal(np.ndarray)
    left_image_signal = pyqtSignal(np.ndarray)
    right_image_signal = pyqtSignal(np.ndarray)


class RobotWorker(Node):
    """ROS2 Node for robot communication"""
    
    def __init__(self, signals: ROS2Signals):
        super().__init__('hospital_ui_node')
        self.signals = signals
        self.bridge = CvBridge()
        
        self.cb_group = ReentrantCallbackGroup()
        
        # Action Client
        self.action_client = ActionClient(
            self, RunDelivery, 'run_delivery',
            callback_group=self.cb_group
        )
        
        # Camera Subscriptions
        self.create_subscription(
            Image, '/front_camera/rgb', 
            self.front_camera_callback, 10,
            callback_group=self.cb_group
        )
        self.create_subscription(
            Image, '/left_camera/rgb',
            self.left_camera_callback, 10,
            callback_group=self.cb_group
        )
        self.create_subscription(
            Image, '/right_camera/rgb',
            self.right_camera_callback, 10,
            callback_group=self.cb_group
        )
        
        # Battery Subscription (optional)
        self.create_subscription(
            BatteryState, '/battery_state',
            self.battery_callback, 10,
            callback_group=self.cb_group
        )
        
        # Current goal handle
        self.current_goal_handle = None
        self.is_executing = False
        
        self.get_logger().info("🏥 Hospital UI Node Initialized")
        self.signals.log_signal.emit("[SYSTEM] ROS2 Node Connected")
        
    def front_camera_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.signals.front_image_signal.emit(cv_image)
        except Exception as e:
            pass
            
    def left_camera_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.signals.left_image_signal.emit(cv_image)
        except Exception as e:
            pass
            
    def right_camera_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.signals.right_image_signal.emit(cv_image)
        except Exception as e:
            pass
            
    def battery_callback(self, msg):
        percentage = int(msg.percentage * 100)
        self.signals.battery_signal.emit(percentage)
        
    def send_goal(self, task_mode: str, item_type: str, pickup: str, dropoff: str):
        """Send delivery goal to action server"""
        if not self.action_client.wait_for_server(timeout_sec=2.0):
            self.signals.log_signal.emit("[ERROR] Action server not available!")
            self.signals.task_finished_signal.emit(False, "Server not available")
            return
            
        goal = RunDelivery.Goal()
        goal.task_mode = task_mode
        goal.item_type = item_type
        goal.pickup_loc = pickup
        goal.dropoff_loc = dropoff
        
        self.signals.log_signal.emit(f"[DISPATCH] Mode: {task_mode}")
        self.signals.log_signal.emit(f"[DISPATCH] Item: {item_type}")
        self.signals.log_signal.emit(f"[DISPATCH] Route: {pickup} → {dropoff}")
        self.signals.log_signal.emit("[ACTION] 📡 Sending Goal...")
        
        self.is_executing = True
        
        send_future = self.action_client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback
        )
        send_future.add_done_callback(self.goal_response_callback)
        
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.signals.log_signal.emit("[ERROR] ❌ Goal Rejected!")
            self.signals.state_signal.emit("ERROR")
            self.signals.task_finished_signal.emit(False, "Goal rejected")
            self.is_executing = False
            return
            
        self.signals.log_signal.emit("[ACTION] ✅ Goal Accepted. Executing...")
        self.current_goal_handle = goal_handle
        
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)
        
    def feedback_callback(self, feedback_msg):
        state = feedback_msg.feedback.current_state
        self.signals.log_signal.emit(f"[STATE] ▶ {state}")
        self.signals.state_signal.emit(state)
        
        # Update progress
        progress = STATE_PROGRESS.get(state, 0)
        self.signals.progress_signal.emit(progress)
        
        # Update camera overlay
        overlay = STATE_OVERLAY.get(state, state)
        self.signals.cam_overlay_signal.emit(overlay)
        
    def result_callback(self, future):
        result = future.result().result
        self.is_executing = False
        self.current_goal_handle = None
        
        if result.success:
            self.signals.log_signal.emit(f"[RESULT] 🎉 {result.message}")
            self.signals.state_signal.emit("COMPLETED")
            self.signals.progress_signal.emit(100)
            self.signals.cam_overlay_signal.emit("JOB DONE ✓")
        else:
            self.signals.log_signal.emit(f"[RESULT] 🛑 {result.message}")
            self.signals.state_signal.emit("ERROR")
            self.signals.cam_overlay_signal.emit("ERROR")
            
        self.signals.task_finished_signal.emit(result.success, result.message)
        
    def cancel_current_goal(self):
        """Cancel current executing goal"""
        if self.current_goal_handle is not None:
            self.signals.log_signal.emit("[CANCEL] 🛑 Canceling current task...")
            cancel_future = self.current_goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self.cancel_callback)
            
    def cancel_callback(self, future):
        cancel_response = future.result()
        self.signals.log_signal.emit("[CANCEL] Task canceled.")
        self.signals.state_signal.emit("CANCELED")
        self.signals.cam_overlay_signal.emit("CANCELED")
        self.is_executing = False
        self.current_goal_handle = None


class ROS2Thread(QThread):
    """Thread for ROS2 spinning"""
    
    def __init__(self, node: RobotWorker):
        super().__init__()
        self.node = node
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(node)
        self._running = True
        
    def run(self):
        while self._running and rclpy.ok():
            self.executor.spin_once(timeout_sec=0.1)
            
    def stop(self):
        self._running = False
        self.executor.shutdown()


# =============================================================================
# MAIN UI WINDOW
# =============================================================================

class HospitalRobotUI(QMainWindow):
    def __init__(self, ros_signals: ROS2Signals, ros_worker: RobotWorker):
        super().__init__()
        
        self.ros_signals = ros_signals
        self.ros_worker = ros_worker
        
        # Task Queue
        self.task_queue = deque()
        self.current_task_index = -1
        
        # Camera frames storage
        self.front_frame = None
        self.left_frame = None
        self.right_frame = None
        self.current_camera = "front"
        self.current_work_side = "Left"
        
        # State
        self.current_state = "IDLE"
        self.battery_level = 100
        
        self.setup_ui()
        self.connect_signals()
        self.setup_timers()
        
        # Initial log
        self.add_log("[SYSTEM] Hospital Robot Control System v2.0")
        self.add_log("[SYSTEM] UI Initialized - Ready for Operation")
        
    def setup_ui(self):
        self.setWindowTitle("🏥 Hospital Delivery Robot Control System")
        self.setMinimumSize(1600, 900)
        self.setStyleSheet(DARK_STYLE)
        
        # Central Widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main Layout - 3 Column Design
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)
        
        # LEFT PANEL (Task Control)
        left_panel = self.create_left_panel()
        left_panel.setFixedWidth(380)
        
        # CENTER PANEL (Camera & Status)
        center_panel = self.create_center_panel()
        
        # RIGHT PANEL (Queue & Logs)
        right_panel = self.create_right_panel()
        right_panel.setFixedWidth(400)
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(center_panel, 1)
        main_layout.addWidget(right_panel)
        
    def create_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)
        
        # === TASK DISPATCH ===
        dispatch_group = QGroupBox("📋 Task Dispatch")
        dispatch_layout = QVBoxLayout(dispatch_group)
        dispatch_layout.setSpacing(10)
        
        # Item Selection
        item_label = QLabel("Item Type")
        item_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        self.item_combo = QComboBox()
        for item, info in ITEM_TYPES.items():
            self.item_combo.addItem(f"{info['icon']} {item}")
        self.item_combo.currentTextChanged.connect(self.update_item_info)
        
        # Item Info Display
        self.item_info_label = QLabel()
        self.item_info_label.setObjectName("itemInfo")
        self.update_item_info(self.item_combo.currentText())
        
        # Pickup Location
        pickup_label = QLabel("Pickup Location")
        pickup_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        self.pickup_combo = QComboBox()
        self.pickup_combo.addItems(LOCATIONS)
        
        # Dropoff Location
        dropoff_label = QLabel("Dropoff Location")
        dropoff_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        self.dropoff_combo = QComboBox()
        self.dropoff_combo.addItems(LOCATIONS)
        self.dropoff_combo.setCurrentIndex(1)
        
        # Dispatch Button
        self.dispatch_btn = QPushButton("🚀 Dispatch Work Order")
        self.dispatch_btn.clicked.connect(self.dispatch_task)
        
        dispatch_layout.addWidget(item_label)
        dispatch_layout.addWidget(self.item_combo)
        dispatch_layout.addWidget(self.item_info_label)
        dispatch_layout.addWidget(pickup_label)
        dispatch_layout.addWidget(self.pickup_combo)
        dispatch_layout.addWidget(dropoff_label)
        dispatch_layout.addWidget(self.dropoff_combo)
        dispatch_layout.addWidget(self.dispatch_btn)
        
        # === STEP EXECUTION ===
        step_group = QGroupBox("🎮 Step-by-Step Execution")
        step_layout = QGridLayout(step_group)
        step_layout.setSpacing(6)
        
        step_buttons = [
            ("1. Nav to Pickup", "NAV_PICKUP_CONT"),
            ("2. Dock Pickup", "DOCK_PICKUP_CONT"),
            ("3. Scan & Pick", "PICK_CONT"),
            ("4. Nav to Dropoff", "NAV_DROPOFF_CONT"),
            ("5. Dock Dropoff", "DOCK_DROPOFF_CONT"),
            ("6. Place Item", "PLACE"),
        ]
        
        self.step_btns = []
        for i, (label, mode) in enumerate(step_buttons):
            btn = QPushButton(label)
            btn.setObjectName("stepBtn")
            btn.setProperty("mode", mode)
            btn.clicked.connect(lambda checked, m=mode: self.execute_step(m))
            step_layout.addWidget(btn, i // 2, i % 2)
            self.step_btns.append(btn)
            
        # Home button
        home_btn = QPushButton("🏠 Return Home")
        home_btn.setObjectName("stepBtn")
        home_btn.clicked.connect(lambda: self.execute_step("HOME"))
        step_layout.addWidget(home_btn, 3, 0, 1, 2)
        self.step_btns.append(home_btn)
        
        # === PATIENT INFO ===
        patient_group = QGroupBox("👤 Patient Quick Select")
        patient_layout = QVBoxLayout(patient_group)
        
        self.patient_table = QTableWidget()
        self.patient_table.setColumnCount(3)
        self.patient_table.setHorizontalHeaderLabels(["Name", "Ward", "Status"])
        self.patient_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.patient_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.patient_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.patient_table.cellClicked.connect(self.auto_fill_destination)
        self.load_patient_data()
        
        patient_layout.addWidget(self.patient_table)
        
        layout.addWidget(dispatch_group)
        layout.addWidget(step_group)
        layout.addWidget(patient_group)
        layout.addStretch()
        
        return panel
        
    def create_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)
        
        # === STATUS BAR ===
        status_frame = QFrame()
        status_frame.setObjectName("statusCard")
        status_layout = QHBoxLayout(status_frame)
        
        # Robot State
        state_container = QVBoxLayout()
        state_title = QLabel("Robot Status")
        state_title.setStyleSheet("color: #8b949e; font-size: 11px;")
        self.state_label = QLabel("IDLE")
        self.state_label.setObjectName("stateLabel")
        state_container.addWidget(state_title)
        state_container.addWidget(self.state_label)
        
        # Battery
        battery_container = QVBoxLayout()
        battery_title = QLabel("Battery")
        battery_title.setStyleSheet("color: #8b949e; font-size: 11px;")
        self.battery_label = QLabel("🔋 100%")
        self.battery_label.setObjectName("batteryLabel")
        battery_container.addWidget(battery_title)
        battery_container.addWidget(self.battery_label)
        
        # Progress
        progress_container = QVBoxLayout()
        progress_title = QLabel("Task Progress")
        progress_title.setStyleSheet("color: #8b949e; font-size: 11px;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        progress_container.addWidget(progress_title)
        progress_container.addWidget(self.progress_bar)
        
        status_layout.addLayout(state_container)
        status_layout.addLayout(battery_container)
        status_layout.addLayout(progress_container, 1)
        
        # === CAMERA VIEW ===
        camera_group = QGroupBox("📹 Live Camera Feed")
        camera_layout = QVBoxLayout(camera_group)
        
        # Camera selector
        cam_select_layout = QHBoxLayout()
        cam_select_layout.addWidget(QLabel("Camera:"))
        self.cam_front_btn = QPushButton("Front")
        self.cam_left_btn = QPushButton("Left")
        self.cam_right_btn = QPushButton("Right")
        self.cam_auto_btn = QPushButton("Auto")
        
        for btn in [self.cam_front_btn, self.cam_left_btn, self.cam_right_btn, self.cam_auto_btn]:
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton { background-color: #21262d; padding: 6px 12px; }
                QPushButton:checked { background-color: #1f6feb; }
            """)
            
        self.cam_auto_btn.setChecked(True)
        self.cam_front_btn.clicked.connect(lambda: self.set_camera("front"))
        self.cam_left_btn.clicked.connect(lambda: self.set_camera("left"))
        self.cam_right_btn.clicked.connect(lambda: self.set_camera("right"))
        self.cam_auto_btn.clicked.connect(lambda: self.set_camera("auto"))
        
        cam_select_layout.addWidget(self.cam_front_btn)
        cam_select_layout.addWidget(self.cam_left_btn)
        cam_select_layout.addWidget(self.cam_right_btn)
        cam_select_layout.addWidget(self.cam_auto_btn)
        cam_select_layout.addStretch()
        
        # Camera display
        self.camera_frame = QFrame()
        self.camera_frame.setObjectName("cameraFrame")
        self.camera_frame.setMinimumHeight(400)
        
        cam_inner = QVBoxLayout(self.camera_frame)
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("background-color: #000;")
        self.camera_label.setText("NO SIGNAL")
        self.camera_label.setStyleSheet("color: #8b949e; font-size: 24px; background-color: #000;")
        
        # Overlay label
        self.overlay_label = QLabel("STANDBY")
        self.overlay_label.setStyleSheet("""
            color: #00ff00; 
            font-size: 14px; 
            font-weight: bold;
            background-color: rgba(0,0,0,150);
            padding: 4px 8px;
            border-radius: 4px;
        """)
        self.overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        cam_inner.addWidget(self.camera_label, 1)
        cam_inner.addWidget(self.overlay_label)
        
        camera_layout.addLayout(cam_select_layout)
        camera_layout.addWidget(self.camera_frame, 1)
        
        # === MAP VIEW ===
        map_group = QGroupBox("🗺️ NAV2 Costmap")
        map_layout = QVBoxLayout(map_group)
        
        self.map_label = QLabel("[NAV2 Costmap Stream]")
        self.map_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.map_label.setStyleSheet("background-color: #161b22; border-radius: 6px; padding: 40px;")
        self.map_label.setMinimumHeight(150)
        
        map_layout.addWidget(self.map_label)
        
        layout.addWidget(status_frame)
        layout.addWidget(camera_group, 1)
        layout.addWidget(map_group)
        
        return panel
        
    def create_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)
        
        # === EMERGENCY STOP ===
        self.estop_btn = QPushButton("⚠️ EMERGENCY STOP")
        self.estop_btn.setObjectName("emergency")
        self.estop_btn.clicked.connect(self.emergency_stop)
        
        # === TASK QUEUE ===
        queue_group = QGroupBox("📝 Task Queue")
        queue_layout = QVBoxLayout(queue_group)
        
        self.queue_list = QListWidget()
        self.queue_list.setMinimumHeight(200)
        
        queue_btn_layout = QHBoxLayout()
        self.delete_btn = QPushButton("🗑️ Delete Selected")
        self.delete_btn.setObjectName("deleteBtn")
        self.delete_btn.clicked.connect(self.delete_selected_task)
        queue_btn_layout.addWidget(self.delete_btn)
        queue_btn_layout.addStretch()
        
        queue_layout.addWidget(self.queue_list)
        queue_layout.addLayout(queue_btn_layout)
        
        # === SYSTEM LOG ===
        log_group = QGroupBox("📊 System Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(300)
        
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(self.estop_btn)
        layout.addWidget(queue_group)
        layout.addWidget(log_group, 1)
        
        return panel
        
    def connect_signals(self):
        """Connect ROS2 signals to UI slots"""
        self.ros_signals.log_signal.connect(self.add_log)
        self.ros_signals.state_signal.connect(self.update_robot_state)
        self.ros_signals.progress_signal.connect(self.update_progress)
        self.ros_signals.battery_signal.connect(self.update_battery)
        self.ros_signals.cam_overlay_signal.connect(self.update_cam_overlay)
        self.ros_signals.task_finished_signal.connect(self.on_task_finished)
        self.ros_signals.front_image_signal.connect(self.update_front_frame)
        self.ros_signals.left_image_signal.connect(self.update_left_frame)
        self.ros_signals.right_image_signal.connect(self.update_right_frame)
        
    def setup_timers(self):
        """Setup UI update timers"""
        # Camera update timer (30 FPS)
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_display)
        self.camera_timer.start(33)
        
    # =========================================================================
    # SLOT METHODS
    # =========================================================================
    
    def add_log(self, msg: str):
        """Add timestamped log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color coding based on message type
        if "[ERROR]" in msg:
            color = "#f85149"
        elif "[RESULT]" in msg and "🎉" in msg:
            color = "#3fb950"
        elif "[STATE]" in msg:
            color = "#58a6ff"
        elif "[ACTION]" in msg:
            color = "#d29922"
        elif "[DISPATCH]" in msg:
            color = "#a371f7"
        elif "[CANCEL]" in msg:
            color = "#f85149"
        else:
            color = "#8b949e"
            
        formatted = f'<span style="color:{color}">[{timestamp}] {msg}</span>'
        self.log_text.append(formatted)
        
        # Auto scroll
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def update_robot_state(self, state: str):
        """Update robot state display"""
        self.current_state = state
        self.state_label.setText(state)
        
        # Color based on state
        if state == "ERROR":
            self.state_label.setStyleSheet("color: #f85149; font-size: 18px; font-weight: bold;")
        elif state == "COMPLETED":
            self.state_label.setStyleSheet("color: #3fb950; font-size: 18px; font-weight: bold;")
        elif "NAVIGATING" in state:
            self.state_label.setStyleSheet("color: #58a6ff; font-size: 18px; font-weight: bold;")
        elif "DOCKING" in state:
            self.state_label.setStyleSheet("color: #d29922; font-size: 18px; font-weight: bold;")
        else:
            self.state_label.setStyleSheet("color: #58a6ff; font-size: 18px; font-weight: bold;")
            
        # Update camera selection if auto mode
        if self.cam_auto_btn.isChecked():
            cam = STATE_CAMERA.get(state, "front")
            if cam == "auto":
                self.current_camera = self.current_work_side.lower()
            else:
                self.current_camera = cam
                
    def update_progress(self, val: int):
        """Update progress bar"""
        self.progress_bar.setValue(val)
        
    def update_battery(self, val: int):
        """Update battery display"""
        self.battery_level = val
        
        if val <= 20:
            self.battery_label.setText(f"🪫 {val}% ⚠️")
            self.battery_label.setStyleSheet("color: #f85149; font-size: 14px;")
        elif val <= 50:
            self.battery_label.setText(f"🔋 {val}%")
            self.battery_label.setStyleSheet("color: #d29922; font-size: 14px;")
        else:
            self.battery_label.setText(f"🔋 {val}%")
            self.battery_label.setStyleSheet("color: #3fb950; font-size: 14px;")
            
    def update_cam_overlay(self, text: str):
        """Update camera overlay text"""
        self.overlay_label.setText(text)
        
        if "ERROR" in text or "CANCELED" in text:
            self.overlay_label.setStyleSheet("""
                color: #f85149; font-size: 14px; font-weight: bold;
                background-color: rgba(0,0,0,180); padding: 4px 8px; border-radius: 4px;
            """)
        elif "DONE" in text:
            self.overlay_label.setStyleSheet("""
                color: #3fb950; font-size: 14px; font-weight: bold;
                background-color: rgba(0,0,0,180); padding: 4px 8px; border-radius: 4px;
            """)
        else:
            self.overlay_label.setStyleSheet("""
                color: #00ff00; font-size: 14px; font-weight: bold;
                background-color: rgba(0,0,0,150); padding: 4px 8px; border-radius: 4px;
            """)
            
    def update_front_frame(self, frame):
        self.front_frame = frame
        
    def update_left_frame(self, frame):
        self.left_frame = frame
        
    def update_right_frame(self, frame):
        self.right_frame = frame
        
    def update_camera_display(self):
        """Update camera display based on current selection"""
        frame = None
        
        if self.current_camera == "front":
            frame = self.front_frame
        elif self.current_camera == "left":
            frame = self.left_frame
        elif self.current_camera == "right":
            frame = self.right_frame
            
        if frame is not None:
            # Resize frame to fit label
            h, w = frame.shape[:2]
            label_size = self.camera_label.size()
            
            # Maintain aspect ratio
            scale = min(label_size.width() / w, label_size.height() / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            resized = cv2.resize(frame, (new_w, new_h))
            
            # Add camera indicator overlay
            cam_text = f"CAM: {self.current_camera.upper()}"
            cv2.putText(resized, cam_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add timestamp
            ts = datetime.now().strftime("%H:%M:%S")
            cv2.putText(resized, ts, (new_w - 80, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Convert to QPixmap
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            self.camera_label.setPixmap(pixmap)
        else:
            self.camera_label.setText("NO SIGNAL")
            self.camera_label.setStyleSheet("color: #8b949e; font-size: 24px; background-color: #000;")
            
    def set_camera(self, cam: str):
        """Set camera selection"""
        self.cam_front_btn.setChecked(cam == "front")
        self.cam_left_btn.setChecked(cam == "left")
        self.cam_right_btn.setChecked(cam == "right")
        self.cam_auto_btn.setChecked(cam == "auto")
        
        if cam != "auto":
            self.current_camera = cam
            
    def update_item_info(self, item_text: str):
        """Update item information display"""
        # Extract item name
        for item_name, info in ITEM_TYPES.items():
            if item_name in item_text:
                priority_colors = {
                    "HIGH": "#f85149",
                    "CRITICAL": "#da3633",
                    "NORMAL": "#58a6ff",
                    "LOW": "#8b949e"
                }
                color = priority_colors.get(info['priority'], "#8b949e")
                
                self.item_info_label.setText(
                    f"Priority: <span style='color:{color}'>{info['priority']}</span> | "
                    f"Speed: {info['speed']} | Recommended: {info['dest_hint']}"
                )
                break
                
    def load_patient_data(self):
        """Load patient data into table"""
        self.patient_table.setRowCount(len(PATIENTS))
        
        condition_colors = {
            "Stable": "#3fb950",
            "Post-Op": "#d29922",
            "Critical": "#f85149",
            "Check-up": "#58a6ff"
        }
        
        for i, (pid, info) in enumerate(PATIENTS.items()):
            name_item = QTableWidgetItem(info['name'])
            ward_item = QTableWidgetItem(info['ward'])
            cond_item = QTableWidgetItem(info['condition'])
            
            color = condition_colors.get(info['condition'], "#c9d1d9")
            cond_item.setForeground(QColor(color))
            
            self.patient_table.setItem(i, 0, name_item)
            self.patient_table.setItem(i, 1, ward_item)
            self.patient_table.setItem(i, 2, cond_item)
            
    def auto_fill_destination(self, row, col):
        """Auto-fill destination when patient row is clicked"""
        ward_item = self.patient_table.item(row, 1)
        if ward_item:
            ward = ward_item.text()
            idx = self.dropoff_combo.findText(ward)
            if idx >= 0:
                self.dropoff_combo.setCurrentIndex(idx)
                self.add_log(f"[UI] Auto-selected destination: {ward}")
                
    # =========================================================================
    # TASK MANAGEMENT
    # =========================================================================
    
    def dispatch_task(self):
        """Dispatch a new work order"""
        item = self.item_combo.currentText()
        pickup = self.pickup_combo.currentText()
        dropoff = self.dropoff_combo.currentText()
        
        # Validation
        if pickup == dropoff:
            QMessageBox.warning(self, "Invalid Route", 
                              "Pickup and Dropoff locations cannot be the same!")
            return
            
        # Create task
        task = {
            "mode": "ALL",
            "item": item,
            "pickup": pickup,
            "dropoff": dropoff,
            "status": "WAITING"
        }
        
        self.task_queue.append(task)
        self.refresh_queue_list()
        
        self.add_log(f"[QUEUE] Work Order Created: {item.split()[1] if len(item.split()) > 1 else item}")
        
        # If no task running, start this one
        if not self.ros_worker.is_executing:
            self.execute_next_task()
            
    def execute_step(self, mode: str):
        """Execute a specific step"""
        item = self.item_combo.currentText()
        pickup = self.pickup_combo.currentText()
        dropoff = self.dropoff_combo.currentText()
        
        # Extract clean item name
        clean_item = item.split(' ', 1)[1] if ' ' in item else item
        
        # Determine work side for camera
        self.current_work_side = "Left"  # Default, actual logic would check room_db
        
        self.add_log(f"[STEP] Starting from: {mode}")
        
        self.ros_worker.send_goal(mode, clean_item, pickup, dropoff)
        self.set_buttons_enabled(False)
        
    def execute_next_task(self):
        """Execute the next task in queue"""
        if not self.task_queue:
            return
            
        # Find first waiting task
        for i, task in enumerate(self.task_queue):
            if task['status'] == "WAITING":
                task['status'] = "RUNNING"
                self.current_task_index = i
                self.refresh_queue_list()
                
                clean_item = task['item'].split(' ', 1)[1] if ' ' in task['item'] else task['item']
                
                self.ros_worker.send_goal(
                    task['mode'],
                    clean_item,
                    task['pickup'],
                    task['dropoff']
                )
                self.set_buttons_enabled(False)
                return
                
    def on_task_finished(self, success: bool, message: str):
        """Handle task completion"""
        self.set_buttons_enabled(True)
        
        # Update queue
        if self.current_task_index >= 0 and self.current_task_index < len(self.task_queue):
            if success:
                self.task_queue[self.current_task_index]['status'] = "COMPLETED"
            else:
                self.task_queue[self.current_task_index]['status'] = "FAILED"
                
        self.current_task_index = -1
        self.refresh_queue_list()
        
        # Execute next task
        if success:
            self.execute_next_task()
            
    def refresh_queue_list(self):
        """Refresh the queue list display"""
        self.queue_list.clear()
        
        for i, task in enumerate(self.task_queue):
            status_icons = {
                "WAITING": "⏳",
                "RUNNING": "▶",
                "COMPLETED": "✅",
                "FAILED": "❌"
            }
            icon = status_icons.get(task['status'], "❓")
            
            item_name = task['item'].split(' ', 1)[1] if ' ' in task['item'] else task['item']
            text = f"{icon} [{task['status']}] {item_name}: {task['pickup']} → {task['dropoff']}"
            
            list_item = QListWidgetItem(text)
            
            if task['status'] == "RUNNING":
                list_item.setForeground(QColor("#58a6ff"))
            elif task['status'] == "COMPLETED":
                list_item.setForeground(QColor("#3fb950"))
            elif task['status'] == "FAILED":
                list_item.setForeground(QColor("#f85149"))
                
            self.queue_list.addItem(list_item)
            
    def delete_selected_task(self):
        """Delete selected task from queue"""
        row = self.queue_list.currentRow()
        if row < 0:
            return
            
        task = self.task_queue[row]
        
        if task['status'] == "RUNNING":
            reply = QMessageBox.question(
                self, "Cancel Running Task",
                "This task is currently running. Cancel it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.ros_worker.cancel_current_goal()
                del self.task_queue[row]
                self.refresh_queue_list()
                self.add_log(f"[QUEUE] Task #{row+1} canceled and removed")
        else:
            del self.task_queue[row]
            self.refresh_queue_list()
            self.add_log(f"[QUEUE] Task #{row+1} removed")
            
    def emergency_stop(self):
        """Emergency stop - cancel all tasks"""
        reply = QMessageBox.warning(
            self, "⚠️ EMERGENCY STOP",
            "This will IMMEDIATELY stop the robot and clear ALL pending tasks.\n\nProceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Cancel current goal
            self.ros_worker.cancel_current_goal()
            
            # Clear queue
            self.task_queue.clear()
            self.current_task_index = -1
            self.refresh_queue_list()
            
            # Update UI
            self.state_label.setText("ESTOP ACTIVATED")
            self.state_label.setStyleSheet("color: #f85149; font-size: 18px; font-weight: bold;")
            self.progress_bar.setValue(0)
            self.overlay_label.setText("EMERGENCY STOP")
            
            self.add_log("[ESTOP] ⚠️ EMERGENCY STOP ACTIVATED")
            self.add_log("[ESTOP] All tasks canceled and queue cleared")
            
            self.set_buttons_enabled(True)
            
    def set_buttons_enabled(self, enabled: bool):
        """Enable/disable action buttons"""
        self.dispatch_btn.setEnabled(enabled)
        for btn in self.step_btns:
            btn.setEnabled(enabled)
            

# =============================================================================
# MAIN
# =============================================================================

def main():
    # Initialize ROS2
    rclpy.init()
    
    # Create Qt Application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Create signals bridge
    ros_signals = ROS2Signals()
    
    # Create ROS2 worker node
    ros_worker = RobotWorker(ros_signals)
    
    # Create and start ROS2 thread
    ros_thread = ROS2Thread(ros_worker)
    ros_thread.start()
    
    # Create main window
    window = HospitalRobotUI(ros_signals, ros_worker)
    window.show()
    
    # Run application
    exit_code = app.exec()
    
    # Cleanup
    ros_thread.stop()
    ros_thread.wait()
    ros_worker.destroy_node()
    rclpy.shutdown()
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()