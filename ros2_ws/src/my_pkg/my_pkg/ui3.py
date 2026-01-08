#!/usr/bin/env python3
import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# [PyQt6 Imports] - QSizePolicy 추가됨
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QGridLayout, QLabel, QPushButton, QComboBox, QListWidget, 
                             QProgressBar, QGroupBox, QTextEdit, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QMessageBox, QFrame, QSplitter, QSizePolicy)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap, QColor, QFont, QIcon, QAction

# [ROS Interfaces]
from moma_interfaces.action import RunDelivery
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np

# ------------------------------------------------------------------
# [설정] 데이터 베이스 (요구사항 반영)
# ------------------------------------------------------------------
LOCATIONS = [
    "Nurse Station A (Base)", "Ward 101", "Ward 102", "Ward 103", "Ward 104", 
    "Ward 105", "Main Pharmacy (Central)", "Sub Pharmacy", 
    "Clinical Lab (Zone C)", "Central Supply", "Doctor's Office"
]

ITEM_TYPES = {
    "Blood Sample (Emergency)": {"icon": "🩸", "priority": "High", "speed": "Slow", "default_dest": "Clinical Lab (Zone C)"},
    "General Medicine": {"icon": "💊", "priority": "Normal", "speed": "Normal", "default_dest": "Ward 102"},
    "Narcotics (Secure)": {"icon": "🔒", "priority": "Critical", "speed": "Fast", "default_dest": "Doctor's Office"},
    "Surgical Kit": {"icon": "✂️", "priority": "Normal", "speed": "Normal", "default_dest": "Central Supply"},
    "Documents/Chart": {"icon": "📄", "priority": "Low", "speed": "Max", "default_dest": "Nurse Station A (Base)"}
}

PATIENTS = [
    {"id": "PT-2401", "name": "김철수", "ward": "Ward 101", "condition": "Stable"},
    {"id": "PT-2402", "name": "이영희", "ward": "Ward 102", "condition": "Post-Op"},
    {"id": "PT-2403", "name": "박지성", "ward": "Ward 105", "condition": "Critical"},
    {"id": "PT-2404", "name": "최민아", "ward": "Ward 102", "condition": "Check-up"}
]

# ------------------------------------------------------------------
# [Thread 1] ROS2 Worker (통신 전담)
# ------------------------------------------------------------------
class RosWorker(QThread):
    # UI 업데이트 시그널
    log_signal = pyqtSignal(str)          # 로그 텍스트
    state_signal = pyqtSignal(str)        # 현재 상태 (라벨용)
    progress_signal = pyqtSignal(int)     # 진행률 (%)
    image_signal = pyqtSignal(QImage)     # 카메라 영상
    task_finished_signal = pyqtSignal(bool, str) # 작업 완료 여부

    def __init__(self):
        super().__init__()
        self.node = None
        self.executor = None
        self.action_client = None
        self.bridge = CvBridge()
        
        # 카메라 토픽 데이터 저장소
        self.latest_front_img = None
        self.latest_left_img = None
        self.latest_right_img = None
        
        # 현재 활성화된 카메라 모드 (Front/Left/Right)
        self.active_camera_mode = "Front" 
        
        # 진행률 계산용 변수
        self.current_step_idx = 0
        self.total_steps = 7

    def run(self):
        # 1. 노드 초기화
        rclpy.init()
        self.node = Node('hospital_gui_node')
        
        # 2. Action Client 설정
        self.action_client = ActionClient(self.node, RunDelivery, 'run_delivery')
        
        # 3. 카메라 구독 (QoS: Sensor Data 최적화)
        qos_policy = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.node.create_subscription(Image, '/front_camera/rgb', self.cb_front_cam, qos_policy)
        self.node.create_subscription(Image, '/left_camera/rgb', self.cb_left_cam, qos_policy)
        self.node.create_subscription(Image, '/right_camera/rgb', self.cb_right_cam, qos_policy)
        
        # 4. 카메라 타이머 (UI로 이미지 전송 - 30fps 제한)
        self.img_timer = self.node.create_timer(0.033, self.publish_image_to_ui)

        self.log_signal.emit("[SYSTEM] ROS2 Node Started. Waiting for Action Server...")
        
        # 5. Spin Loop
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        try:
            self.executor.spin()
        except Exception as e:
            self.log_signal.emit(f"[ERROR] ROS Spin Failed: {e}")
        finally:
            self.node.destroy_node()
            rclpy.shutdown()

    # --- 카메라 콜백 ---
    def cb_front_cam(self, msg): self.latest_front_img = msg
    def cb_left_cam(self, msg): self.latest_left_img = msg
    def cb_right_cam(self, msg): self.latest_right_img = msg

    def publish_image_to_ui(self):
        """현재 모드에 맞는 이미지를 선택하여 UI로 전송"""
        target_msg = None
        overlay_text = f"CAM: {self.active_camera_mode}"
        
        if self.active_camera_mode == "Left":
            target_msg = self.latest_left_img
        elif self.active_camera_mode == "Right":
            target_msg = self.latest_right_img
        else:
            target_msg = self.latest_front_img # Default

        if target_msg:
            try:
                cv_img = self.bridge.imgmsg_to_cv2(target_msg, desired_encoding="bgr8")
                
                # [오버레이] 상태 정보 표시
                h, w, _ = cv_img.shape
                cv2.rectangle(cv_img, (10, 10), (250, 40), (0, 0, 0), -1)
                cv2.putText(cv_img, overlay_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # [변환] OpenCV -> QImage
                bytes_per_line = 3 * w
                qt_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
                self.image_signal.emit(qt_img)
            except Exception as e:
                pass # 변환 에러 무시 (로그 스팸 방지)

    # --- 액션 요청 함수 ---
    def send_goal(self, task_mode, item_type, pickup, dropoff):
        if not self.action_client.wait_for_server(timeout_sec=3.0):
            self.log_signal.emit("[ERROR] Action Server not available!")
            self.task_finished_signal.emit(False, "Server Timeout")
            return

        goal_msg = RunDelivery.Goal()
        goal_msg.task_mode = task_mode
        goal_msg.item_type = item_type
        goal_msg.pickup_loc = pickup
        goal_msg.dropoff_loc = dropoff
        
        self.log_signal.emit(f"[SEND] Goal: {task_mode} | Item: {item_type}")
        
        # 진행률 초기화
        self.current_step_idx = 0
        self.progress_signal.emit(0)

        self._send_goal_future = self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.log_signal.emit("[ERROR] Goal Rejected by Server.")
            self.task_finished_signal.emit(False, "Goal Rejected")
            return

        self.log_signal.emit("[INFO] Goal Accepted. Executing...")
        self._goal_handle = goal_handle # 취소를 위해 저장
        
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """Action Feedback 수신 시 상태 업데이트 및 카메라 전환"""
        state = feedback_msg.feedback.current_state
        self.state_signal.emit(state)
        self.log_signal.emit(f"[FEEDBACK] {state}")
        
        # [핵심] 상태 기반 카메라 자동 전환 로직
        state_upper = state.upper()
        if "LEFT" in state_upper and ("PICK" in state_upper or "SCAN" in state_upper):
            self.active_camera_mode = "Left"
        elif "RIGHT" in state_upper and ("PICK" in state_upper or "SCAN" in state_upper):
            self.active_camera_mode = "Right"
        elif "DOCK" in state_upper:
            self.active_camera_mode = "Front" # 도킹 시에는 정밀 유도 오버레이가 있는 Front 사용
        else:
            self.active_camera_mode = "Front" # 이동 중 기본값

        # [핵심] 진행률 추정 (단순화)
        self.current_step_idx += 1
        pct = min(100, int((self.current_step_idx / 25) * 100)) # 대략적인 진행률
        self.progress_signal.emit(pct)

    def get_result_callback(self, future):
        result = future.result().result
        status = future.result().status
        
        # 상태 리셋
        self.active_camera_mode = "Front"
        
        if result.success:
            self.log_signal.emit(f"[DONE] {result.message}")
            self.task_finished_signal.emit(True, result.message)
            self.progress_signal.emit(100)
        else:
            self.log_signal.emit(f"[FAIL] {result.message}")
            self.task_finished_signal.emit(False, result.message)

    def cancel_goal(self):
        """현재 실행 중인 Goal 취소"""
        if hasattr(self, '_goal_handle') and self._goal_handle:
            self.log_signal.emit("[CANCEL] Sending Cancel Request...")
            self._goal_handle.cancel_goal_async()

# ------------------------------------------------------------------
# [GUI] Main Window
# ------------------------------------------------------------------
class HospitalGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medical Robot Control System (ROS2 Humble)")
        self.resize(1400, 900)
        
        # 스타일시트 (전문적인 Dark Medical Theme)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; color: #cdd6f4; }
            QGroupBox { 
                border: 1px solid #45475a; border-radius: 6px; margin-top: 10px; 
                font-weight: bold; color: #89b4fa; background-color: #181825;
            }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
            QLabel { color: #cdd6f4; font-size: 13px; }
            QPushButton { 
                background-color: #313244; border: 1px solid #45475a; border-radius: 4px; 
                color: #cdd6f4; padding: 6px; min-height: 25px;
            }
            QPushButton:hover { background-color: #45475a; border-color: #89b4fa; }
            QPushButton:pressed { background-color: #585b70; }
            QComboBox, QListWidget, QTextEdit, QTableWidget {
                background-color: #11111b; border: 1px solid #45475a; border-radius: 4px; color: #a6adc8;
            }
            QProgressBar {
                border: 1px solid #45475a; border-radius: 4px; text-align: center; background-color: #11111b;
            }
            QProgressBar::chunk { background-color: #89b4fa; }
            QHeaderView::section { background-color: #313244; color: #cdd6f4; border: none; padding: 4px; }
        """)

        # ROS Worker 스레드 시작
        self.ros_thread = RosWorker()
        self.ros_thread.log_signal.connect(self.append_log)
        self.ros_thread.state_signal.connect(self.update_state_label)
        self.ros_thread.progress_signal.connect(self.update_progress)
        self.ros_thread.image_signal.connect(self.update_camera_view)
        self.ros_thread.task_finished_signal.connect(self.on_task_finished)
        self.ros_thread.start()

        # 작업 큐
        self.task_queue = []
        self.is_running = False

        self.init_ui()
        
        # 배터리 시뮬레이션 타이머
        self.battery_level = 100
        self.timer_bat = QTimer()
        self.timer_bat.timeout.connect(self.update_battery)
        self.timer_bat.start(5000) # 5초마다 감소

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ==================== [Left Panel] Control & Data ====================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(500) # 고정 너비

        # 1. 작업 디스패치 (Task Dispatch)
        gb_dispatch = QGroupBox("🚑 Task Dispatch")
        grid_disp = QGridLayout()
        
        # 아이템 선택
        grid_disp.addWidget(QLabel("Item Type:"), 0, 0)
        self.cb_item = QComboBox()
        self.cb_item.addItems(ITEM_TYPES.keys())
        self.cb_item.currentTextChanged.connect(self.on_item_changed)
        grid_disp.addWidget(self.cb_item, 0, 1)

        # 정보 라벨
        self.lbl_item_info = QLabel("Details: 🩸 High Priority | 🐢 Slow Speed")
        self.lbl_item_info.setStyleSheet("color: #f38ba8; font-style: italic;")
        grid_disp.addWidget(self.lbl_item_info, 1, 0, 1, 2)

        # 출발지/도착지
        grid_disp.addWidget(QLabel("Pickup Loc:"), 2, 0)
        self.cb_pickup = QComboBox()
        self.cb_pickup.addItems(LOCATIONS)
        grid_disp.addWidget(self.cb_pickup, 2, 1)

        grid_disp.addWidget(QLabel("Dropoff Loc:"), 3, 0)
        self.cb_dropoff = QComboBox()
        self.cb_dropoff.addItems(LOCATIONS)
        self.cb_dropoff.setCurrentText("Clinical Lab (Zone C)") # Default
        grid_disp.addWidget(self.cb_dropoff, 3, 1)

        # 디스패치 버튼
        self.btn_dispatch = QPushButton("📩 Dispatch Work Order (Queue)")
        self.btn_dispatch.setStyleSheet("background-color: #313244; color: #89b4fa; font-weight: bold; height: 40px;")
        self.btn_dispatch.clicked.connect(self.add_to_queue)
        grid_disp.addWidget(self.btn_dispatch, 4, 0, 1, 2)
        
        gb_dispatch.setLayout(grid_disp)
        left_layout.addWidget(gb_dispatch)

        # 2. 단계별 수동 실행 (Manual Step Control)
        gb_manual = QGroupBox("🛠️ Manual Step Execution")
        manual_layout = QGridLayout()
        
        steps = [
            ("NAV_PICKUP", "🚗 Nav to Pickup"), ("DOCK_PICKUP", "⚓ Dock (Pick)"), ("PICK", "🦾 Pick Item"),
            ("NAV_DROPOFF", "🚗 Nav to Drop"), ("DOCK_DROPOFF", "⚓ Dock (Drop)"), ("PLACE", "🤲 Place Item")
        ]
        
        self.step_buttons = []
        for i, (mode, text) in enumerate(steps):
            btn = QPushButton(text)
            btn.clicked.connect(lambda checked, m=mode: self.execute_manual_step(m))
            manual_layout.addWidget(btn, i//2, i%2)
            self.step_buttons.append(btn)
            
        gb_manual.setLayout(manual_layout)
        left_layout.addWidget(gb_manual)

        # 3. 작업 큐 리스트
        gb_queue = QGroupBox("📋 Task Queue")
        vbox_q = QVBoxLayout()
        self.list_queue = QListWidget()
        vbox_q.addWidget(self.list_queue)
        
        btn_del_q = QPushButton("🗑️ Remove Selected")
        btn_del_q.clicked.connect(self.remove_selected_task)
        vbox_q.addWidget(btn_del_q)
        
        gb_queue.setLayout(vbox_q)
        left_layout.addWidget(gb_queue)

        # 4. 환자 정보 (클릭 시 자동 입력)
        gb_patient = QGroupBox("🏥 Patient Info (Click to Auto-fill)")
        vbox_pt = QVBoxLayout()
        self.table_pt = QTableWidget()
        self.table_pt.setColumnCount(3)
        self.table_pt.setHorizontalHeaderLabels(["Name", "Ward", "Condition"])
        self.table_pt.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_pt.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_pt.verticalHeader().setVisible(False)
        self.table_pt.cellClicked.connect(self.on_patient_clicked)
        
        # 데이터 채우기
        self.table_pt.setRowCount(len(PATIENTS))
        for i, pt in enumerate(PATIENTS):
            self.table_pt.setItem(i, 0, QTableWidgetItem(pt['name']))
            self.table_pt.setItem(i, 1, QTableWidgetItem(pt['ward']))
            self.table_pt.setItem(i, 2, QTableWidgetItem(pt['condition']))
            
        vbox_pt.addWidget(self.table_pt)
        gb_patient.setLayout(vbox_pt)
        left_layout.addWidget(gb_patient)

        # ==================== [Right Panel] Monitor & View ====================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 1. 상태 헤더 (Status & Battery & ESTOP)
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #11111b; border-radius: 8px;")
        hbox_header = QHBoxLayout(header_frame)

        self.lbl_status = QLabel("STATUS: IDLE")
        self.lbl_status.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.lbl_status.setStyleSheet("color: #a6adc8;")
        hbox_header.addWidget(self.lbl_status)

        hbox_header.addStretch()

        self.lbl_bat = QLabel("🔋 100%")
        self.prog_bat = QProgressBar()
        self.prog_bat.setFixedWidth(150)
        self.prog_bat.setValue(100)
        self.prog_bat.setStyleSheet("QProgressBar::chunk { background-color: #a6e3a1; }")
        hbox_header.addWidget(self.lbl_bat)
        hbox_header.addWidget(self.prog_bat)

        self.btn_estop = QPushButton("🚨 ESTOP")
        self.btn_estop.setFixedSize(100, 40)
        self.btn_estop.setStyleSheet("""
            background-color: #f38ba8; color: #11111b; font-weight: bold; border: 2px solid #ff0000; font-size: 14px;
        """)
        self.btn_estop.clicked.connect(self.trigger_estop)
        hbox_header.addWidget(self.btn_estop)

        right_layout.addWidget(header_frame)

        # 2. 카메라 뷰어
        self.lbl_camera = QLabel("Waiting for Camera Stream...")
        self.lbl_camera.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_camera.setStyleSheet("background-color: #000; border: 2px solid #45475a; border-radius: 6px;")
        self.lbl_camera.setMinimumHeight(400)
        
        # [수정된 부분] QSizePolicy 사용
        self.lbl_camera.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout.addWidget(self.lbl_camera)

        # 3. 로그 (Console Style)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet("background-color: #000; color: #a6e3a1; font-family: Consolas; font-size: 12px;")
        self.txt_log.setFixedHeight(150)
        right_layout.addWidget(self.txt_log)
        
        # 4. 전체 진행률
        self.prog_task = QProgressBar()
        self.prog_task.setValue(0)
        self.prog_task.setFixedHeight(25)
        right_layout.addWidget(self.prog_task)

        # 패널 배치
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

    # ------------------------------------------------------------------
    # [Logic] UI Event Handlers
    # ------------------------------------------------------------------
    def on_item_changed(self, text):
        info = ITEM_TYPES.get(text, {})
        self.lbl_item_info.setText(f"Details: {info.get('icon','')} Priority: {info.get('priority')} | Speed: {info.get('speed')}")
        
        # 추천 목적지 자동 설정
        dest = info.get('default_dest')
        if dest:
            idx = self.cb_dropoff.findText(dest)
            if idx >= 0: self.cb_dropoff.setCurrentIndex(idx)

    def on_patient_clicked(self, row, col):
        ward = self.table_pt.item(row, 1).text()
        name = self.table_pt.item(row, 0).text()
        
        # 도착지를 해당 병실로 설정
        idx = self.cb_dropoff.findText(ward)
        if idx >= 0:
            self.cb_dropoff.setCurrentIndex(idx)
            self.append_log(f"[UI] Destination set to {ward} for Patient {name}")

    def add_to_queue(self):
        item = self.cb_item.currentText()
        pickup = self.cb_pickup.currentText()
        dropoff = self.cb_dropoff.currentText()
        
        task_info = f"[{item}] {pickup} -> {dropoff}"
        
        # 큐 데이터 저장
        task_data = {
            "mode": "ALL",
            "item": item,
            "pickup": pickup,
            "dropoff": dropoff,
            "display": task_info
        }
        
        self.task_queue.append(task_data)
        self.list_queue.addItem(f"⏳ {task_info}")
        self.append_log(f"[QUEUE] Added: {task_info}")
        
        self.process_queue()

    def remove_selected_task(self):
        row = self.list_queue.currentRow()
        if row >= 0:
            # 실행 중인 작업(0번 인덱스)은 취소 불가 (ESTOP 써야 함)
            if row == 0 and self.is_running:
                QMessageBox.warning(self, "Warning", "Cannot remove currently running task! Use ESTOP.")
                return
            
            self.list_queue.takeItem(row)
            del self.task_queue[row]
            self.append_log(f"[QUEUE] Removed item at index {row}")

    def process_queue(self):
        """큐를 확인하고 대기 중인 작업 실행"""
        if self.is_running or not self.task_queue:
            return
        
        # 큐의 첫 번째 작업 가져오기
        current_task = self.task_queue[0]
        self.is_running = True
        
        # UI 업데이트 (실행 중 표시)
        item = self.list_queue.item(0)
        if item: item.setText(f"▶ {current_task['display']}")
        
        # ROS 액션 요청
        self.ros_thread.send_goal(
            current_task['mode'],
            current_task['item'],
            current_task['pickup'],
            current_task['dropoff']
        )
        
        # 버튼 비활성화
        self.toggle_inputs(False)

    def execute_manual_step(self, mode):
        """특정 단계만 수동 실행"""
        if self.is_running:
            QMessageBox.warning(self, "Busy", "Robot is currently running a task.")
            return

        item = self.cb_item.currentText()
        pickup = self.cb_pickup.currentText()
        dropoff = self.cb_dropoff.currentText()
        
        self.is_running = True
        self.ros_thread.send_goal(mode, item, pickup, dropoff)
        self.toggle_inputs(False)
        self.append_log(f"[MANUAL] Executing Step: {mode}")

    def trigger_estop(self):
        """긴급 정지"""
        self.append_log("[ESTOP] 🚨 EMERGENCY STOP TRIGGERED!")
        
        # 1. ROS Cancel 요청
        self.ros_thread.cancel_goal()
        
        # 2. UI 강제 초기화
        self.task_queue.clear()
        self.list_queue.clear()
        self.is_running = False
        self.toggle_inputs(True)
        
        # 3. 시각적 알림
        self.lbl_status.setText("STATUS: ESTOP ACTIVATED")
        self.lbl_status.setStyleSheet("color: #ff0000; font-weight: bold; font-size: 20px;")
        QMessageBox.critical(self, "EMERGENCY STOP", "System Halted. Queue Cleared.")

    def on_task_finished(self, success, msg):
        """작업 완료 시그널 처리"""
        self.is_running = False
        self.toggle_inputs(True)
        
        if success:
            self.lbl_status.setText("STATUS: IDLE")
            self.lbl_status.setStyleSheet("color: #a6adc8;")
            # 큐에서 완료된 작업 제거
            if self.task_queue:
                self.task_queue.pop(0)
                self.list_queue.takeItem(0)
            
            # 다음 작업 자동 실행
            self.process_queue()
        else:
            self.lbl_status.setText("STATUS: ERROR/CANCELED")
            self.lbl_status.setStyleSheet("color: #f38ba8;")

    # ------------------------------------------------------------------
    # [Utility] UI Updates
    # ------------------------------------------------------------------
    def append_log(self, text):
        timestamp = time.strftime("%H:%M:%S")
        self.txt_log.append(f"[{timestamp}] {text}")
        self.txt_log.verticalScrollBar().setValue(self.txt_log.verticalScrollBar().maximum())

    def update_state_label(self, state):
        self.lbl_status.setText(f"STATUS: {state}")
        
        # 상태별 컬러 코딩
        style = "font-weight: bold; font-size: 16px; "
        if "NAV" in state: style += "color: #89b4fa;" # Blue
        elif "DOCK" in state: style += "color: #a6e3a1;" # Green
        elif "PICK" in state or "PLACE" in state: style += "color: #fab387;" # Orange
        else: style += "color: #cdd6f4;" # White
        self.lbl_status.setStyleSheet(style)

    def update_progress(self, value):
        self.prog_task.setValue(value)

    def update_camera_view(self, q_image):
        pixmap = QPixmap.fromImage(q_image)
        # 비율 유지하며 스케일링
        scaled_pixmap = pixmap.scaled(self.lbl_camera.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.lbl_camera.setPixmap(scaled_pixmap)

    def update_battery(self):
        if self.is_running:
            self.battery_level = max(0, self.battery_level - 1)
        
        self.lbl_bat.setText(f"🔋 {self.battery_level}%")
        self.prog_bat.setValue(self.battery_level)
        
        if self.battery_level < 20:
            self.prog_bat.setStyleSheet("QProgressBar::chunk { background-color: #f38ba8; }") # Red

    def toggle_inputs(self, enable):
        """작업 중 입력 막기"""
        self.btn_dispatch.setEnabled(enable)
        for btn in self.step_buttons:
            btn.setEnabled(enable)

def main():
    app = QApplication(sys.argv)
    window = HospitalGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()