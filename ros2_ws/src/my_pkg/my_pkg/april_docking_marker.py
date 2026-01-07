#!/usr/bin/env python3
"""
Simple Precision Docking Controller (Optimized)
- 정밀 회전 시 부드러운 감속 로직 추가
- 도킹 완료 후 자동 종료 기능 추가
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool  # 비전 트리거 신호용

from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
# 패키지명은 실제 패키지 이름으로 변경 (예: my_pkg)
from moma_interfaces.action import Dock 
import time

import numpy as np
import math
from enum import Enum

def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z

class DockingState(Enum):
    IDLE = 0
    ROTATE_TO_TARGET = 1    
    APPROACH = 2            
    FINAL_ALIGN = 3         
    ALIGN_TO_MARKER = 4
    VERIFY_ALIGNMENT = 5  
    DOCKED = 6        

class SimplePrecisionDocking(Node):
    def __init__(self):
        super().__init__('simple_precision_docking')
        
        # Parameters
        self.declare_parameter('docking_distance_threshold', 2.0)
        self.declare_parameter('rotation_threshold', 0.087)
        self.declare_parameter('approach_speed', 0.4)
        self.declare_parameter('rotation_speed', 0.5)
        self.declare_parameter('final_speed', 0.15)
        self.declare_parameter('auto_start', False)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        
        self.docking_threshold = self.get_parameter('docking_distance_threshold').value
        self.rotation_threshold = self.get_parameter('rotation_threshold').value
        self.approach_speed = self.get_parameter('approach_speed').value
        self.rotation_speed = self.get_parameter('rotation_speed').value
        self.final_speed = self.get_parameter('final_speed').value
        self.auto_start = self.get_parameter('auto_start').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        
        # State variables
        self.state = DockingState.IDLE
        self.latest_dock_pose = None
        self.latest_pose_time = None
        self.docking_enabled = self.auto_start
        
        # [추가] Yaw 필터링을 위한 변수 (EMA 필터)
        self.filtered_yaw = None
        self.alpha = 0.6  # 0.0~1.0 사이. 클수록 최신값 반영 비율 높음 (반응성 좋음)
        
        self.current_yaw_raw = 0.0
        
        # [추가] 정렬 중 마커 놓침 방지용 카운터
        self.marker_lost_count = 0
        
        # 재정렬 카운터 (최대 2번 재시도)
        self.realignment_count = 0
        self.verification_start_time = None
        
        # Subscribers/Publishers
        self.create_subscription(PoseStamped, 'detected_dock_pose', self.dock_pose_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.trigger_pub = self.create_publisher(Bool, '/docking/trigger', 10)
        
        # Action
        self.callback_group = ReentrantCallbackGroup()

        self._action_server = ActionServer(
            self,
            Dock,
            'dock_robot',
            self.execute_callback,
            callback_group=self.callback_group,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # 현재 액션 핸들을 저장할 변수 (Feedback/Result 전송용)
        self.goal_handle = None
        
        self.create_timer(0.05, self.control_loop)
        
        # [추가] 초기 설정된 auto_start 값에 맞춰 퍼블리셔 상태 동기화
        trigger_msg = Bool()
        trigger_msg.data = self.docking_enabled
        self.trigger_pub.publish(trigger_msg)
        
        self.get_logger().info('🎯 Docking node ready')

    def goal_callback(self, goal_request):
        # 이미 도킹 중이면 거절하거나, 선점 로직 구현 가능
        self.get_logger().info('🔔 Action Goal Received')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('⚠️ Action Cancel Received')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info('🚀 Executing Docking Action...')
        self.goal_handle = goal_handle
        
        # 1. 도킹 시작 설정
        self.docking_enabled = True
        
        # 마커 인식 시작 신호 전송
        self.trigger_pub.publish(Bool(data=True))
        
        self.state = DockingState.IDLE
        self.realignment_count = 0
        self.verification_start_time = None
        self.filtered_yaw = None
        
        
        # 2. 완료 대기 루프 (Timer가 로직을 수행하는 동안 여기서 대기)
        # Feedback은 Timer Loop에서 publish 하거나 여기서 polling 할 수 있음
        while self.docking_enabled and rclpy.ok():
            # 취소 요청 확인
            if goal_handle.is_cancel_requested:
                self.stop_robot()
                self.docking_enabled = False
                goal_handle.canceled()
                self.get_logger().info('🛑 Action Canceled')
                return Dock.Result(success=False, message="Canceled by user")
            
            # Action 처리를 위해 약간의 sleep 필요 (Busy waiting 방지)
            time.sleep(0.1)

        # 3. 루프 탈출 후 결과 반환 (성공/실패 여부는 _finish_docking에서 state로 판단 가능)
        # Timer 로직에서 docking_enabled를 False로 만들면 여기로 내려옴
        
        result = Dock.Result()
        if self.state == DockingState.DOCKED:
            result.success = True
            result.message = "Docking Completed Successfully"
            goal_handle.succeed()
        else:
            result.success = False
            result.message = "Docking Failed or Aborted"
            goal_handle.abort() # 또는 succeed(False) 처리
            
        self.goal_handle = None
        return result
        
    def dock_pose_callback(self, msg):
        self.latest_dock_pose = msg
        self.latest_pose_time = self.get_clock().now()
        
        # IDLE 상태에서만 자동 시작 체크
        if self.docking_enabled and self.state == DockingState.IDLE:
            distance = msg.pose.position.z
            if distance > 0.5:
                self.state = DockingState.ROTATE_TO_TARGET
                self.get_logger().info(f'🚀 Auto-start Triggered! (Dist={distance:.2f}m)')
                
    def _finish_docking(self, success):
        """
        도킹 프로세스 종료 처리 헬퍼 메서드
        - success: True(성공), False(실패/포기)
        """
        # 1. 로봇 즉시 정지
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

        # 3. 도킹 활성화 플래그 끄기
        self.docking_enabled = False
        
        # 마커 인식 중지 신호 전송
        self.trigger_pub.publish(Bool(data=False))
        
        # 4. 상태 전환
        # 성공이든 실패든 프로세스가 끝났으므로 DOCKED 상태로 전환하여 IDLE 자동 시작 방지
        # (실패 시 IDLE로 보내면 마커 인식되자마자 다시 시작될 위험 있음)
        self.state = DockingState.DOCKED
        
        if success:
            self.get_logger().info("🏁 Docking Sequence Completed Successfully.")
        else:
            self.get_logger().warn("🛑 Docking Sequence Ended (Failed or Cancelled).")
    
    def control_loop(self):
        if not self.docking_enabled:
            return
        
        if self.docking_enabled and self.goal_handle is not None:
            # Feedback 메시지 생성 및 전송
            feedback_msg = Dock.Feedback()
            feedback_msg.state = self.state.name
            
            if self.latest_dock_pose:
                feedback_msg.distance_to_target = self.latest_dock_pose.pose.position.z
                # 필터링된 값이 있으면 그거 쓰고, 없으면 방금 계산한 Raw 값 사용 (모니터링 목적)
                feedback_msg.yaw_error = self.filtered_yaw if self.filtered_yaw is not None else self.current_yaw_raw
            
            self.goal_handle.publish_feedback(feedback_msg)
        
        # 데이터 신선도 체크: 0.2초 이상 된 데이터는 '과거 정보'로 간주
        if self.latest_pose_time is not None:
            pose_age = (self.get_clock().now() - self.latest_pose_time).nanoseconds / 1e9
            # 데이터가 오래된 경우 로봇을 멈추되, 상태(State)는 유지하여 다음 데이터를 기다림
            if pose_age > 0.2 and self.state not in [DockingState.IDLE, DockingState.ALIGN_TO_MARKER, DockingState.VERIFY_ALIGNMENT, DockingState.DOCKED]:
                self.get_logger().warn(f"⌛ Stale data ({pose_age:.2f}s)! Holding position...", throttle_duration_sec=1.0)
                hold_cmd = Twist()
                hold_cmd.linear.x = 0.0
                hold_cmd.angular.z = 0.0
                self.cmd_vel_pub.publish(hold_cmd)
                return

        if self.state == DockingState.IDLE:
            self.get_logger().info("💤 IDLE: Waiting for marker...", throttle_duration_sec=2.0)
            return

        # Marker 기반 데이터 계산
        if self.latest_dock_pose is None: return
        
        # Marker Loss 체크 (1초)
        if (self.get_clock().now() - self.latest_pose_time).nanoseconds / 1e9 > 1.0:
            self.get_logger().warn('⚠️ Marker lost - STOPPING!')
            hold_cmd = Twist()
            hold_cmd.linear.x = 0.0
            hold_cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(hold_cmd)
            return

        lateral = -self.latest_dock_pose.pose.position.x
        distance = self.latest_dock_pose.pose.position.z
        bearing_angle = np.arctan2(lateral, distance)
        
        # [추가] 여기서 미리 Yaw를 계산해서 저장해둠 (모니터링용)
        q = self.latest_dock_pose.pose.orientation
        # 쿼터니언이 유효할 때만 계산
        if not (q.w == 0.0 and q.x == 0.0 and q.y == 0.0 and q.z == 0.0):
            _, self.current_yaw_raw, _ = euler_from_quaternion(q.x, q.y, q.z, q.w)
        
        cmd = Twist()
        
        if self.state == DockingState.ROTATE_TO_TARGET:
            self.get_logger().info(
                f"🔄 ROTATING | Cur: {math.degrees(bearing_angle):.1f}° / Thresh: {math.degrees(self.rotation_threshold):.1f}°", 
                throttle_duration_sec=0.5
            )
            
            if abs(bearing_angle) > self.rotation_threshold:
                cmd.angular.z = np.clip(3.0 * bearing_angle, -self.rotation_speed, self.rotation_speed)
            else:
                self.state = DockingState.APPROACH
                self.get_logger().info("✅ Rotation aligned. Moving to APPROACH.")
                
        elif self.state == DockingState.APPROACH:
            self.get_logger().info(
                f"➡️ APPROACH | Dist: {distance:.2f}m | Drift: {math.degrees(bearing_angle):.1f}°", 
                throttle_duration_sec=0.5
            )
            
            if abs(bearing_angle) > 0.25: # 약 14도 이상 틀어지면 다시 회전
                self.state = DockingState.ROTATE_TO_TARGET
                return
            
            if distance > (self.docking_threshold + 0.5):
                cmd.linear.x = self.approach_speed
                cmd.angular.z = np.clip(4.0 * bearing_angle, -0.6, 0.6)
            else:
                self.state = DockingState.FINAL_ALIGN
                
        elif self.state == DockingState.FINAL_ALIGN:
            if distance > self.docking_threshold:
                cmd.linear.x = self.final_speed
                cmd.angular.z = np.clip(2.5 * bearing_angle, -0.2, 0.2)
            else:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                    
                self.state = DockingState.ALIGN_TO_MARKER
                self.get_logger().info(f"🎯 Distance Reached. Starting Grid Snap.")

        elif self.state == DockingState.ALIGN_TO_MARKER:
                        
            # EMA 필터 적용 (노이즈/튀는 값 억제)
            if self.filtered_yaw is None:
                self.filtered_yaw = self.current_yaw_raw
            else:
                self.filtered_yaw = (self.alpha * self.current_yaw_raw) + ((1 - self.alpha) * self.filtered_yaw)
            
            # 제어에는 필터된 값 사용
            yaw_error = self.filtered_yaw
            
            self.get_logger().info(
                f"📐 ALIGNING | Marker Yaw: {math.degrees(yaw_error):.2f}°",
                throttle_duration_sec=0.2
            )
            
            # 허용 오차 (약 1.5도)
            if abs(yaw_error) > 0.02:  # 약 1도                    
                # ============ 재정렬 시 속도 감소 ============
                if self.realignment_count > 0:
                    # 재정렬 중: 더 느리고 부드럽게
                    if abs(yaw_error) > 0.05:
                        gain = 4.0  
                        limit = 0.15  
                    else:
                        gain = 3.0   
                        limit = 0.15  
                    min_speed = 0.02  
                else:
                    # 첫 정렬: 기존 속도
                    # 1. 오차가 큰 경우 (예: 2.8도/0.05rad 이상): 강한 P-제어
                    if abs(yaw_error) > 0.05:  # 5.7도 이상
                        gain = 8.0
                        limit = 0.3
                        
                    # 2. 중간 오차 (예: 1.0도/0.017rad ~ 2.8도 사이): 부드러운 감속 제어
                    else:
                        gain = 4.0
                        limit = 0.15
                    min_speed = 0.03
                    
                
                speed = -np.clip(gain * yaw_error, -limit, limit)
                
                # 최소 회전 속도 보장 (Dead zone 극복)
                if abs(speed) < min_speed:
                    speed = min_speed if yaw_error > 0 else -min_speed
                
                cmd.linear.x = 0.0
                cmd.angular.z = speed
            else:
                # 정렬 완료 -> 검증 단계로
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                
                self.filtered_yaw = None
                
                self.verification_start_time = self.get_clock().now()
                self.state = DockingState.VERIFY_ALIGNMENT
                self.get_logger().info("⏸️ Marker Alignment Done. Verifying...")

        elif self.state == DockingState.VERIFY_ALIGNMENT:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
                    
            if self.filtered_yaw is None:
                self.filtered_yaw = self.current_yaw_raw
            else:
                self.filtered_yaw = (self.alpha * self.current_yaw_raw) + ((1 - self.alpha) * self.filtered_yaw)
            
            wait_time = (self.get_clock().now() - self.verification_start_time).nanoseconds / 1e9
            
            if wait_time < 0.5:
                self.get_logger().info(f"⏳ Stabilizing... ({wait_time:.1f}/0.5s)", throttle_duration_sec=0.5)
            else:
                final_yaw_rad = self.filtered_yaw
                
                # 2. 판단을 위해 도로 변환
                final_deg_error = math.degrees(abs(final_yaw_rad))
                
                TARGET_TOLERANCE_DEG = 1.2 
                
                if final_deg_error > TARGET_TOLERANCE_DEG:
                    if self.realignment_count < 3:
                        # 재시도 횟수 남아있으면 -> 다시 정렬 상태로 복귀
                        self.realignment_count += 1
                        self.state = DockingState.ALIGN_TO_MARKER
                        self.get_logger().warn(
                            f"🔄 Drift detected! Error: {math.degrees(final_deg_error):.2f}° -> Re-aligning (Retry {self.realignment_count}/3)"
                        )
                    else:                       
                        if final_deg_error <= 5.0:
                            # Case A: 5도 이내 -> 허용 범위 성공 처리
                            self.get_logger().warn(
                                f"⚠️ Alignment acceptable (Retries exhausted). Final Error: {final_deg_error:.2f}° (Target < 5.0°)"
                            )
                            self._finish_docking(success=True) # 성공으로 처리하여 종료
                        else:
                            # Case B: 5도 초과 -> 실제 실패
                            self.get_logger().error(
                                f"❌ Alignment Failed. Deviation too large. Final Error: {final_deg_error:.2f}°"
                            )
                            self._finish_docking(success=False) # 실패 처리
                else:
                    # 정렬 성공
                    self.get_logger().info(
                        f"✅ VERIFIED! Stable. Final Error: {final_deg_error:.2f}°"
                    )
                    self._finish_docking(success=True)
                    
        elif self.state == DockingState.DOCKED:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            
        self.cmd_vel_pub.publish(cmd)
        
    def stop_robot(self):
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
        
        self.state = DockingState.IDLE
        
        # 재시도 카운터 초기화
        self.realignment_count = 0
        self.verification_start_time = None
        
        self.get_logger().info("🛑 Robot Stopped and Controller Reset to IDLE")

def main(args=None):
    rclpy.init(args=args)
    node = SimplePrecisionDocking()
    
    # MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()