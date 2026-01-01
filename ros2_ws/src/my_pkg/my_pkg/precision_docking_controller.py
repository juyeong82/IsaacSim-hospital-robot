#!/usr/bin/env python3
"""
Precision Docking Controller with State Machine

3-Stage Docking:
1. ROTATE_TO_TARGET: 마커를 향해 회전 (heading alignment)
2. APPROACH: 정면으로 접근하면서 미세 조정
3. FINAL_ALIGN: 정밀 정렬 및 최종 접근
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from geometry_msgs.msg import PoseStamped, Twist
from opennav_docking_msgs.action import DockRobot
import numpy as np
from enum import Enum

class DockingState(Enum):
    IDLE = 0
    ROTATE_TO_TARGET = 1    # 마커를 향해 회전
    APPROACH = 2            # 접근하면서 조정
    FINAL_ALIGN = 3         # 정밀 정렬
    DOCKED = 4

class PrecisionDockingController(Node):
    def __init__(self):
        super().__init__('precision_docking_controller')
        
        # Parameters
        self.declare_parameter('docking_distance_threshold', 0.4)
        self.declare_parameter('rotation_threshold', 0.087)  # 5도
        self.declare_parameter('lateral_threshold', 0.05)    # 5cm
        self.declare_parameter('approach_speed', 0.3)
        self.declare_parameter('rotation_speed', 0.5)
        self.declare_parameter('final_speed', 0.15)
        
        self.docking_threshold = self.get_parameter('docking_distance_threshold').value
        self.rotation_threshold = self.get_parameter('rotation_threshold').value
        self.lateral_threshold = self.get_parameter('lateral_threshold').value
        self.approach_speed = self.get_parameter('approach_speed').value
        self.rotation_speed = self.get_parameter('rotation_speed').value
        self.final_speed = self.get_parameter('final_speed').value
        
        # State
        self.state = DockingState.IDLE
        self.latest_dock_pose = None
        
        # Subscribers
        self.create_subscription(
            PoseStamped,
            'detected_dock_pose',
            self.dock_pose_callback,
            10
        )
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Action Server
        self._action_server = ActionServer(
            self,
            DockRobot,
            'dock_robot',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )
        
        # Control loop
        self.create_timer(0.05, self.control_loop)  # 20Hz
        
        self.get_logger().info('🎯 Precision Docking Controller Started')
        
    def goal_callback(self, goal_request):
        self.get_logger().info('📥 Docking goal received')
        return GoalResponse.ACCEPT
        
    def cancel_callback(self, goal_handle):
        self.get_logger().info('❌ Docking cancelled')
        return CancelResponse.ACCEPT
        
    async def execute_callback(self, goal_handle):
        self.get_logger().info('🚀 Starting docking sequence')
        self.state = DockingState.ROTATE_TO_TARGET
        
        result = DockRobot.Result()
        feedback = DockRobot.Feedback()
        
        rate = self.create_rate(20)  # 20Hz
        
        while self.state != DockingState.DOCKED:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.stop_robot()
                result.success = False
                return result
                
            # Publish feedback
            feedback.state = self.state.value
            if self.latest_dock_pose:
                feedback.distance_to_dock = float(self.latest_dock_pose.pose.position.z)
            goal_handle.publish_feedback(feedback)
            
            await rate.sleep()
            
        self.stop_robot()
        result.success = True
        goal_handle.succeed()
        self.get_logger().info('✅ Docking completed!')
        return result
        
    def dock_pose_callback(self, msg):
        """Camera frame에서 받은 dock pose 저장"""
        self.latest_dock_pose = msg
        
    def control_loop(self):
        """Main control loop - State machine"""
        if self.state == DockingState.IDLE or self.latest_dock_pose is None:
            return
            
        # Camera frame에서 데이터 가져오기
        # X: 좌우, Y: 상하, Z: 거리
        lateral = self.latest_dock_pose.pose.position.x      # 좌우 오차
        distance = self.latest_dock_pose.pose.position.z     # 전방 거리
        
        # Bearing angle: 마커를 향하는 각도
        bearing_angle = np.arctan2(lateral, distance)
        
        cmd = Twist()
        
        # ============================================
        # State Machine
        # ============================================
        
        if self.state == DockingState.ROTATE_TO_TARGET:
            """Stage 1: 마커를 정면으로 향하도록 회전"""
            
            if abs(bearing_angle) > self.rotation_threshold:
                # 회전만 수행
                cmd.linear.x = 0.0
                cmd.angular.z = np.clip(
                    2.0 * bearing_angle,
                    -self.rotation_speed,
                    self.rotation_speed
                )
                self.get_logger().info(
                    f"🔄 ROTATE: angle={np.degrees(bearing_angle):.1f}°, "
                    f"dist={distance:.2f}m"
                )
            else:
                # 회전 완료 → 접근 단계로
                self.state = DockingState.APPROACH
                self.get_logger().info("✅ Rotation complete → APPROACH")
                
        elif self.state == DockingState.APPROACH:
            """Stage 2: 접근하면서 미세 조정"""
            
            if distance > 0.8:
                # 멀리 있을 때: 빠르게 접근
                cmd.linear.x = self.approach_speed
                cmd.angular.z = np.clip(
                    1.5 * bearing_angle,
                    -0.3,
                    0.3
                )
                self.get_logger().info(
                    f"➡️ APPROACH: dist={distance:.2f}m, "
                    f"lateral={lateral:.3f}m"
                )
            else:
                # 가까워지면 → 정밀 정렬 단계로
                self.state = DockingState.FINAL_ALIGN
                self.get_logger().info("✅ Close enough → FINAL_ALIGN")
                
        elif self.state == DockingState.FINAL_ALIGN:
            """Stage 3: 정밀 정렬 및 최종 접근"""
            
            if distance > self.docking_threshold:
                # 매우 천천히 접근
                cmd.linear.x = self.final_speed
                # 미세 각도 조정
                cmd.angular.z = np.clip(
                    3.0 * bearing_angle,
                    -0.2,
                    0.2
                )
                self.get_logger().info(
                    f"🎯 FINAL: dist={distance:.2f}m, "
                    f"angle={np.degrees(bearing_angle):.1f}°"
                )
            else:
                # 도킹 완료!
                self.state = DockingState.DOCKED
                self.get_logger().info("✅✅✅ DOCKED!")
                
        # Publish command
        self.cmd_vel_pub.publish(cmd)
        
    def stop_robot(self):
        """로봇 정지"""
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
        self.state = DockingState.IDLE

def main(args=None):
    rclpy.init(args=args)
    controller = PrecisionDockingController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
        
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()