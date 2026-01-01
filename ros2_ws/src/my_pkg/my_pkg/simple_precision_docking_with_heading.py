#!/usr/bin/env python3
"""
Simple Precision Docking Controller with Absolute Heading Alignment
AprilTag를 보면 자동으로 도킹 시작 + 최종 단계에서 90도 단위 정렬
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_srvs.srv import Trigger
from tf2_ros import TransformListener, Buffer
import numpy as np
from enum import Enum

def euler_from_quaternion(quat):
    """
    Quaternion을 Euler 각도로 변환
    Args: quat = [x, y, z, w]
    Returns: (roll, pitch, yaw)
    """
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

class DockingState(Enum):
    IDLE = 0
    ROTATE_TO_TARGET = 1    # 마커를 향해 회전
    APPROACH = 2            # 접근하면서 조정
    FINAL_ALIGN = 3         # 정밀 정렬
    HEADING_ALIGN = 4       # 절대 방향 정렬 (90도 단위)
    DOCKED = 5

class SimplePrecisionDocking(Node):
    def __init__(self):
        super().__init__('simple_precision_docking')
        
        # Parameters
        self.declare_parameter('docking_distance_threshold', 0.4)
        self.declare_parameter('rotation_threshold', 0.087)  # 5도
        self.declare_parameter('approach_speed', 0.3)
        self.declare_parameter('rotation_speed', 0.5)
        self.declare_parameter('final_speed', 0.15)
        self.declare_parameter('auto_start', True)
        self.declare_parameter('align_to_grid', True)  # 90도 단위 정렬 활성화
        self.declare_parameter('target_yaw', 0.0)  # 목표 yaw (기본: 90도)
        
        self.docking_threshold = self.get_parameter('docking_distance_threshold').value
        self.rotation_threshold = self.get_parameter('rotation_threshold').value
        self.approach_speed = self.get_parameter('approach_speed').value
        self.rotation_speed = self.get_parameter('rotation_speed').value
        self.final_speed = self.get_parameter('final_speed').value
        self.auto_start = self.get_parameter('auto_start').value
        self.align_to_grid = self.get_parameter('align_to_grid').value
        self.target_yaw = self.get_parameter('target_yaw').value
        
        # State
        self.state = DockingState.IDLE
        self.latest_dock_pose = None
        self.latest_pose_time = None
        self.docking_enabled = self.auto_start
        self.current_yaw = None
        
        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Subscribers
        self.create_subscription(
            PoseStamped,
            'detected_dock_pose',
            self.dock_pose_callback,
            10
        )
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Services
        self.start_service = self.create_service(
            Trigger,
            'start_docking',
            self.start_docking_callback
        )
        self.stop_service = self.create_service(
            Trigger,
            'stop_docking',
            self.stop_docking_callback
        )
        
        # Control loop
        self.create_timer(0.05, self.control_loop)  # 20Hz
        
        self.get_logger().info('🎯 Precision Docking with Heading Alignment Started')
        if self.auto_start:
            self.get_logger().info('✅ Auto-start enabled')
        if self.align_to_grid:
            self.get_logger().info(f'✅ Grid alignment enabled: target={np.degrees(self.target_yaw):.0f}°')
        
    def get_current_yaw(self):
        """TF를 통해 현재 로봇의 yaw 각도 획득"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom',
                'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            q = transform.transform.rotation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            return yaw
            
        except Exception as e:
            # TF 실패 시 None 반환
            return None
    
    def normalize_angle(self, angle):
        """각도를 -π ~ π 범위로 정규화"""
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle
        
    def start_docking_callback(self, request, response):
        self.docking_enabled = True
        self.state = DockingState.IDLE
        response.success = True
        response.message = "Docking enabled"
        self.get_logger().info('🚀 Docking enabled via service')
        return response
        
    def stop_docking_callback(self, request, response):
        self.docking_enabled = False
        self.stop_robot()
        response.success = True
        response.message = "Docking stopped"
        self.get_logger().info('🛑 Docking stopped via service')
        return response
        
    def dock_pose_callback(self, msg):
        """Camera frame에서 받은 dock pose 저장"""
        self.latest_dock_pose = msg
        self.latest_pose_time = self.get_clock().now()
        
        # 자동 시작
        if self.docking_enabled and self.state == DockingState.IDLE:
            distance = msg.pose.position.z
            if distance > 0.5:
                self.state = DockingState.ROTATE_TO_TARGET
                self.get_logger().info(f'🎯 Auto-start docking! Distance={distance:.2f}m')
        
    def control_loop(self):
        """Main control loop - State machine"""
        
        if not self.docking_enabled:
            return
            
        if self.state == DockingState.IDLE or self.latest_dock_pose is None:
            return
        
        # 마커 감지 체크
        current_time = self.get_clock().now()
        if self.latest_pose_time is not None:
            time_since_detection = (current_time - self.latest_pose_time).nanoseconds / 1e9
            if time_since_detection > 1.0:
                self.get_logger().warn(
                    f'⚠️ Marker lost for {time_since_detection:.1f}s - STOPPING!'
                )
                self.stop_robot()
                return
        
        # 현재 yaw 획득
        self.current_yaw = self.get_current_yaw()
            
        # Camera frame 데이터
        lateral = -self.latest_dock_pose.pose.position.x  # 부호 반전
        distance = self.latest_dock_pose.pose.position.z
        bearing_angle = np.arctan2(lateral, distance)
        
        cmd = Twist()
        
        # ============================================
        # State Machine
        # ============================================
        
        if self.state == DockingState.ROTATE_TO_TARGET:
            """Stage 1: 마커를 향해 회전"""
            
            if abs(bearing_angle) > self.rotation_threshold:
                cmd.linear.x = 0.0
                cmd.angular.z = np.clip(
                    3.0 * bearing_angle,
                    -self.rotation_speed,
                    self.rotation_speed
                )
                self.get_logger().info(
                    f"🔄 ROTATE: angle={np.degrees(bearing_angle):.1f}°, "
                    f"dist={distance:.2f}m",
                    throttle_duration_sec=0.5
                )
            else:
                self.state = DockingState.APPROACH
                self.get_logger().info("✅ Rotation complete → APPROACH")
                
        elif self.state == DockingState.APPROACH:
            """Stage 2: 접근하면서 미세 조정"""
            
            if distance > 0.8:
                cmd.linear.x = self.approach_speed
                cmd.angular.z = np.clip(
                    3.5 * bearing_angle,
                    -0.5,
                    0.5
                )
                self.get_logger().info(
                    f"➡️ APPROACH: dist={distance:.2f}m, "
                    f"lateral={lateral:.3f}m, angle={np.degrees(bearing_angle):.1f}°",
                    throttle_duration_sec=0.5
                )
            else:
                self.state = DockingState.FINAL_ALIGN
                self.get_logger().info("✅ Close enough → FINAL_ALIGN")
                
        elif self.state == DockingState.FINAL_ALIGN:
            """Stage 3: 정밀 정렬 및 최종 접근"""
            
            if distance > self.docking_threshold:
                cmd.linear.x = self.final_speed
                cmd.angular.z = np.clip(
                    4.0 * bearing_angle,
                    -0.3,
                    0.3
                )
                self.get_logger().info(
                    f"🎯 FINAL: dist={distance:.2f}m, "
                    f"angle={np.degrees(bearing_angle):.1f}°",
                    throttle_duration_sec=0.5
                )
            else:
                # 거리 도달 → 절대 방향 정렬로
                if self.align_to_grid and self.current_yaw is not None:
                    self.state = DockingState.HEADING_ALIGN
                    self.get_logger().info("✅ Distance reached → HEADING_ALIGN")
                else:
                    self.state = DockingState.DOCKED
                    self.stop_robot()
                    self.get_logger().info("✅✅✅ DOCKED!")
                    
        elif self.state == DockingState.HEADING_ALIGN:
            """Stage 4: 절대 방향 정렬 (target_yaw로)"""
            
            if self.current_yaw is None:
                self.get_logger().warn("⚠️ No TF available, skipping heading align")
                self.state = DockingState.DOCKED
                self.stop_robot()
                return
            
            # ✅ 목표 yaw와 현재 yaw의 차이
            yaw_error = self.normalize_angle(self.target_yaw - self.current_yaw)
            
            if abs(yaw_error) > 0.05:  # 약 3도
                cmd.linear.x = 0.0
                cmd.angular.z = np.clip(
                    2.0 * yaw_error,
                    -0.3,
                    0.3
                )
                self.get_logger().info(
                    f"🧭 HEADING_ALIGN: current={np.degrees(self.current_yaw):.1f}°, "
                    f"target={np.degrees(self.target_yaw):.1f}°, "
                    f"error={np.degrees(yaw_error):.1f}°",
                    throttle_duration_sec=0.5
                )
            else:
                self.state = DockingState.DOCKED
                self.stop_robot()
                self.get_logger().info(
                    f"✅✅✅ DOCKED! Final heading: {np.degrees(self.current_yaw):.1f}°"
                )
                        
        elif self.state == DockingState.DOCKED:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            
        self.cmd_vel_pub.publish(cmd)
        
    def stop_robot(self):
        """로봇 정지"""
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
        self.state = DockingState.IDLE

def main(args=None):
    rclpy.init(args=args)
    controller = SimplePrecisionDocking()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
        
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()