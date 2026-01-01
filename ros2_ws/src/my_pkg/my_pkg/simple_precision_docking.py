#!/usr/bin/env python3
"""
Simple Precision Docking Controller (TF-based Grid Alignment)
TF로 map->base_link transform 직접 읽어서 실시간 반영
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_srvs.srv import Trigger
import numpy as np
import math
from enum import Enum
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

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
    ALIGN_TO_GRID = 4
    DOCKED = 5

class SimplePrecisionDocking(Node):
    def __init__(self):
        super().__init__('simple_precision_docking')
        
        # Parameters
        self.declare_parameter('docking_distance_threshold', 0.40)
        self.declare_parameter('rotation_threshold', 0.087)
        self.declare_parameter('approach_speed', 0.3)
        self.declare_parameter('rotation_speed', 0.5)
        self.declare_parameter('final_speed', 0.15)
        self.declare_parameter('auto_start', True)
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
        
        # TF Setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.current_yaw = 0.0
        
        # Subscribers
        self.create_subscription(PoseStamped, 'detected_dock_pose', self.dock_pose_callback, 10)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Services
        self.create_service(Trigger, 'start_docking', self.start_docking_callback)
        self.create_service(Trigger, 'stop_docking', self.stop_docking_callback)
        
        # Control loop (20Hz)
        self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('🎯 Simple Precision Docking Started (TF-based)')

    def get_robot_yaw_from_tf(self):
        """TF에서 map->base_link transform 읽어 Yaw 추출 (실시간)"""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time()  # 최신 데이터
            )
            q = transform.transform.rotation
            _, _, yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
            return yaw, True
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=2.0)
            return 0.0, False

    def start_docking_callback(self, request, response):
        self.docking_enabled = True
        self.state = DockingState.IDLE
        response.success = True
        response.message = "Docking enabled"
        return response
        
    def stop_docking_callback(self, request, response):
        self.docking_enabled = False
        self.stop_robot()
        response.success = True
        response.message = "Docking stopped"
        return response
        
    def dock_pose_callback(self, msg):
        self.latest_dock_pose = msg
        self.latest_pose_time = self.get_clock().now()
        
        if self.docking_enabled and self.state == DockingState.IDLE:
            distance = msg.pose.position.z
            if distance > 0.5:
                self.state = DockingState.ROTATE_TO_TARGET
                self.get_logger().info(f'🚀 Auto-start! Detect Dist={distance:.2f}m')
        
    def control_loop(self):
        if not self.docking_enabled:
            return

        # TF에서 실시간 Yaw 읽기 (ALIGN_TO_GRID 단계에서만)
        if self.state == DockingState.ALIGN_TO_GRID or self.state == DockingState.DOCKED:
            yaw, success = self.get_robot_yaw_from_tf()
            if success:
                self.current_yaw = yaw

        if self.state != DockingState.ALIGN_TO_GRID and self.state != DockingState.DOCKED:
            if self.state == DockingState.IDLE or self.latest_dock_pose is None:
                if self.state == DockingState.IDLE:
                    self.get_logger().info("💤 IDLE: Waiting for marker...", throttle_duration_sec=2.0)
                return
            
            current_time = self.get_clock().now()
            if self.latest_pose_time is not None:
                time_since_detection = (current_time - self.latest_pose_time).nanoseconds / 1e9
                if time_since_detection > 1.0:
                    self.get_logger().warn('⚠️ Marker lost - STOPPING!')
                    self.stop_robot()
                    return

            lateral = -self.latest_dock_pose.pose.position.x
            distance = self.latest_dock_pose.pose.position.z
            bearing_angle = np.arctan2(lateral, distance)
        
        cmd = Twist()
        
        if self.state == DockingState.ROTATE_TO_TARGET:
            self.get_logger().info(
                f"🔄 ROTATING | Cur Angle: {math.degrees(bearing_angle):.1f}° / Thresh: {math.degrees(self.rotation_threshold):.1f}°", 
                throttle_duration_sec=0.5
            )

            if abs(bearing_angle) > self.rotation_threshold:
                cmd.angular.z = np.clip(3.0 * bearing_angle, -self.rotation_speed, self.rotation_speed)
            else:
                self.get_logger().info("✅ Rotation aligned. Moving to APPROACH.")
                self.state = DockingState.APPROACH
                
        elif self.state == DockingState.APPROACH:
            self.get_logger().info(
                f"➡️ APPROACH | Dist: {distance:.2f}m | Drift: {math.degrees(bearing_angle):.1f}°", 
                throttle_duration_sec=0.5
            )

            if abs(bearing_angle) > 0.175:
                self.get_logger().warn(f"⚠️ Drift too high ({math.degrees(bearing_angle):.1f}°). Correcting orientation.")
                self.state = DockingState.ROTATE_TO_TARGET
                return
            
            slowdown_distance = self.docking_threshold + 0.5 

            if distance > slowdown_distance:
                # 아직 목표 지점까지 여유가 많음 -> 빠른 접근
                cmd.linear.x = self.approach_speed
                cmd.angular.z = np.clip(5.0 * bearing_angle, -0.8, 0.8)
            else:
                # 목표 지점 근처 도달 -> FINAL_ALIGN (정밀/감속) 모드로 전환
                self.get_logger().info(f"📉 Slowing down for FINAL_ALIGN. (Dist: {distance:.2f}m)")
                self.state = DockingState.FINAL_ALIGN
                
        elif self.state == DockingState.FINAL_ALIGN:
            self.get_logger().info(
                f"🔍 FINAL APP | Dist: {distance:.3f}m / Goal: {self.docking_threshold:.3f}m", 
                throttle_duration_sec=0.5
            )

            if distance > self.docking_threshold:
                cmd.linear.x = self.final_speed
                cmd.angular.z = np.clip(4.0 * bearing_angle, -0.3, 0.3)
            else:
                self.stop_robot()
                self.state = DockingState.ALIGN_TO_GRID
                self.get_logger().info(f"✅ Position Reached (Dist: {distance:.3f}m). Starting Grid Snap (TF-based).")

        elif self.state == DockingState.ALIGN_TO_GRID:
            # 가장 가까운 90도 배수 찾기
            target_yaw = round(self.current_yaw / (math.pi / 2.0)) * (math.pi / 2.0)
            yaw_error = target_yaw - self.current_yaw
            
            while yaw_error > math.pi: yaw_error -= 2 * math.pi
            while yaw_error < -math.pi: yaw_error += 2 * math.pi
            
            self.get_logger().info(
                f"🧭 SNAPPING (TF) | Cur: {math.degrees(self.current_yaw):.1f}° -> Tgt: {math.degrees(target_yaw):.0f}° | Err: {math.degrees(yaw_error):.2f}°",
                throttle_duration_sec=0.2  # 더 자주 로깅
            )

            if abs(yaw_error) > 0.02:  # 약 0.1도
                cmd.linear.x = 0.0
                # 오버슛 방지: 에러 크면 강하게, 작으면 약하게
                if abs(yaw_error) > 0.02:  # 5.7도 이상
                    cmd.angular.z = np.clip(6.0 * yaw_error, -0.5, 0.5)
                else:
                    cmd.angular.z = np.clip(4.0 * yaw_error, -0.15, 0.15)
            else:
                self.state = DockingState.DOCKED
                self.stop_robot()
                self.get_logger().info(f"🎉 DOCKED & ALIGNED at Map Yaw: {math.degrees(self.current_yaw):.1f}°")

        elif self.state == DockingState.DOCKED:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.get_logger().info("🅿️ Robot is DOCKED.", throttle_duration_sec=5.0)
            
        self.cmd_vel_pub.publish(cmd)
        
    def stop_robot(self):
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