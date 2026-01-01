#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from apriltag_msgs.msg import AprilTagDetectionArray
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped
import cv2
import numpy as np

class DockPosePublisher(Node):
    def __init__(self):
        super().__init__('dock_pose_publisher')
        
        self.target_id = 4       
        self.tag_size = 0.25
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # [설정] 카메라가 로봇 중심보다 얼마나 앞에 있는지 (미터 단위)
        self.camera_x_offset = 0.2 
        
        self.create_subscription(CameraInfo, '/front_camera/camera_info', self.camera_info_callback, 10)
        self.create_subscription(AprilTagDetectionArray, '/detections', self.detection_callback, 10)
        self.publisher = self.create_publisher(PoseStamped, 'detected_dock_pose', 10)
        
        self.get_logger().info(f"🚀 Fixed Mode 2: Y-axis Flipped & Orientation Reset")

    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)

    def detection_callback(self, msg):
        if self.camera_matrix is None:
            return

        for detection in msg.detections:
            det_id = detection.id[0] if isinstance(detection.id, (list, tuple)) else detection.id
            if det_id != self.target_id: continue

            # AprilTag 코너 좌표
            image_points = np.array([
                [detection.corners[0].x, detection.corners[0].y],
                [detection.corners[1].x, detection.corners[1].y],
                [detection.corners[2].x, detection.corners[2].y],
                [detection.corners[3].x, detection.corners[3].y]
            ], dtype=np.float32)

            s = self.tag_size / 2.0
            # [중요] AprilTag 표준 좌표계 (Counter-Clockwise)
            object_points = np.array([
                [-s,  s, 0], # Bottom Left
                [ s,  s, 0], # Bottom Right
                [ s, -s, 0], # Top Right
                [-s, -s, 0]  # Top Left
            ], dtype=np.float32)

            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points, self.camera_matrix, self.dist_coeffs
            )

            if success:
                raw_z = tvec[2][0] # 거리
                raw_x = tvec[0][0] # 좌우
                
                # 1. 좌표 변환 (Y축 부호 수정됨)
                # 이전: base_y = -raw_x (왼쪽이 +, 오른쪽이 -라고 가정했으나 반대일 수 있음)
                # 수정: base_y = raw_x (직접 매핑)
                base_x = raw_z + self.camera_x_offset
                base_y = raw_x 
                
                pose_msg = PoseStamped()
                pose_msg.header.frame_id = "base_link" 
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                
                pose_msg.pose.position.x = base_x
                pose_msg.pose.position.y = base_y
                pose_msg.pose.position.z = 0.0 
                
                # 2. 방향 설정 (테스트용: 0도)
                # 로봇과 독이 같은 방향을 보고 있다고 가정 (진입 방향 정렬)
                pose_msg.pose.orientation.x = 0.0
                pose_msg.pose.orientation.y = 0.0
                pose_msg.pose.orientation.z = 0.0
                pose_msg.pose.orientation.w = 1.0 # 0도 (Identity)
                
                self.publisher.publish(pose_msg)
                
                self.get_logger().info(
                    f"🎯 Target: X={base_x:.2f} (Front), Y={base_y:.2f} (Side)"
                )
            return

def main(args=None):
    rclpy.init(args=args)
    node = DockPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()