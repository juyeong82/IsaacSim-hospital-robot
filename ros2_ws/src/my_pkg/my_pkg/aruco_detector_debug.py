import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
from geometry_msgs.msg import PoseStamped
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation
import sys

# 카메라 설정 딕셔너리
CAMERA_MODES = {
    '1': {
        'name': 'Gripper Camera',
        'rgb_topic': '/gripper_camera/rgb',
        'info_topic': '/gripper_camera/camera_info',
        'frame_id': 'gripper_Camera'
    },
    '2': {
        'name': 'Right Camera',
        'rgb_topic': '/right_camera/rgb',
        'info_topic': '/right_camera/camera_info',
        'frame_id': 'right_Camera'
    },
    '3': {
        'name': 'Left Camera',
        'rgb_topic': '/left_camera/rgb',
        'info_topic': '/left_camera/camera_info',
        'frame_id': 'left_Camera'
    },
    '4': {
        'name': 'Front Camera',
        'rgb_topic': '/front_camera/rgb',
        'info_topic': '/front_camera/camera_info',
        'frame_id': 'Camera'  # Front는 보통 'Camera' 프레임 사용
    }
}

class ArucoDetector(Node):
    def __init__(self, selected_mode):
        super().__init__('aruco_detector_debug')
        
        # 선택된 모드 설정 저장
        self.mode_name = selected_mode['name']
        self.source_frame = selected_mode['frame_id']
        
        # [수정] 마커 및 검출기 설정 (OpenCV 4.7+ 대응)
        self.marker_size = 0.13
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.params)
        
        self.bridge = CvBridge()
        
        # TF 리스너
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # [동적 구독] 선택된 토픽으로 Subscriber 생성
        self.create_subscription(Image, selected_mode['rgb_topic'], self.image_callback, 10)
        self.create_subscription(CameraInfo, selected_mode['info_topic'], self.info_callback, 10)
        
        # RMPFlow 타겟 퍼블리셔
        self.pose_pub = self.create_publisher(PoseStamped, '/rmp_target_pose', 10)
        
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # 그리퍼 Orientation (바닥 보기)
        euler = np.array([0, np.pi/2, 0])  # roll, pitch, yaw
        rot = Rotation.from_euler('xyz', euler)
        self.default_quat = rot.as_quat()  # [x, y, z, w]
        
        self.get_logger().info(f"✅ Debug Mode Started using: [{self.mode_name}]")
        self.get_logger().info(f"   - Frame: {self.source_frame}")
        self.get_logger().info(f"   - Topic: {selected_mode['rgb_topic']}")

    def info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        if self.camera_matrix is None: return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)

        if ids is not None:
            marker_half = self.marker_size / 2.0
            obj_points = np.array([
                [-marker_half, marker_half, 0],
                [marker_half, marker_half, 0],
                [marker_half, -marker_half, 0],
                [-marker_half, -marker_half, 0]
            ], dtype=np.float32)

            for i in range(len(ids)):
                _, rvec, tvec = cv2.solvePnP(
                    obj_points, 
                    corners[i][0], 
                    self.camera_matrix, 
                    self.dist_coeffs
                )

                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)

                try:
                    # =========================================================
                    # [수정] 로컬 오프셋 적용 로직 (Matrix 연산)
                    # =========================================================
                    R, _ = cv2.Rodrigues(rvec)
                    
                    T_cam_marker = np.eye(4)
                    T_cam_marker[:3, :3] = R
                    T_cam_marker[:3, 3] = tvec.squeeze()
                    
                    T_offset = np.eye(4)
                    # 아루코 마커 기준 그립을 위한 에셋상단 위치 
                    T_offset[0, 3] = 0.0      # X (좌우)
                    T_offset[1, 3] = 0.1      # Y (위아래, 위가 +)
                    T_offset[2, 3] = -0.04    # Z (앞뒤, 뒤가 -)
                    
                    T_cam_target = T_cam_marker @ T_offset
                    # =========================================================

                    target_frame = "base_link"
                    
                    # [수정] 선택된 모드의 Frame ID 사용
                    p_cam = PoseStamped()
                    p_cam.header.frame_id = self.source_frame # 동적 할당됨
                    p_cam.header.stamp = msg.header.stamp
                    
                    p_cam.pose.position.x = T_cam_target[0, 3]
                    p_cam.pose.position.y = T_cam_target[1, 3]
                    p_cam.pose.position.z = T_cam_target[2, 3]
                    p_cam.pose.orientation.w = 1.0

                    transform = self.tf_buffer.lookup_transform(
                        target_frame,
                        self.source_frame, # 동적 할당됨
                        rclpy.time.Time(), 
                        timeout=rclpy.duration.Duration(seconds=0.1)
                    )
                    
                    p_robot_pose = tf2_geometry_msgs.do_transform_pose(p_cam.pose, transform)
                    
                    robot_x = p_robot_pose.position.x
                    robot_y = p_robot_pose.position.y
                    robot_z = p_robot_pose.position.z

                    self.get_logger().info(f"ID {ids[i][0]}: Target -> X:{robot_x:.3f}, Y:{robot_y:.3f}, Z:{robot_z:.3f}")

                    target_msg = PoseStamped()
                    target_msg.header.frame_id = target_frame
                    target_msg.header.stamp = self.get_clock().now().to_msg()
                    target_msg.pose.position.x = robot_x
                    target_msg.pose.position.y = robot_y
                    target_msg.pose.position.z = robot_z
                    
                    target_msg.pose.orientation.x = self.default_quat[0]
                    target_msg.pose.orientation.y = self.default_quat[1]
                    target_msg.pose.orientation.z = self.default_quat[2]
                    target_msg.pose.orientation.w = self.default_quat[3]
                    
                    self.pose_pub.publish(target_msg)

                except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
                    continue

        cv2.imshow("Aruco View", frame)
        cv2.waitKey(1)

def main():
    # -----------------------------------------------------
    # [터미널 선택 인터페이스]
    # -----------------------------------------------------
    print("\n===================================")
    print(" 📷 Select Camera Mode")
    print("===================================")
    print(" 1. Gripper Camera")
    print(" 2. Right Camera")
    print(" 3. Left Camera")
    print(" 4. Front Camera")
    print("===================================")
    
    while True:
        choice = input("Enter number (1-4): ").strip()
        if choice in CAMERA_MODES:
            selected_config = CAMERA_MODES[choice]
            break
        print("❌ Invalid input. Please enter 1, 2, 3, or 4.")

    rclpy.init()
    # 선택된 설정을 Node에 전달
    node = ArucoDetector(selected_config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()