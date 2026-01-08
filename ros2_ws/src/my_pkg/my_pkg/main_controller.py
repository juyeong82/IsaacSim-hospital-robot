import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient, ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped, Quaternion
from std_msgs.msg import Bool
import math
import time
import numpy as np
from action_msgs.msg import GoalStatus
# [Action Interfaces]
from nav2_msgs.action import NavigateToPose
from moma_interfaces.action import Dock, MoveManipulator, RunDelivery
from moma_interfaces.msg import MarkerArray

from scipy.spatial.transform import Rotation 
import numpy as np

class HospitalOrchestrator(Node):
    def __init__(self):
        super().__init__('hospital_main_node')
        
        # ---------------------------------------------------------
        # 1. 환경 설정 (Room Database & Item Config)
        # ---------------------------------------------------------
        # [설정] 방 별 테이블 중심 좌표 (UI에서 주는 정보라 가정)
        # 형식: "Room Name": {"coords": [x, y, z], "approach": "Left" or "Right"}
        self.room_db = {
            "Nurse Station A (Base)":  {"coords": [23.129, 9.392, 0.0], "approach": "Left"},
            "Ward 102":                {"coords": [24.62435, 14.62949, 0.0], "approach": "Left"},
            "Main Pharmacy (Central)": {"coords": [-9.0, 5.07121, 0.0], "approach": "Left"},
            "Sub Pharmacy": {"coords": [-2.5, 5.07121, 0.0], "approach": "Left"},
            "Clinical Lab (Zone C)":   {"coords": [23.129, 9.392, 0.0], "approach": "Right"}, # 테스트용 (우측접근)
        }

        # 오프셋 기준: 마커 중심으로부터 [x(우), y(하/위), z(앞/뒤)] (OpenCV 좌표계 기준 아님, 마커 자체 로컬 좌표계)
        # ---------------------------------------------------------
        self.item_db = {
            "Blood Sample": {
                "id": 0, 
                "offset": [0.0, 0.03, -0.04]  # 요청하신 블러드 튜브 옵셋
            },
            "Medicine": {
                "id": 1, 
                "offset": [0.0, 0.0, -0.06]     # (예시) 약통은 마커 정중앙 잡기
            },
            "Narcotics": {
                "id": 2, 
                "offset": [0.0, 0.05, -0.02]  # (예시) 금고 손잡이 위치 등
            },
        }

        # [설정] 도킹 오프셋 (테이블 중심 기준)
        # Left Approach 기준 (User Provided)
        # Table: (23.129, 9.392) -> Dock: (25.603, 8.400)
        # Diff: X +2.474, Y -0.992
        self.offset_x = 2.474
        self.offset_y = 1.2 # 절대값으로 저장 (Left: -y, Right: +y 적용 예정)
        
        self.quat_left = Quaternion(x=-0.000, y=-0.000, z=0.996, w=0.087)
        self.quat_right = Quaternion(x=-0.000, y=0.000, z=0.996, w=-0.087)
        
        # [추가] 방향별 그립/검증 공통 오리엔테이션 (CLI 테스트 성공 값)
        # Left Approach (Target Y > 0): 카메라가 오른쪽을 보며 파지
        self.grasp_quat_left = Quaternion(x=0.0, y=0.707, z=0.0, w=0.707)

        # Right Approach (Target Y < 0): 카메라가 왼쪽을 보며 파지
        self.grasp_quat_right = Quaternion(x=-0.707, y=0.0, z=0.707, w=0.0)
        # ---------------------------------------------------------
        # 2. ROS2 통신 설정
        # ---------------------------------------------------------
        self.cb_group = ReentrantCallbackGroup()

        # Action Clients
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose', callback_group=self.cb_group)
        self.dock_client = ActionClient(self, Dock, 'dock_robot', callback_group=self.cb_group)
        self.arm_client = ActionClient(self, MoveManipulator, 'move_manipulator', callback_group=self.cb_group)
        
        # Action Server (UI와 통신)
        self._action_server = ActionServer(
            self, RunDelivery, 'run_delivery', 
            self.execute_delivery_callback, 
            callback_group=self.cb_group,
            cancel_callback=self.cancel_callback
        )

        # Vision Control Publishers
        self.pub_enable_left = self.create_publisher(Bool, '/vision/enable_left', 10)
        self.pub_enable_right = self.create_publisher(Bool, '/vision/enable_right', 10)

        # Vision Data Subscribers (일회성 수신용)
        self.detected_markers = {} # ID별 Pose 저장
        self.create_subscription(MarkerArray, '/vision/left_markers', self.vision_cb_left, 10, callback_group=self.cb_group)
        self.create_subscription(MarkerArray, '/vision/right_markers', self.vision_cb_right, 10, callback_group=self.cb_group)

        self.get_logger().info("🏥 Hospital Main Node Ready (Waiting for UI Command...)")

    # [추가] Action Server 취소 요청 수락 콜백
    def cancel_callback(self, goal_handle):
        self.get_logger().info('⚠️ Received Cancel Request!')
        return CancelResponse.ACCEPT

    # [추가] 실행 중 취소 여부 확인 헬퍼 함수
    def check_cancel(self, goal_handle, result):
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.success = False
            result.message = "Task Canceled by User"
            self.get_logger().warn("🛑 Delivery Sequence Canceled!")
            return True # 취소됨
        return False # 취소 안됨

    # [수정] PoseStamped를 받아서 frame_id를 유지하도록 변경
    def apply_grasp_offset(self, base_pose_stamped, offset_xyz):
        """
        base_pose_stamped: PoseStamped 객체 (header 포함)
        """
        # 1. Pose 정보 추출
        base_pose = base_pose_stamped.pose
        
        t = [base_pose.position.x, base_pose.position.y, base_pose.position.z]
        q = [base_pose.orientation.x, base_pose.orientation.y, base_pose.orientation.z, base_pose.orientation.w]
        
        R = Rotation.from_quat(q).as_matrix()
        T_base_marker = np.eye(4)
        T_base_marker[:3, :3] = R
        T_base_marker[:3, 3] = t
        
        # 2. Offset 행렬 생성
        T_offset = np.eye(4)
        T_offset[0, 3] = offset_xyz[0]
        T_offset[1, 3] = offset_xyz[1]
        T_offset[2, 3] = offset_xyz[2]
        
        # 3. 행렬 곱
        T_base_target = T_base_marker @ T_offset
        
        final_pos = T_base_target[:3, 3]
        final_rot = Rotation.from_matrix(T_base_target[:3, :3]).as_quat()
        
        new_pose = PoseStamped()
        
        # [핵심] 하드코딩 삭제 -> 원본 메시지의 frame_id를 그대로 사용
        new_pose.header.frame_id = base_pose_stamped.header.frame_id 
        
        new_pose.pose.position.x = final_pos[0]
        new_pose.pose.position.y = final_pos[1]
        new_pose.pose.position.z = final_pos[2]
        new_pose.pose.orientation.x = final_rot[0]
        new_pose.pose.orientation.y = final_rot[1]
        new_pose.pose.orientation.z = final_rot[2]
        new_pose.pose.orientation.w = final_rot[3]
        
        return new_pose.pose # Action Server에는 Pose 타입으로 전달
    
    # ---------------------------------------------------------
    # Helper: 좌표 계산 로직
    # ---------------------------------------------------------
    def get_docking_pose(self, room_name):
        """테이블 좌표와 접근 방향을 기반으로 도킹 좌표 계산"""
        if room_name not in self.room_db:
            self.get_logger().error(f"❌ Unknown Room: {room_name}")
            return None, None

        info = self.room_db[room_name]
        tx, ty, tz = info['coords']
        approach = info['approach']

        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        
        # 오프셋 적용
        # 현재 맵 기준 X축은 동일하게 증가, Y축만 접근 방향에 따라 반전된다고 가정
        final_x = tx + self.offset_x
        
        if approach == "Left":
            final_y = ty - self.offset_y
            pose.pose.orientation = self.quat_left
        else: # Right
            final_y = ty + self.offset_y
            pose.pose.orientation = self.quat_right
            
        pose.pose.position.x = final_x
        pose.pose.position.y = final_y
        pose.pose.position.z = 0.0
        
        self.get_logger().info(f"📍 Calculated Dock Pose for {room_name} ({approach}): ({final_x:.2f}, {final_y:.2f})")
        return pose, approach

    # ---------------------------------------------------------
    # Helper: 비전 콜백 및 제어
    # ---------------------------------------------------------
    def vision_cb_left(self, msg):
        for m in msg.markers:
            ps = PoseStamped()
            ps.header = msg.header  # 핵심: 여기서 frame_id를 받아옵니다.
            ps.pose = m.pose
            self.detected_markers[m.id] = ps

    def vision_cb_right(self, msg):
        for m in msg.markers:
            ps = PoseStamped()
            ps.header = msg.header  # 핵심: 여기서 frame_id를 받아옵니다.
            ps.pose = m.pose
            self.detected_markers[m.id] = ps

    def set_vision(self, side, enable):
        msg = Bool()
        msg.data = enable
        if side == "Left":
            self.pub_enable_left.publish(msg)
        elif side == "Right":
            self.pub_enable_right.publish(msg)

    async def wait_for_marker(self, target_id, side, timeout=5.0):
        """특정 ID 마커가 보일 때까지 대기"""
        self.detected_markers.clear()
        self.set_vision(side, True) # 카메라 켜기
        
        start_time = time.time()
        self.get_logger().info(f"👀 Scanning for Item ID {target_id} using {side} Camera...")
        
        found_pose = None
        while time.time() - start_time < timeout:
            if target_id in self.detected_markers:
                found_pose = self.detected_markers[target_id]
                self.get_logger().info(f"✅ Found Marker {target_id}!")
                break
            time.sleep(0.1)
            
        # self.set_vision(side, False) # 카메라 끄기
        
        if found_pose is None:
            self.get_logger().error("❌ Marker detection failed (Timeout)")
        
        return found_pose

    # ---------------------------------------------------------
    # Helper: 액션 클라이언트 래퍼 (취소 연동 수정됨)
    # ---------------------------------------------------------
    async def call_nav2(self, pose, main_goal_handle):
        goal = NavigateToPose.Goal()
        goal.pose = pose
        
        self.nav_client.wait_for_server()
        send_goal_future = self.nav_client.send_goal_async(goal)
        nav_goal_handle = await send_goal_future
        
        if not nav_goal_handle.accepted:
            self.get_logger().error("❌ Nav2 Goal Rejected!")
            return False
            
        result_future = nav_goal_handle.get_result_async()
        
        # [핵심] 결과가 나올 때까지 기다리면서, 메인 취소 요청이 들어오는지 감시
        while not result_future.done():
            if main_goal_handle.is_cancel_requested:
                self.get_logger().warn("🛑 Cancelling Nav2 because Main Task was Canceled...")
                await nav_goal_handle.cancel_goal_async() # Nav2에 멈추라고 명령
                return False
            time.sleep(0.1) # CPU 점유율 방지
        
        wrapped_result = result_future.result()
        if wrapped_result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("✅ Nav2 Arrived Successfully")
            return True
        else:
            self.get_logger().error(f"❌ Nav2 Failed or Canceled status: {wrapped_result.status}")
            return False

    async def call_docking(self, main_goal_handle):
        goal = Dock.Goal()
        self.dock_client.wait_for_server()
        send_goal_future = self.dock_client.send_goal_async(goal)
        dock_handle = await send_goal_future
        if not dock_handle.accepted: return False
        
        result_future = dock_handle.get_result_async()
        while not result_future.done():
            if main_goal_handle.is_cancel_requested:
                await dock_handle.cancel_goal_async()
                return False
            time.sleep(0.1)

        res = result_future.result()
        return res.result.success

    async def call_arm(self, action_type, main_goal_handle, pose=None):
        goal = MoveManipulator.Goal()
        goal.action_type = action_type
        if pose:
            ps = PoseStamped()
            ps.header.frame_id = "base_link"
            ps.pose = pose
            goal.target_pose = ps
            
        self.arm_client.wait_for_server()
        send_goal_future = self.arm_client.send_goal_async(goal)
        arm_handle = await send_goal_future
        if not arm_handle.accepted: return False
        
        result_future = arm_handle.get_result_async()
        while not result_future.done():
            if main_goal_handle.is_cancel_requested:
                await arm_handle.cancel_goal_async()
                return False
            time.sleep(0.1)

        res = result_future.result()
        return res.result.success
    # ---------------------------------------------------------
    # Main Workflow: Run Delivery
    # ---------------------------------------------------------
    async def execute_delivery_callback(self, goal_handle):
        request = goal_handle.request
        feedback = RunDelivery.Feedback()
        result = RunDelivery.Result()
        
        # 1. 입력값 파싱
        # task_mode가 비어있으면 기본값 "ALL" 처리
        mode = request.task_mode if request.task_mode else "ALL"
        item_name = request.item_type
        clean_name = item_name.split('(')[0].strip()
        
        # 2. 아이템 정보 로드
        if clean_name in self.item_db:
            item_info = self.item_db[clean_name]
            target_id = item_info['id']
            grasp_offset = item_info['offset']
        else:
            self.get_logger().warn(f"⚠️ Unknown Item: {clean_name}, using default.")
            target_id = 0
            grasp_offset = [0.0, 0.0, 0.0]

        # 3. 좌표 및 접근 방향 미리 계산 (중간 단계 실행 시에도 필요함)
        pickup_pose, pickup_side = self.get_docking_pose(request.pickup_loc)
        dropoff_pose, dropoff_side = self.get_docking_pose(request.dropoff_loc)
        
        self.get_logger().info(f"🚀 TASK START [Mode: {mode}] | Item: {clean_name}")

        try:
            # =================================================
            # [STEP 1] 픽업지 이동 (NAV_PICKUP)
            # =================================================
            if mode in ["ALL", "NAV_PICKUP"]:
                feedback.current_state = "NAVIGATING TO PICKUP"
                goal_handle.publish_feedback(feedback)
                
                if not pickup_pose: raise Exception("Invalid Pickup Location")
                self.get_logger().info(f"🚗 Navigating to {request.pickup_loc}...")
                
                if not await self.call_nav2(pickup_pose, goal_handle):
                    raise Exception("Navigation to Pickup Failed")
                
                # 부분 실행이면 여기서 종료
                if mode != "ALL": 
                    goal_handle.succeed()
                    result.success = True
                    result.message = "Step 'NAV_PICKUP' Completed"
                    return result

            if self.check_cancel(goal_handle, result): return result

            # =================================================
            # [STEP 2] 픽업지 도킹 (DOCK_PICKUP)
            # =================================================
            if mode in ["ALL", "DOCK_PICKUP"]:
                feedback.current_state = "DOCKING AT PICKUP"
                goal_handle.publish_feedback(feedback)
                
                self.get_logger().info("⚓ Starting Precision Docking (Pickup)...")
                if not await self.call_docking(goal_handle):
                    raise Exception("Docking Failed")
                
                if mode != "ALL":
                    goal_handle.succeed()
                    result.success = True
                    result.message = "Step 'DOCK_PICKUP' Completed"
                    return result

            if self.check_cancel(goal_handle, result): return result

            # =================================================
            # [STEP 3] 물체 인식 및 파지 (PICK)
            # =================================================
            if mode in ["ALL", "PICK"]:
                feedback.current_state = "SCANNING & PICKING"
                goal_handle.publish_feedback(feedback)
                
                # 접근 방향의 반대쪽 카메라 선택 (기존 로직 유지)
                camera_side = "Right" if pickup_side == "Left" else "Left"
                self.get_logger().info(f"👀 Approach: {pickup_side} -> Using Camera: {camera_side}")

                marker_raw_pose = await self.wait_for_marker(target_id, camera_side)
                
                if marker_raw_pose:
                    self.get_logger().info(f"🔎 Applying Offset {grasp_offset}")
                    # 1. 마커 위치 오프셋 계산 (위치만 계산)
                    final_grasp_pose = self.apply_grasp_offset(marker_raw_pose, grasp_offset)

                    # 2. [수정] 접근 방향에 따라 그립 오리엔테이션 분기 적용
                    # pickup_side는 get_docking_pose()에서 반환된 값 ("Left" or "Right")
                    if pickup_side == "Left":
                        # 로봇 기준 왼쪽에 있는 테이블 -> Left 전용 쿼터니언 사용
                        final_grasp_pose.orientation = self.grasp_quat_right
                        self.get_logger().info("🧭 Applying LEFT Grasp Orientation")
                    else:
                        # 로봇 기준 오른쪽에 있는 테이블 -> Right 전용 쿼터니언 사용
                        final_grasp_pose.orientation = self.grasp_quat_left
                        self.get_logger().info("🧭 Applying RIGHT Grasp Orientation")

                    self.get_logger().info("🦾 Sending PICK Command...")
                    if not await self.call_arm('pick', goal_handle, final_grasp_pose):
                        raise Exception("Pick Action Failed")
                    
                    # 테이블 픽업 후 로봇 적재함에 싣기 (Stow)
                    self.get_logger().info("📦 Stowing Item to Cargo Area...")
                    
                    stow_pose = PoseStamped()
                    stow_pose.header.frame_id = "base_link"
                    stow_pose.pose.position.x = -0.5
                    stow_pose.pose.position.y = 0.0
                    stow_pose.pose.position.z = 0.72
                    # 요청한 Quaternion: x: -0.5, y: 0.5, z: 0.5, w: 0.5
                    stow_pose.pose.orientation = Quaternion(x=-0.5, y=0.5, z=0.5, w=0.5)

                    if not await self.call_arm('place', goal_handle, stow_pose.pose):
                        raise Exception("Stowing Action (Place to Cargo) Failed")
                
                self.get_logger().info("💤 Turning OFF Camera after PICK phase")
                self.set_vision(camera_side, False)

            # =================================================
            # [STEP 4] 하역지 이동 (NAV_DROPOFF)
            # =================================================
            if mode in ["ALL", "NAV_DROPOFF"]:
                # 팔 접기 (안전)
                await self.call_arm('home', goal_handle)
                
                feedback.current_state = "NAVIGATING TO DROPOFF"
                goal_handle.publish_feedback(feedback)
                
                self.get_logger().info(f"🚗 Navigating to {request.dropoff_loc}...")
                if not await self.call_nav2(dropoff_pose, goal_handle):
                    raise Exception("Navigation to Dropoff Failed")
                
                if mode != "ALL":
                    goal_handle.succeed()
                    result.success = True
                    result.message = "Step 'NAV_DROPOFF' Completed"
                    return result
            
            if self.check_cancel(goal_handle, result): return result

            # =================================================
            # [STEP 5] 하역지 도킹 (DOCK_DROPOFF)
            # =================================================
            if mode in ["ALL", "DOCK_DROPOFF"]:
                feedback.current_state = "DOCKING AT DROPOFF"
                goal_handle.publish_feedback(feedback)
                
                self.get_logger().info("⚓ Docking at Drop-off...")
                if not await self.call_docking(goal_handle):
                    raise Exception("Docking at Drop-off Failed")

                if mode != "ALL":
                    goal_handle.succeed()
                    result.success = True
                    result.message = "Step 'DOCK_DROPOFF' Completed"
                    return result

            if self.check_cancel(goal_handle, result): return result

            # =================================================
            # [STEP 6] 내려놓기 (PLACE)
            # =================================================
            if mode in ["ALL", "PLACE"]:
                # 적재함에서 물건 다시 집기 (Retrieve)
                self.get_logger().info("📦 Retrieving Item from Cargo Area...")
                
                retrieve_pose = PoseStamped()
                retrieve_pose.header.frame_id = "base_link"
                retrieve_pose.pose.position.x = -0.5
                retrieve_pose.pose.position.y = 0.0
                retrieve_pose.pose.position.z = 0.7
                # 요청한 Quaternion: x: -0.5, y: 0.5, z: 0.5, w: 0.5
                retrieve_pose.pose.orientation = Quaternion(x=-0.5, y=0.5, z=0.5, w=0.5)

                if not await self.call_arm('pick', goal_handle, retrieve_pose.pose):
                    raise Exception("Retrieving Action (Pick from Cargo) Failed")
                
                feedback.current_state = "PLACING"
                goal_handle.publish_feedback(feedback)
                
                # 1. 고정 좌표 설정 (Base Link 기준)
                place_pose = PoseStamped()
                place_pose.header.frame_id = "base_link"
                place_pose.pose.position.x = -0.16
                place_pose.pose.position.z = 1.0

                # 2. 접근 방향(dropoff_side)에 따른 Y좌표 및 오리엔테이션 분기
                # (현재 위치가 하역장이므로 pickup_side가 아닌 dropoff_side를 사용)
                if dropoff_side == "Left":
                    place_pose.pose.position.y = -0.8
                    place_pose.pose.orientation = self.grasp_quat_right
                    self.get_logger().info("🧭 PLACING: Left Approach -> Right Quat, Y=-0.8")
                else: # Right
                    place_pose.pose.position.y = 0.8
                    place_pose.pose.orientation = self.grasp_quat_left
                    self.get_logger().info("🧭 PLACING: Right Approach -> Left Quat, Y=+0.8")

                # 3. 마커 인식 없이 바로 Place 명령 전송
                self.get_logger().info(f"🦾 Sending FIXED PLACE Command... (y={place_pose.pose.position.y})")
                if not await self.call_arm('place', goal_handle, place_pose.pose):
                    raise Exception("Place Action Failed")
                
                # 끝나면 팔 접기
                await self.call_arm('home', goal_handle)
                
                # [추가] PLACE 페이즈 완료 후 끄기
                self.get_logger().info("💤 Turning OFF Camera after PLACE phase")
                self.set_vision(drop_camera_side, False)

                if mode != "ALL":
                    goal_handle.succeed()
                    result.success = True
                    result.message = "Step 'PLACE' Completed"
                    return result

            # =================================================
            # [STEP 7] 홈 위치 복귀 (HOME) - 유틸리티
            # =================================================
            if mode == "HOME":
                self.get_logger().info("🏠 Moving Arm to HOME...")
                await self.call_arm('home', goal_handle)
                goal_handle.succeed()
                result.success = True
                result.message = "Arm Homed"
                return result

            # 여기까지 오면 ALL 모드의 전체 완료
            self.get_logger().info("✅ Full Delivery Sequence Complete!")
            result.success = True
            result.message = "All tasks finished."
            goal_handle.succeed()

        except Exception as e:
            self.get_logger().error(f"🛑 Task Aborted: {str(e)}")
            result.success = False
            result.message = str(e)
            goal_handle.abort()

        return result

def main(args=None):
    rclpy.init(args=args)
    node = HospitalOrchestrator()
    
    # 멀티스레드 실행 (액션 서버와 클라이언트 동시 동작 위함)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()