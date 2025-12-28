import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from moma_interfaces.action import MoveManipulator

class MasterController(Node):
    def __init__(self):
        super().__init__('master_controller')
        
        # Action Client 생성 (서버 이름: /move_manipulator)
        self._action_client = ActionClient(self, MoveManipulator, '/move_manipulator')
        
        self.get_logger().info("🧠 Master Controller Initialized. Waiting for Action Server...")
        self._action_client.wait_for_server()
        self.get_logger().info("✅ Action Server Connected! Ready to command.")

    def send_goal(self, action_type, pose=None):
        """액션 서버에 명령 전송"""
        goal_msg = MoveManipulator.Goal()
        goal_msg.action_type = action_type
        
        if pose:
            goal_msg.target_pose = pose
        
        self.get_logger().info(f"📤 Sending Goal: {action_type}...")
        
        # 1. 목표 전송
        send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f"❌ Goal {action_type} Rejected!")
            return False

        # 2. 결과 대기
        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)
        
        result = get_result_future.result().result
        status = get_result_future.result().status
        
        if result.success:
            self.get_logger().info(f"🎉 {action_type} Completed: {result.message}")
            return True
        else:
            self.get_logger().error(f"💀 {action_type} Failed: {result.message}")
            return False

    def feedback_callback(self, feedback_msg):
        """실시간 진행 상황 출력"""
        feedback = feedback_msg.feedback
        # 너무 자주 출력되지 않게 하려면 로직 추가 가능
        # self.get_logger().info(f"   Using feedback: {feedback.current_state}")

    def create_pose(self, x, y, z, qx=0.0, qy=0.707, qz=0.0, qw=0.707):
        """PoseStamped 메시지 생성 헬퍼"""
        p = PoseStamped()
        p.header.frame_id = "base_link"
        p.pose.position.x = x
        p.pose.position.y = y
        p.pose.position.z = z
        p.pose.orientation.x = qx
        p.pose.orientation.y = qy
        p.pose.orientation.z = qz
        p.pose.orientation.w = qw
        return p

def main(args=None):
    rclpy.init(args=args)
    master = MasterController()

    try:
        # ====================================================
        # 🧪 [시나리오] Pick & Place 전체 테스트
        # ====================================================
        
        # 1. 좌표 정의 (테스트했던 성공 좌표 사용)
        pick_pose = master.create_pose(x=-0.15, y=0.8, z=0.93)   # 잡기 위치
        place_pose = master.create_pose(x=-0.4, y=-0.0, z=0.80) # 놓기 위치 (반대편)

        # 2. Pick 실행
        master.get_logger().info("\n▶️ [STEP 1] Starting PICK Sequence")
        if not master.send_goal('pick', pick_pose):
            master.get_logger().error("🛑 Pick Failed. Aborting Mission.")
            return # 실패 시 종료

        # 3. Place 실행
        master.get_logger().info("\n▶️ [STEP 2] Starting PLACE Sequence")
        if not master.send_goal('place', place_pose):
            master.get_logger().error("🛑 Place Failed.")
            return

        # 4. Home 복귀 (선택 사항)
        master.get_logger().info("\n▶️ [STEP 3] Returning HOME")
        master.send_goal('home')

        master.get_logger().info("\n✅ Mission Complete!")

    except KeyboardInterrupt:
        pass
    finally:
        master.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()