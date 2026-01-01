import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    pkg_dir = get_package_share_directory('my_pkg')
    config_file = os.path.join(pkg_dir, 'config', 'nova_docking.yaml')
    
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # 1. AprilTag 인식 노드
    apriltag_node = ComposableNodeContainer(
        package='rclcpp_components',
        name='apriltag_container',
        namespace='',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='apriltag_ros',
                plugin='AprilTagNode',
                name='apriltag',
                remappings=[
                    ('image_rect', '/front_camera/rgb'),
                    ('camera_info', '/front_camera/camera_info')
                ],
                parameters=[{'size': 0.25, 'family': '36h11'}],
                extra_arguments=[{'use_intra_process_comms': True}],
            )
        ],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # 2. 도킹 포즈 퍼블리셔 (Camera frame 출력)
    dock_pose_publisher = Node(
        package='my_pkg',
        executable='dock_pose_publisher',
        name='dock_pose_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # 3. 도킹 서버 (YAML에서 좌표 변환 처리)
    docking_server = Node(
        package='opennav_docking',
        executable='opennav_docking',
        name='docking_server',
        output='screen',
        parameters=[config_file, {'use_sim_time': use_sim_time}], 
        remappings=[('cmd_vel', '/cmd_vel')]
    )

    # 4. Lifecycle Manager
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_docking',
        output='screen',
        parameters=[
            {'autostart': True}, 
            {'node_names': ['docking_server']}, 
            {'use_sim_time': use_sim_time}
        ],
    )

    # 5. TF (map → odom)
    tf_map_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments = [
            '--x', '0', '--y', '0', '--z', '0', 
            '--yaw', '0', '--pitch', '0', '--roll', '0', 
            '--frame-id', 'map', '--child-frame-id', 'odom'
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    return LaunchDescription([
        apriltag_node,
        dock_pose_publisher,
        docking_server,
        lifecycle_manager,
        tf_map_odom
    ])