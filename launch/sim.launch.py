import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    pkg_path = get_package_share_directory('spatial_mpc_control')
    urdf_file = os.path.join(pkg_path, 'urdf', 'spatial_platform.urdf.xacro')
    
    doc = xacro.process_file(urdf_file)
    robot_description = {'robot_description': doc.toxml()}

    # Gazebo Server & Client
    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    # Spawn Entity
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'spatial_platform', '-z', '0.5'],
        output='screen'
    )

    # Robot State Publisher
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    # HyperMPC Node
    mpc_node = Node(
        package='spatial_mpc_control',
        executable='spatial_mpc_node',
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        rsp,
        spawn_entity,
        mpc_node
    ])
