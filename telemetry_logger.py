import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import numpy as np
import pandas as pd
import os
import time

class TelemetryLoggerNode(Node):
    def __init__(self):
        super().__init__('telemetry_logger_node')

        # Logging rate (Hz). 100Hz matches your MPC dt=0.01
        self.declare_parameter('log_rate', 100.0)
        rate = self.get_parameter('log_rate').value

        # Data holding registers (latest values)
        self.latest_data = {
            'time_sec': 0.0,
            'z': 0.0,
            'phi': 0.0,
            'theta': 0.0,
            'z_dot': 0.0,
            'phi_dot': 0.0,
            'theta_dot': 0.0,
            'cmd_force_z': 0.0,
            'cmd_torque_x': 0.0,
            'cmd_torque_y': 0.0
        }

        self.history = []
        self.start_time = time.time()

        # Subscribers
        self.sub_imu = self.create_subscription(Imu, '/platform/imu', self.imu_callback, 10)
        self.sub_odom = self.create_subscription(Odometry, '/platform/odom', self.odom_callback, 10)
        self.sub_wrench = self.create_subscription(Wrench, '/platform/cmd_wrench', self.wrench_callback, 10)

        # Logging Timer
        self.timer = self.create_timer(1.0 / rate, self.record_state)
        self.get_logger().info("Telemetry Logger Active. Recording at 100Hz... Press Ctrl+C to save and exit.")

    def imu_callback(self, msg):
        # Extract Roll and Pitch from Quaternion
        q = msg.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.latest_data['phi'] = np.arctan2(siny_cosp, cosy_cosp)

        sinp = 2 * (q.w * q.y - q.z * q.x)
        self.latest_data['theta'] = np.arcsin(np.clip(sinp, -1.0, 1.0))

        # Extract Angular Velocities
        self.latest_data['phi_dot'] = msg.angular_velocity.x
        self.latest_data['theta_dot'] = msg.angular_velocity.y

    def odom_callback(self, msg):
        self.latest_data['z'] = msg.pose.pose.position.z
        self.latest_data['z_dot'] = msg.twist.twist.linear.z

    def wrench_callback(self, msg):
        self.latest_data['cmd_force_z'] = msg.force.z
        self.latest_data['cmd_torque_x'] = msg.torque.x
        self.latest_data['cmd_torque_y'] = msg.torque.y

    def record_state(self):
        # Snapshot the current state with a relative timestamp
        snapshot = self.latest_data.copy()
        snapshot['time_sec'] = time.time() - self.start_time
        self.history.append(snapshot)

    def save_data(self):
        self.get_logger().info("Saving telemetry data...")
        if len(self.history) == 0:
            self.get_logger().warn("No data recorded.")
            return

        df = pd.DataFrame(self.history)
        
        # Ensure 'data' directory exists
        os.makedirs('data', exist_ok=True)
        
        filename = f"data/gazebo_telemetry_{int(time.time())}.csv"
        df.to_csv(filename, index=False)
        self.get_logger().info(f"Successfully saved {len(df)} rows to {filename}")

def main(args=None):
    rclpy.init(args=args)
    logger_node = TelemetryLoggerNode()

    try:
        rclpy.spin(logger_node)
    except KeyboardInterrupt:
        # Gracefully handle Ctrl+C to trigger the save function
        pass
    finally:
        logger_node.save_data()
        logger_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()