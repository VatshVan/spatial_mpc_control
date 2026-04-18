import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import threading
import os

from spatial_mpc_control.model import SpatialMPCOrchestrator

# --- 1. The Explicit MPC (Affine Field) Network ---
class ExplicitMPCPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.net(x)

# --- 2. The ROS 2 Node ---
class SpatialMPCNode(Node):
    def __init__(self):
        super().__init__('spatial_mpc_node')

        self.declare_parameter('dt', 0.01)
        self.declare_parameter('mpc_horizon', 40)
        self.declare_parameter('mass_nominal', 12.0)

        dt = self.get_parameter('dt').value
        N = self.get_parameter('mpc_horizon').value
        m_nom = self.get_parameter('mass_nominal').value

        self.pos_arr = [(0.5, 0.4), (0.5, -0.4), (-0.5, 0.4), (-0.5, -0.4)]
        k_arr = [150.0, 150.0, 150.0, 150.0]
        c_arr = [60.0, 60.0, 60.0, 60.0]

        Q = sp.csc_matrix(np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0]))
        R = sp.csc_matrix(np.diag([5000.0, 5000.0, 5000.0, 5000.0]))
        u_min = [-300.0, -300.0, -300.0, -300.0]
        u_max = [300.0, 300.0, 300.0, 300.0]

        # The Heavy "Oracle" Solver (Runs on CPU via background thread)
        self.orchestrator = SpatialMPCOrchestrator(
            m_init=m_nom, Ixx=2.5, Iyy=3.2,
            k_arr=k_arr, c_arr=c_arr, pos_arr=self.pos_arr,
            N=N, Q=Q, R=R, u_min=u_min, u_max=u_max, dt=dt
        )

        # --- CUDA & NEURAL NETWORK SETUP ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"🚀 Neural Network accelerating via: {self.device.type.upper()}")
        
        self.affine_policy = ExplicitMPCPolicy().to(self.device)
        
        # Load weights (Make sure 'affine_policy.pth' is in your workspace root, or provide full path)
        # Fallback to current directory for standard ROS 2 launches
        weights_path = r"/home/vatshvan/ros2_ws/src/spatial_mpc_control/spatial_mpc_control/affine_policy.pth"
        if os.path.exists(weights_path):
            self.affine_policy.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.get_logger().info("✅ Explicit MPC weights loaded successfully.")
        else:
            self.get_logger().error(f"❌ Could not find {weights_path}! Fast-path will output garbage.")
            
        self.affine_policy.eval()

        # State Variables
        self.z = 0.0
        self.phi = 0.0
        self.theta = 0.0
        self.z_dot = 0.0
        self.phi_dot = 0.0
        self.theta_dot = 0.0

        # Async Variables
        self.latest_u_opt = np.zeros(4)
        self.mpc_is_running = False

        self.sub_imu = self.create_subscription(Imu, '/platform/imu', self.imu_callback, 10)
        self.sub_odom = self.create_subscription(Odometry, '/platform/odom', self.odom_callback, 10)
        self.pub_wrench = self.create_publisher(Wrench, '/platform/cmd_wrench', 10)

        self.timer = self.create_timer(dt, self.control_loop)

    def imu_callback(self, msg):
        q = msg.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.phi = np.arctan2(siny_cosp, cosy_cosp)
        sinp = 2 * (q.w * q.y - q.z * q.x)
        self.theta = np.arcsin(np.clip(sinp, -1.0, 1.0))
        self.phi_dot = msg.angular_velocity.x
        self.theta_dot = msg.angular_velocity.y

    def odom_callback(self, msg):
        self.z = msg.pose.pose.position.z
        self.z_dot = msg.twist.twist.linear.z

    # Heavy OSQP Solver Thread (Always runs on CPU)
    def solve_mpc_thread(self, z_meas_cpu):
        self.mpc_is_running = True
        try:
            u_opt = self.orchestrator.dispatch_control(z_meas_cpu).detach().numpy()
            if not np.any(np.isnan(u_opt)):
                self.latest_u_opt = u_opt
                self.get_logger().info("🔧 OSQP Oracle updated safe trajectory.")
        except Exception:
            pass
        self.mpc_is_running = False

    def control_loop(self):
        z_state = self.z - 0.5 
        
        if np.isnan(z_state) or np.isnan(self.z_dot):
            return
            
        # Push state directly to CUDA for lightning-fast inference
        z_meas_gpu = torch.tensor(
            [z_state, self.phi, self.theta, self.z_dot, self.phi_dot, self.theta_dot], 
            dtype=torch.float32, device=self.device
        )

        # --- HYBRID CONTROL LOGIC ---
        # Trust Region: Max 11 degrees tilt, Max 0.2m displacement
        is_safe = (abs(self.phi) < 0.2 and abs(self.theta) < 0.2 and abs(z_state) < 0.2)

        if is_safe:
            # FAST PATH: GPU Explicit MPC Inference
            with torch.no_grad():
                self.latest_u_opt = self.affine_policy(z_meas_gpu).cpu().numpy()
        else:
            # FALLBACK PATH: Severe Disturbance! 
            # GPU holds immediate stability...
            with torch.no_grad():
                self.latest_u_opt = self.affine_policy(z_meas_gpu).cpu().numpy()
            
            # ...while CPU recalculates the optimal 1.0s trajectory.
            if not self.mpc_is_running:
                self.get_logger().warn("⚠️ Disturbance outside Trust Region! Engaging OSQP Oracle...")
                z_meas_cpu = z_meas_gpu.cpu() # Send a copy back to CPU for SciPy
                threading.Thread(target=self.solve_mpc_thread, args=(z_meas_cpu,)).start()

        # --- VIRTUAL IMPEDANCE (Runs at 100Hz on CPU) ---
        k_stiffness = 150.0
        c_damping = 60.0
        
        f_passive = np.zeros(4)
        for i in range(4):
            x_i, y_i = self.pos_arr[i]
            z_corner = z_state + (y_i * self.phi) - (x_i * self.theta)
            v_corner = self.z_dot + (y_i * self.phi_dot) - (x_i * self.theta_dot)
            f_passive[i] = -(k_stiffness * z_corner) - (c_damping * v_corner)

        # Combine GPU-predicted active forces with immediate passive physics
        total_corner_forces = self.latest_u_opt + f_passive
        total_corner_forces = np.clip(total_corner_forces, -300.0, 300.0)

        # Publish to Gazebo
        force_z = float(np.sum(total_corner_forces)) + (12.0 * 9.81) 
        torque_x = float(sum(total_corner_forces[i] * self.pos_arr[i][1] for i in range(4)))
        torque_y = float(sum(total_corner_forces[i] * -self.pos_arr[i][0] for i in range(4)))

        w = Wrench()
        w.force.z = float(np.clip(force_z, -1500.0, 1500.0))
        w.torque.x = float(np.clip(torque_x, -500.0, 500.0))
        w.torque.y = float(np.clip(torque_y, -500.0, 500.0))
        
        self.pub_wrench.publish(w)

def main(args=None):
    rclpy.init(args=args)
    node = SpatialMPCNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()