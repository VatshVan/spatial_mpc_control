import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# ==============================================================================
# 1. THE CONTROLLERS
# ==============================================================================

class ExplicitMPCPolicy(nn.Module):
    """The Neural MPC (Continuous Piecewise Affine Field)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.net(x)

class SpatialPIDController:
    """A benchmark MIMO PID Controller for comparison against the Neural MPC"""
    def __init__(self, kp, ki, kd, dt):
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)
        self.dt = dt
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        
        # Actuator geometry for Wrench allocation
        self.pos_arr = [(0.5, 0.4), (0.5, -0.4), (-0.5, 0.4), (-0.5, -0.4)]
        self.T = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [0.4, -0.4, 0.4, -0.4],   # y positions for Roll torque
            [-0.5, -0.5, 0.5, 0.5]    # -x positions for Pitch torque
        ])
        self.T_pinv = np.linalg.pinv(self.T) # Pseudo-inverse to map Wrench -> Actuators

    def compute(self, state):
        # We only care about positional errors for PID (z, phi, theta)
        # Target is [0, 0, 0] since state is already shifted to hovering origin
        error = np.array([0.0 - state[0], 0.0 - state[1], 0.0 - state[2]])
        
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        
        # Calculate desired restoring Wrench [Force_Z, Torque_X, Torque_Y]
        wrench = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        # Allocate Wrench to the 4 actuators
        u_cmd = self.T_pinv @ wrench
        return np.clip(u_cmd, -300.0, 300.0)


# ==============================================================================
# 2. THE PHYSICS ENGINE (Headless Simulator)
# ==============================================================================

class HeadlessSpatialPlant:
    def __init__(self, m=12.0, Ixx=2.5, Iyy=3.2, dt=0.01):
        self.dt = dt
        self.m = m
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.pos_arr = [(0.5, 0.4), (0.5, -0.4), (-0.5, 0.4), (-0.5, -0.4)]
        self.k_stiffness = 150.0
        self.c_damping = 60.0

    def get_passive_forces(self, state):
        z, phi, theta, z_dot, phi_dot, theta_dot = state
        f_passive = np.zeros(4)
        for i in range(4):
            x_i, y_i = self.pos_arr[i]
            z_corner = z + (y_i * phi) - (x_i * theta)
            v_corner = z_dot + (y_i * phi_dot) - (x_i * theta_dot)
            f_passive[i] = -(self.k_stiffness * z_corner) - (self.c_damping * v_corner)
        return f_passive

    def step(self, state, u_active, ext_wrench=np.zeros(3)):
        """RK4 Integration for rigid body dynamics"""
        def dynamics(x, u):
            z, phi, theta, z_dot, phi_dot, theta_dot = x
            f_pass = self.get_passive_forces(x)
            f_total = u + f_pass
            
            # Sum of forces and torques
            Fz = np.sum(f_total) + ext_wrench[0]
            Tx = sum(f_total[i] * self.pos_arr[i][1] for i in range(4)) + ext_wrench[1]
            Ty = sum(f_total[i] * -self.pos_arr[i][0] for i in range(4)) + ext_wrench[2]
            
            # Accelerations (F = ma)
            z_ddot = Fz / self.m
            phi_ddot = Tx / self.Ixx
            theta_ddot = Ty / self.Iyy
            
            return np.array([z_dot, phi_dot, theta_dot, z_ddot, phi_ddot, theta_ddot])

        # Runge-Kutta 4
        k1 = dynamics(state, u_active)
        k2 = dynamics(state + 0.5 * self.dt * k1, u_active)
        k3 = dynamics(state + 0.5 * self.dt * k2, u_active)
        k4 = dynamics(state + self.dt * k3, u_active)
        
        next_state = state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return next_state


# ==============================================================================
# 3. THE SIMULATION PIPELINE
# ==============================================================================

def run_pipeline():
    dt = 0.01
    sim_time = 3.0 # Simulate 3 seconds per run
    steps = int(sim_time / dt)

    # 1. Load Controllers
    print("Loading Neural MPC Weights...")
    neural_mpc = ExplicitMPCPolicy()
    try:
        neural_mpc.load_state_dict(torch.load('affine_policy.pth', weights_only=True))
    except FileNotFoundError:
        print("WARNING: 'affine_policy.pth' not found. Neural MPC will use random weights!")
    neural_mpc.eval()

    # Highly tuned PID for a 12kg hovering plate
    pid_ctrl = SpatialPIDController(kp=[2000, 1000, 1000], ki=[100, 50, 50], kd=[500, 200, 200], dt=dt)

    controllers = {
        "Neural_MPC": lambda x: neural_mpc(torch.tensor(x, dtype=torch.float32)).detach().numpy(),
        "PID_Control": lambda x: pid_ctrl.compute(x)
    }

    # 2. Define Test Cases
    cases = {
        "Case_1_Nominal_Recovery": {
            "mass": 12.0,
            "initial_state": [0.2, 0.3, -0.2, 0.0, 0.0, 0.0], # Start severely tilted/lifted
            "ext_wrench_func": lambda t: np.zeros(3)
        },
        "Case_2_Heavy_Payload": {
            "mass": 20.0, # 8kg extra payload dropped on it
            "initial_state": [0.0, 0.0, 0.0, -1.0, 0.0, 0.0], # Start falling at 1m/s
            "ext_wrench_func": lambda t: np.zeros(3)
        },
        "Case_3_Continuous_Wind_Torque": {
            "mass": 12.0,
            "initial_state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            # Constant 50Nm twisting force applied after 0.5s
            "ext_wrench_func": lambda t: np.array([0.0, 50.0, -30.0]) if t > 0.5 else np.zeros(3) 
        }
    }

    # 3. Setup Storage Directory
    timestamp = int(time.time())
    data_dir = f"sim_results_{timestamp}"
    os.makedirs(data_dir, exist_ok=True)
    print(f"\nCreated data directory: {data_dir}/\n")

    # 4. Run the Pipeline
    for case_name, params in cases.items():
        print(f"--- Simulating {case_name} ---")
        
        for ctrl_name, get_u in controllers.items():
            # Reset Plant and Controller state
            plant = HeadlessSpatialPlant(m=params['mass'], dt=dt)
            state = np.array(params['initial_state'])
            if ctrl_name == "PID_Control":
                pid_ctrl.integral = np.zeros(3) 
                
            history = []

            for k in range(steps):
                t = k * dt
                
                # Inference
                u_opt = get_u(state)
                u_opt = np.clip(u_opt, -300.0, 300.0)
                
                # External Disturbances
                ext_w = params['ext_wrench_func'](t)
                
                # Log Data
                history.append({
                    "time": t,
                    "z": state[0], "phi": state[1], "theta": state[2],
                    "z_dot": state[3], "phi_dot": state[4], "theta_dot": state[5],
                    "u1_fl": u_opt[0], "u2_fr": u_opt[1], "u3_rl": u_opt[2], "u4_rr": u_opt[3]
                })

                # Physics Step
                state = plant.step(state, u_opt, ext_wrench=ext_w)

            # Save to CSV
            df = pd.DataFrame(history)
            filename = f"{data_dir}/{case_name}_{ctrl_name}.csv"
            df.to_csv(filename, index=False)
            print(f"Saved: {filename}")

    print("\n✅ Pipeline complete. All data stored successfully.")

if __name__ == "__main__":
    run_pipeline()