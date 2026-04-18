import torch
import numpy as np
import scipy.sparse as sp
import pickle
from spatial_mpc_control.model import SpatialMPCOrchestrator

def generate_data(num_samples=20000):
    print(f"Generating {num_samples} MPC solutions... This might take a minute.")
    
    # 1. Instantiate your MPC exactly as it is in the ROS node
    pos_arr = [(0.5, 0.4), (0.5, -0.4), (-0.5, 0.4), (-0.5, -0.4)]
    Q = sp.csc_matrix(np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0]))
    R = sp.csc_matrix(np.diag([5000.0, 5000.0, 5000.0, 5000.0]))
    
    orchestrator = SpatialMPCOrchestrator(
        m_init=12.0, Ixx=2.5, Iyy=3.2,
        k_arr=[150.0]*4, c_arr=[60.0]*4, pos_arr=pos_arr,
        N=40, Q=Q, R=R, u_min=[-300.0]*4, u_max=[300.0]*4, dt=0.01
    )

    X_data = []
    U_data = []

    for i in range(num_samples):
        # 2. Randomize a state within the "Trust Region"
        # z_err(±0.3m), phi/theta(±0.3rad), velocities(±1.0m/s)
        state = np.array([
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(-1.0, 1.0)
        ])
        
        z_meas = torch.tensor(state, dtype=torch.float32)
        
        # 3. Ask OSQP for the answer
        try:
            u_opt = orchestrator.dispatch_control(z_meas).detach().numpy()
            if not np.any(np.isnan(u_opt)):
                X_data.append(state)
                U_data.append(u_opt)
        except Exception:
            pass # Skip if OSQP solver fails to find a solution

        if i > 0 and i % 2000 == 0:
            print(f"Solved {i}/{num_samples} states...")

    # 4. Save the dataset
    dataset = {'X': np.array(X_data), 'U': np.array(U_data)}
    with open('mpc_expert_data.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Successfully saved {len(X_data)} valid states to mpc_expert_data.pkl")

if __name__ == "__main__":
    generate_data()