import pickle
import matplotlib.pyplot as plt
import numpy as np

def render_telemetry_dashboard(filepath='data/spatial_telemetry.pkl', target_scenario="6. Full Spatial Translation"):
    try:
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Could not locate {filepath}")
        return

    trajectories = payload['trajectory_samples']
    dt = payload['dt']
    
    if target_scenario not in trajectories:
        print(f"Scenario '{target_scenario}' not found.")
        return

    data = trajectories[target_scenario]
    X = data['X']
    U = data['U']
    M_track = data['M_track']
    target = data['Target']
    
    steps = X.shape[0]
    time = np.arange(steps) * dt

    fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(f'Spatial HyperMPC Telemetry: {target_scenario}', fontsize=16)

    # 1. Spatial Displacement (Heave, Roll, Pitch)
    axs[0].plot(time, X[:, 0], label='Heave (z) [m]', color='#2ecc71', linewidth=2)
    axs[0].plot(time, X[:, 1], label='Roll (phi) [rad]', color='#e74c3c', linewidth=2)
    axs[0].plot(time, X[:, 2], label='Pitch (theta) [rad]', color='#3498db', linewidth=2)
    axs[0].axhline(target[0], color='#2ecc71', linestyle='--', alpha=0.5)
    axs[0].axhline(target[1], color='#e74c3c', linestyle='--', alpha=0.5)
    axs[0].axhline(target[2], color='#3498db', linestyle='--', alpha=0.5)
    axs[0].set_ylabel('Displacement')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, linestyle=':', alpha=0.7)

    # 2. Spatial Velocities
    axs[1].plot(time, X[:, 3], label='z_dot', color='#2ecc71', alpha=0.7)
    axs[1].plot(time, X[:, 4], label='phi_dot', color='#e74c3c', alpha=0.7)
    axs[1].plot(time, X[:, 5], label='theta_dot', color='#3498db', alpha=0.7)
    axs[1].set_ylabel('Velocities')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, linestyle=':', alpha=0.7)

    # 3. Actuator Control Effort (U)
    axs[2].plot(time, U[:, 0], label='F1 (FL)', linestyle='-')
    axs[2].plot(time, U[:, 1], label='F2 (FR)', linestyle='--')
    axs[2].plot(time, U[:, 2], label='F3 (RL)', linestyle='-.')
    axs[2].plot(time, U[:, 3], label='F4 (RR)', linestyle=':')
    axs[2].set_ylabel('Actuator Force [N]')
    axs[2].legend(loc='upper right', ncol=4)
    axs[2].grid(True, linestyle=':', alpha=0.7)

    # 4. Joint EKF Mass Tracking
    axs[3].plot(time, M_track[:, 0], label='True Plant Mass', color='black', linewidth=2)
    axs[3].plot(time, M_track[:, 1], label='EKF Estimated Mass', color='#9b59b6', linestyle='--', linewidth=2)
    axs[3].set_ylabel('Mass [kg]')
    axs[3].set_xlabel('Time [seconds]')
    axs[3].legend(loc='upper right')
    axs[3].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig('spatial_dashboard.png', dpi=300)
    print("Render complete. Dashboard saved as 'spatial_dashboard.png'.")
    plt.show()

if __name__ == "__main__":
    render_telemetry_dashboard()