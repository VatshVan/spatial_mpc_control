import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetic style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'lines.linewidth': 2})

def get_latest_sim_dir():
    """Finds the most recently created sim_results directory."""
    dirs = glob.glob("sim_results_*")
    if not dirs:
        raise FileNotFoundError("No 'sim_results_*' directories found. Run the pipeline first.")
    latest_dir = max(dirs, key=os.path.getmtime)
    print(f"Loading data from: {latest_dir}")
    return latest_dir

def calculate_metrics(df, dt=0.01):
    """Calculates ITAE (Integral Time Absolute Error) and Actuator Effort."""
    time = df['time']
    # ITAE: Sum of (time * absolute_error)
    itae_z = np.sum(time * np.abs(df['z'] - 0.0)) * dt
    itae_phi = np.sum(time * np.abs(df['phi'] - 0.0)) * dt
    itae_theta = np.sum(time * np.abs(df['theta'] - 0.0)) * dt
    total_itae = itae_z + itae_phi + itae_theta

    # Effort: Sum of absolute forces
    effort = np.sum(np.abs(df['u1_fl']) + np.abs(df['u2_fr']) + 
                    np.abs(df['u3_rl']) + np.abs(df['u4_rr'])) * dt
    
    return total_itae, effort

def generate_plots():
    data_dir = get_latest_sim_dir()
    plot_dir = os.path.join(data_dir, "analysis_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    cases = ["Case_1_Nominal_Recovery", "Case_2_Heavy_Payload", "Case_3_Continuous_Wind_Torque"]
    controllers = ["Neural_MPC", "PID_Control"]
    
    metrics_data = []

    for case in cases:
        print(f"Generating plots for {case}...")
        
        # Load data for both controllers
        df_mpc = pd.read_csv(f"{data_dir}/{case}_Neural_MPC.csv")
        df_pid = pd.read_csv(f"{data_dir}/{case}_PID_Control.csv")
        
        time = df_mpc['time']
        
        # Calculate Metrics
        itae_mpc, eff_mpc = calculate_metrics(df_mpc)
        itae_pid, eff_pid = calculate_metrics(df_pid)
        metrics_data.extend([
            {"Case": case.replace("Case_", "").replace("_", " "), "Controller": "Neural MPC", "ITAE": itae_mpc, "Effort": eff_mpc},
            {"Case": case.replace("Case_", "").replace("_", " "), "Controller": "PID", "ITAE": itae_pid, "Effort": eff_pid}
        ])

        # ======================================================
        # PLOT 1: Displacements (z, phi, theta)
        # ======================================================
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'{case}: Spatial Displacements\nNeural MPC vs PID', fontsize=16, fontweight='bold')
        
        labels = ['Heave (z) [m]', 'Roll (phi) [rad]', 'Pitch (theta) [rad]']
        cols = ['z', 'phi', 'theta']
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        
        for ax, col, label, color in zip(axs, cols, labels, colors):
            ax.plot(time, df_mpc[col], label='Neural MPC', color=color)
            ax.plot(time, df_pid[col], label='PID', color=color, linestyle='--', alpha=0.6)
            ax.axhline(0, color='black', linewidth=1, linestyle=':')
            ax.set_ylabel(label)
            ax.legend(loc='upper right')
        
        axs[-1].set_xlabel('Time [s]')
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{case}_1_Displacements.png", dpi=300)
        plt.close()

        # ======================================================
        # PLOT 2: Velocities (z_dot, phi_dot, theta_dot)
        # ======================================================
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'{case}: Spatial Velocities\nNeural MPC vs PID', fontsize=16, fontweight='bold')
        
        labels = ['Heave Vel (z_dot) [m/s]', 'Roll Vel (phi_dot) [rad/s]', 'Pitch Vel (theta_dot) [rad/s]']
        cols = ['z_dot', 'phi_dot', 'theta_dot']
        
        for ax, col, label, color in zip(axs, cols, labels, colors):
            ax.plot(time, df_mpc[col], label='Neural MPC', color=color)
            ax.plot(time, df_pid[col], label='PID', color=color, linestyle='--', alpha=0.6)
            ax.axhline(0, color='black', linewidth=1, linestyle=':')
            ax.set_ylabel(label)
            ax.legend(loc='upper right')
            
        axs[-1].set_xlabel('Time [s]')
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{case}_2_Velocities.png", dpi=300)
        plt.close()

        # ======================================================
        # PLOT 3: Actuator Control Effort (The 4 corners)
        # ======================================================
        fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        fig.suptitle(f'{case}: Actuator Control Forces\nNeural MPC vs PID (Bounds: ±300N)', fontsize=16, fontweight='bold')
        
        labels = ['Front-Left [N]', 'Front-Right [N]', 'Rear-Left [N]', 'Rear-Right [N]']
        cols = ['u1_fl', 'u2_fr', 'u3_rl', 'u4_rr']
        
        for ax, col, label in zip(axs, cols, labels):
            ax.plot(time, df_mpc[col], label='Neural MPC', color='#9b59b6')
            ax.plot(time, df_pid[col], label='PID', color='#f39c12', linestyle='--', alpha=0.7)
            # Draw physical saturation limits
            ax.axhline(300, color='red', linestyle=':', alpha=0.5)
            ax.axhline(-300, color='red', linestyle=':', alpha=0.5)
            ax.set_ylabel(label)
            ax.legend(loc='upper right')
            
        axs[-1].set_xlabel('Time [s]')
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{case}_3_Actuators.png", dpi=300)
        plt.close()

        # ======================================================
        # PLOT 4: Phase Portraits (State vs Velocity)
        # ======================================================
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{case}: Phase Portraits (Convergence to Origin)', fontsize=16, fontweight='bold')
        
        vars_pairs = [('z', 'z_dot', 'Heave Phase'), 
                      ('phi', 'phi_dot', 'Roll Phase'), 
                      ('theta', 'theta_dot', 'Pitch Phase')]
        
        for ax, (pos, vel, title) in zip(axs, vars_pairs):
            # Plot MPC Trajectory
            ax.plot(df_mpc[pos], df_mpc[vel], label='Neural MPC', color='#2980b9')
            # Plot PID Trajectory
            ax.plot(df_pid[pos], df_pid[vel], label='PID', color='#d35400', linestyle='--', alpha=0.7)
            
            # Mark Start (O) and End (X)
            ax.scatter(df_mpc[pos].iloc[0], df_mpc[vel].iloc[0], color='green', marker='o', s=100, label='Start')
            ax.scatter(0, 0, color='red', marker='X', s=100, label='Origin (Target)')
            
            ax.set_title(title)
            ax.set_xlabel(f'{pos} (Position)')
            ax.set_ylabel(f'{vel} (Velocity)')
            ax.grid(True, linestyle='--')
            if pos == 'z': ax.legend()
            
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{case}_4_PhasePortraits.png", dpi=300)
        plt.close()

    # ======================================================
    # PLOT 5: Global Performance Dashboard (Bar Charts)
    # ======================================================
    print("Generating Global Dashboard...")
    df_metrics = pd.DataFrame(metrics_data)
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Global Performance Benchmarks: Neural MPC vs PID', fontsize=18, fontweight='bold')
    
    # ITAE Chart (Lower is better)
    sns.barplot(data=df_metrics, x='Case', y='ITAE', hue='Controller', ax=axs[0], palette=['#9b59b6', '#f39c12'])
    axs[0].set_title('Cumulative Tracking Error (ITAE)\nLower = Better tracking over time')
    axs[0].set_ylabel('ITAE Score')
    axs[0].tick_params(axis='x', rotation=15)
    
    # Control Effort Chart (Lower is better)
    sns.barplot(data=df_metrics, x='Case', y='Effort', hue='Controller', ax=axs[1], palette=['#9b59b6', '#f39c12'])
    axs[1].set_title('Total Actuator Energy Expended\nLower = More efficient / Less aggressive')
    axs[1].set_ylabel('Sum of Force over Time [N*s]')
    axs[1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/Global_Performance_Dashboard.png", dpi=300)
    plt.close()

    print(f"\n✅ All 13 analysis plots successfully saved to: {plot_dir}/")

if __name__ == "__main__":
    generate_plots()