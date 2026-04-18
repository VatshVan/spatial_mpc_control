import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def plot_latest_telemetry(data_dir='data'):
    # Find the most recently created CSV file in the data directory
    search_pattern = os.path.join(data_dir, 'gazebo_telemetry_*.csv')
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"No telemetry CSV files found in '{data_dir}/'. Have you run the logger yet?")
        return

    latest_file = max(files, key=os.path.getctime)
    print(f"Loading data from: {latest_file}")

    # Load the data
    df = pd.read_csv(latest_file)
    time = df['time_sec']

    # Create a figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f'Platform Telemetry Analysis\n{os.path.basename(latest_file)}', fontsize=16)

    # --- Plot 1: Spatial Displacements ---
    axs[0].plot(time, df['z'], label='Heave (z) [m]', color='#2ecc71', linewidth=2)
    axs[0].plot(time, df['phi'], label='Roll (phi) [rad]', color='#e74c3c', linewidth=2)
    axs[0].plot(time, df['theta'], label='Pitch (theta) [rad]', color='#3498db', linewidth=2)
    
    # Add reference lines for the target state (assuming 0.5m heave, 0 roll, 0 pitch)
    axs[0].axhline(0.5, color='#2ecc71', linestyle='--', alpha=0.5)
    axs[0].axhline(0.0, color='gray', linestyle='--', alpha=0.5)
    
    axs[0].set_ylabel('Displacement')
    axs[0].set_title('Platform Kinematics (Position & Orientation)')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, linestyle=':', alpha=0.7)

    # --- Plot 2: Velocities ---
    axs[1].plot(time, df['z_dot'], label='z_dot [m/s]', color='#2ecc71', alpha=0.8)
    axs[1].plot(time, df['phi_dot'], label='phi_dot [rad/s]', color='#e74c3c', alpha=0.8)
    axs[1].plot(time, df['theta_dot'], label='theta_dot [rad/s]', color='#3498db', alpha=0.8)
    
    axs[1].set_ylabel('Velocity')
    axs[1].set_title('Platform Dynamics (Velocities)')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, linestyle=':', alpha=0.7)

    # --- Plot 3: Commanded Wrench (MPC Output) ---
    axs[2].plot(time, df['cmd_force_z'], label='Net Force Z [N]', color='#9b59b6', linewidth=2)
    
    # Create a secondary y-axis for torques since they are usually on a smaller scale than forces
    ax2_twin = axs[2].twinx()
    ax2_twin.plot(time, df['cmd_torque_x'], label='Torque X (Roll) [Nm]', color='#f39c12', linestyle='--')
    ax2_twin.plot(time, df['cmd_torque_y'], label='Torque Y (Pitch) [Nm]', color='#d35400', linestyle='-.')
    
    axs[2].set_ylabel('Force [N]', color='#9b59b6')
    ax2_twin.set_ylabel('Torque [Nm]', color='#d35400')
    axs[2].set_xlabel('Time [seconds]')
    axs[2].set_title('MPC Commanded Wrench')
    
    # Combine legends from both axes in the third subplot
    lines_1, labels_1 = axs[2].get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    ax2_twin.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    
    axs[2].grid(True, linestyle=':', alpha=0.7)

    # Final Layout Adjustments
    plt.tight_layout()
    
    # Save the plot automatically
    output_img = latest_file.replace('.csv', '.png')
    plt.savefig(output_img, dpi=300)
    print(f"Plot saved successfully as: {output_img}")
    
    # Display the plot
    plt.show()

if __name__ == '__main__':
    # Ensure pandas and matplotlib are installed
    try:
        plot_latest_telemetry()
    except ModuleNotFoundError as e:
        print(f"Error: {e}")
        print("Please install required packages using: pip install pandas matplotlib")