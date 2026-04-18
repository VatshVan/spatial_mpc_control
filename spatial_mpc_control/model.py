import torch
import numpy as np
import scipy.sparse as sp
import pandas as pd
from spatial_mpc_control.MPC import MIMO_HyperRTIMPC
from spatial_mpc_control.State_space import SpatialHybridDynamics, AlphaConfig, HyperResidualNet

class MIMO_AdversarialPlant:
    def __init__(self, m_nom, Ixx_nom, Iyy_nom, k_arr, c_arr, pos_arr, dt, noise_cov, param_drift_rate):
        self.dt = dt
        self.Q_v = noise_cov
        self.drift_rate = param_drift_rate
        
        self.m_true = m_nom
        self.Ixx_true = Ixx_nom
        self.Iyy_true = Iyy_nom
        
        self.k_nom = k_arr
        self.c_nom = c_arr
        self.k_true = list(k_arr)
        self.c_true = list(c_arr)
        
        T_list = [[1.0, float(y_i), -float(x_i)] for x_i, y_i in pos_arr]
        self.T = torch.tensor(T_list, dtype=torch.float32).T
        
        self.A_true, self.B_true = self._update_true_matrices()

    def _update_true_matrices(self):
        self.m_true += np.random.normal(0, self.drift_rate)
        self.k_true = [max(0.1, k + np.random.normal(0, self.drift_rate * 5)) for k in self.k_nom]
        self.c_true = [max(0.1, c + np.random.normal(0, self.drift_rate * 2)) for c in self.c_nom]
        
        K = torch.diag(torch.tensor(self.k_true, dtype=torch.float32))
        C = torch.diag(torch.tensor(self.c_true, dtype=torch.float32))
        
        M_inv = torch.diag(torch.tensor([1.0/self.m_true, 1.0/self.Ixx_true, 1.0/self.Iyy_true], dtype=torch.float32))
        
        A_pos = -M_inv @ (self.T @ K @ self.T.T)
        A_vel = -M_inv @ (self.T @ C @ self.T.T)
        
        A = torch.zeros((6, 6), dtype=torch.float32)
        A[0:3, 3:6] = torch.eye(3)
        A[3:6, 0:3] = A_pos
        A[3:6, 3:6] = A_vel
        
        B = torch.zeros((6, 4), dtype=torch.float32)
        B[3:6, :] = M_inv @ self.T
        
        return A, B

    def _rk4_true(self, x, u):
        self.A_true, self.B_true = self._update_true_matrices()
        k1 = x @ self.A_true.T + u @ self.B_true.T
        k2 = (x + 0.5 * self.dt * k1) @ self.A_true.T + u @ self.B_true.T
        k3 = (x + 0.5 * self.dt * k2) @ self.A_true.T + u @ self.B_true.T
        k4 = (x + self.dt * k3) @ self.A_true.T + u @ self.B_true.T
        return x + (self.dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

    def step(self, x, u, add_disturbance=False):
        x_true_next = self._rk4_true(x, u)
        
        if add_disturbance:
            x_true_next[0, 3:6] += torch.tensor([2.5, 0.8, -0.8], dtype=torch.float32)
            
        measurement_noise = np.random.multivariate_normal(np.zeros(6), self.Q_v)
        x_measured = x_true_next + torch.tensor(measurement_noise, dtype=torch.float32)
        
        return x_measured, self.m_true

class JointEKF:
    def __init__(self, x_init, m_init, P_init, Q_ekf, R_ekf, dt):
        self.x_hat = np.concatenate([x_init, [m_init]])
        self.P = P_init
        self.Q = Q_ekf
        self.R = R_ekf
        self.dt = dt

    def predict(self, u, jacobians_func):
        m_hat = self.x_hat[6]
        x_state = self.x_hat[0:6]
        
        # Unpack both matrices from the unified function
        A, B = jacobians_func(m_hat)
        
        x_pred = x_state + self.dt * (A @ x_state + B @ u)
        self.x_hat[0:6] = x_pred
        
        F = np.eye(7)
        F[0:6, 0:6] += self.dt * A
        
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        H = np.zeros((6, 7))
        H[0:6, 0:6] = np.eye(6)
        
        y = z - H @ self.x_hat
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x_hat = self.x_hat + K @ y
        self.P = (np.eye(7) - K @ H) @ self.P

class SpatialMPCOrchestrator:
    def __init__(self, m_init, Ixx, Iyy, k_arr, c_arr, pos_arr, N, Q, R, u_min, u_max, dt):
        self.dt = dt
        alpha_cfg = AlphaConfig(alpha_max=0.5, alpha_init=0.0, loss_ref=1e-2, gain=5.0)
        net = HyperResidualNet()
        
        self.dynamics = SpatialHybridDynamics(
            m_init=m_init, Ixx=Ixx, Iyy=Iyy, 
            k_arr=k_arr, c_arr=c_arr, pos_arr=pos_arr, 
            net=net, alpha_cfg=alpha_cfg
        )
        self.mpc = MIMO_HyperRTIMPC(self.dynamics, N, Q, R, u_min, u_max, dt)
        
        P_init = np.eye(7) * 0.1
        Q_ekf = np.eye(7) * 1e-4
        R_ekf = np.eye(6) * 1e-2
        self.estimator = JointEKF(np.zeros(6), m_init, P_init, Q_ekf, R_ekf, dt)

    def _get_jacobians_for_ekf(self, m_hat):
        self.dynamics.update_mass_estimate(m_hat)
        A_t, B_t = self.dynamics._compute_matrices()
        return A_t.detach().numpy(), B_t.detach().numpy()

    def dispatch_control(self, z_measurement):
        self.estimator.update(z_measurement.numpy().flatten())
        
        x_est = torch.tensor(self.estimator.x_hat[0:6], dtype=torch.float32)
        m_est = float(self.estimator.x_hat[6])
        
        self.dynamics.update_mass_estimate(m_est)
        u_opt = self.mpc.step(x_est)
        
        self.estimator.predict(u_opt.detach().numpy(), self._get_jacobians_for_ekf)
        
        return u_opt

    def update_target(self, target_state):
        self.mpc.set_target(target_state)

def execute_mimo_simulation(orchestrator, plant, steps, x_init, target_state=None):
    if target_state is not None:
        orchestrator.update_target(target_state)
        
    X = np.zeros((steps, 6))
    U = np.zeros((steps, 4))
    M_tracking = np.zeros((steps, 2))
    
    x_true = x_init.clone()
    u_prev = torch.zeros(4)
    
    for k in range(steps):
        disturbance_flag = (k == int(steps * 0.4))
        
        z_meas, m_true_actual = plant.step(x_true.unsqueeze(0), u_prev.unsqueeze(0), add_disturbance=disturbance_flag)
        z_meas = z_meas.squeeze(0)
        
        u_opt = orchestrator.dispatch_control(z_meas)
        
        delta_u = torch.clamp(u_opt - u_prev, -5.0, 5.0)
        u_applied = u_prev + delta_u
        
        X[k] = z_meas.detach().numpy()
        U[k] = u_applied.detach().numpy()
        M_tracking[k, 0] = m_true_actual
        M_tracking[k, 1] = orchestrator.estimator.x_hat[6]
        
        x_true = z_meas
        u_prev = u_applied
        
    return X, U, M_tracking

def calculate_spatial_metrics(X, U, M_tracking, dt):
    steps = X.shape[0]
    time_vector = np.arange(steps) * dt
    
    itae_z = np.sum(time_vector * np.abs(X[:, 0])) * dt
    itae_phi = np.sum(time_vector * np.abs(X[:, 1])) * dt
    itae_theta = np.sum(time_vector * np.abs(X[:, 2])) * dt
    total_itae = itae_z + itae_phi + itae_theta
    
    control_tv = np.sum(np.abs(np.diff(U, axis=0)))
    max_error_z = np.max(np.abs(X[:, 0]))
    
    mass_estimation_rmse = np.sqrt(np.mean((M_tracking[:, 0] - M_tracking[:, 1])**2))
    
    threshold = 0.02 * np.abs(X[0, 0]) if np.abs(X[0, 0]) > 0 else 0.01
    settled_indices = np.where(np.abs(X[:, 0]) > threshold)[0]
    settling_time = (settled_indices[-1] * dt) if len(settled_indices) > 0 else 0.0
    
    return total_itae, control_tv, max_error_z, settling_time, mass_estimation_rmse
