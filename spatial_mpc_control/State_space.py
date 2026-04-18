import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jacrev, vmap
from dataclasses import dataclass
from typing import Tuple, Dict

class HyperResidualNet(nn.Module):
    def __init__(self, input_dim: int = 11, hidden_dim: int = 64, output_dim: int = 3, n_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, u: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if x.dim() > 1:
            theta_exp = theta.expand(x.shape[0], -1)
            xut = torch.cat([x, u, theta_exp], dim=-1)
        else:
            xut = torch.cat([x, u, theta], dim=-1)
        return self.net(xut)

@dataclass
class AlphaConfig:
    alpha_max: float = 0.5
    alpha_init: float = 0.0
    loss_ref: float = 1e-2
    gain: float = 5.0

class AlphaScheduler:
    def __init__(self, cfg: AlphaConfig = AlphaConfig()):
        self.cfg = cfg
        self._alpha = cfg.alpha_init

    @property
    def alpha(self) -> float:
        return self._alpha

    def update(self, val_loss: float) -> float:
        ratio = val_loss / max(self.cfg.loss_ref, 1e-12)
        raw = torch.sigmoid(torch.tensor(self.cfg.gain * (1.0 - ratio))).item()
        new_alpha = self.cfg.alpha_max * raw
        self._alpha = max(self._alpha, new_alpha)
        return self._alpha

    def reset(self):
        self._alpha = self.cfg.alpha_init

class SpatialHybridDynamics(nn.Module):
    def __init__(self, m_init: float, Ixx: float, Iyy: float, 
                 k_arr: list[float], c_arr: list[float], 
                 pos_arr: list[Tuple[float, float]], 
                 net: HyperResidualNet | None = None, 
                 alpha_cfg: AlphaConfig = AlphaConfig(), 
                 device: str | torch.device = "cpu"):
        super().__init__()
        self.register_buffer("m_hat", torch.tensor([m_init], dtype=torch.float32))
        self.Ixx = float(Ixx)
        self.Iyy = float(Iyy)
        
        T_list = [[1.0, float(y_i), -float(x_i)] for x_i, y_i in pos_arr]
        T = torch.tensor(T_list, dtype=torch.float32).T
        self.register_buffer("T", T)

        K = torch.diag(torch.tensor(k_arr, dtype=torch.float32))
        C = torch.diag(torch.tensor(c_arr, dtype=torch.float32))
        self.register_buffer("K", K)
        self.register_buffer("C", C)
        
        self.net = net if net is not None else HyperResidualNet()
        self._scheduler = AlphaScheduler(alpha_cfg)
        self.to(device)

    def update_mass_estimate(self, m_new: float):
        self.m_hat[0] = m_new

    def extract_telemetry(self, x: torch.Tensor, u: torch.Tensor) -> Dict[str, torch.Tensor]:
        pos = x[..., 0:3]
        vel = x[..., 3:6]
        
        z_corners = pos @ self.T
        v_corners = vel @ self.T
        
        f_spring = -(z_corners @ self.K)
        f_damper = -(v_corners @ self.C)
        f_passive = f_spring + f_damper
        
        f_net_corners = f_passive + u
        wrench_cg = f_net_corners @ self.T.T
        
        M_inv = torch.diag(torch.tensor([1.0/self.m_hat.item(), 1.0/self.Ixx, 1.0/self.Iyy], dtype=torch.float32, device=self.T.device))
        accel_nominal = wrench_cg @ M_inv
        
        accel_residual = self.net(x, u, self.m_hat)
        
        return {
            "z_corners": z_corners,
            "v_corners": v_corners,
            "f_spring": f_spring,
            "f_damper": f_damper,
            "f_passive": f_passive,
            "f_net_corners": f_net_corners,
            "wrench_cg": wrench_cg,
            "accel_nominal": accel_nominal,
            "accel_residual": accel_residual
        }

    def _compute_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        M_inv = torch.diag(torch.tensor([1.0/self.m_hat.item(), 1.0/self.Ixx, 1.0/self.Iyy], dtype=torch.float32, device=self.T.device))
        
        A_pos = -M_inv @ (self.T @ self.K @ self.T.T)
        A_vel = -M_inv @ (self.T @ self.C @ self.T.T)
        
        A = torch.zeros((6, 6), dtype=torch.float32, device=self.T.device)
        A[0:3, 3:6] = torch.eye(3, device=self.T.device)
        A[3:6, 0:3] = A_pos
        A[3:6, 3:6] = A_vel
        
        B = torch.zeros((6, 4), dtype=torch.float32, device=self.T.device)
        B[3:6, :] = M_inv @ self.T
        
        return A, B

    def _alpha_tensor(self) -> torch.Tensor:
        return torch.tensor(self._scheduler.alpha, dtype=torch.float32, device=self.T.device)

    def _f_physics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        A, B = self._compute_matrices()
        return x @ A.T + u @ B.T

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        alpha = self._alpha_tensor().to(dtype=x.dtype, device=x.device)
        f_p = self._f_physics(x, u)
        f_d_accel = self.net(x, u, self.m_hat)
        f_d = torch.cat([torch.zeros_like(f_d_accel), f_d_accel], dim=-1)
        return f_p + alpha * f_d

    def rk4_step(self, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        k1 = self(x, u)
        k2 = self(x + 0.5 * dt * k1, u)
        k3 = self(x + 0.5 * dt * k2, u)
        k4 = self(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

    def rk4_jacobians(self, x: torch.Tensor, u: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        def step_fn(x_i: torch.Tensor, u_i: torch.Tensor) -> torch.Tensor:
            return self.rk4_step(x_i.unsqueeze(0), u_i.unsqueeze(0), dt).squeeze(0)

        A_k = vmap(jacrev(step_fn, argnums=0))(x, u)
        B_k = vmap(jacrev(step_fn, argnums=1))(x, u)
        return A_k, B_k