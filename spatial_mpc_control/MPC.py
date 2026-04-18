import torch
import numpy as np
import scipy.sparse as sp
import osqp

class MIMO_HyperRTIMPC:
    def __init__(self, dynamics, N, Q, R, u_min, u_max, dt):
        self.sys = dynamics
        self.N = N
        self.nx = 6
        self.nu = 4
        self.Q = Q
        self.R = R
        self.u_min = np.array(u_min) if isinstance(u_min, (list, np.ndarray)) else np.full(self.nu, u_min)
        self.u_max = np.array(u_max) if isinstance(u_max, (list, np.ndarray)) else np.full(self.nu, u_max)
        self.dt = dt
        self.nz = N * (self.nx + self.nu) + self.nx
        self.x_ref = np.zeros(self.nx)
        self.P, self.q = self._build_hessian()
        self.u_guess = torch.zeros((self.N, self.nu))
        self.x_guess = torch.zeros((self.N + 1, self.nx))
        self.solver = osqp.OSQP()
        self.solver_initialized = False

    def set_target(self, x_target):
        self.x_ref = np.array(x_target)
        self.P, self.q = self._build_hessian()
        if self.solver_initialized:
            self.solver.update(q=self.q)

    def _build_hessian(self):
        blocks = []
        for _ in range(self.N):
            blocks.append(self.Q)
            blocks.append(self.R)
        blocks.append(self.Q)
        P = sp.block_diag(blocks, format='csc')
        
        q = np.zeros(self.nz)
        for k in range(self.N):
            idx_x = k * (self.nx + self.nu)
            q[idx_x:idx_x+self.nx] = -self.Q @ self.x_ref
        q[-self.nx:] = -self.Q @ self.x_ref
        
        return P, q

    def _build_kkt(self, x_init, A_seq, B_seq, x_bar):
        A_eq = sp.lil_matrix((self.nx * (self.N + 1), self.nz))
        l_eq = np.zeros(self.nx * (self.N + 1))
        
        for i in range(self.nx):
            A_eq[i, i] = 1.0
        l_eq[0:self.nx] = x_init.detach().numpy()
        
        for k in range(self.N):
            row = self.nx * (k + 1)
            idx_x = k * (self.nx + self.nu)
            idx_u = idx_x + self.nx
            idx_x_next = idx_u + self.nu
            
            Ak = A_seq[k].detach().numpy()
            Bk = B_seq[k].detach().numpy()
            
            A_eq[row:row+self.nx, idx_x:idx_x+self.nx] = -Ak
            A_eq[row:row+self.nx, idx_u:idx_u+self.nu] = -Bk
            A_eq[row:row+self.nx, idx_x_next:idx_x_next+self.nx] = np.eye(self.nx)
            
            x_k_next_val = x_bar[k+1].detach().numpy()
            x_k_val = x_bar[k].detach().numpy()
            u_k_val = self.u_guess[k].detach().numpy()
            
            rk = x_k_next_val - Ak @ x_k_val - Bk @ u_k_val
            l_eq[row:row+self.nx] = rk
            
        u_eq = l_eq.copy()
        
        A_ineq = sp.lil_matrix((self.N * self.nu, self.nz))
        l_ineq = np.tile(self.u_min, self.N)
        u_ineq = np.tile(self.u_max, self.N)
        
        for k in range(self.N):
            row = k * self.nu
            idx_u = k * (self.nx + self.nu) + self.nx
            A_ineq[row:row+self.nu, idx_u:idx_u+self.nu] = np.eye(self.nu)
            
        A_qp = sp.vstack([A_eq, A_ineq], format='csc')
        l_qp = np.hstack([l_eq, l_ineq])
        u_qp = np.hstack([u_eq, u_ineq])
        
        return A_qp, l_qp, u_qp

    def _shift_horizon(self):
        self.u_guess[:-1] = self.u_guess[1:].clone()
        self.u_guess[-1] = self.u_guess[-1].clone()
        self.x_guess[:-1] = self.x_guess[1:].clone()
        self.x_guess[-1] = self.x_guess[-1].clone()

    # def step(self, x_current):
    #     with torch.no_grad():
    #         self._shift_horizon()
    #         self.x_guess[0] = x_current
    #         for k in range(self.N):
    #             self.x_guess[k+1] = self.sys.rk4_step(
    #                 self.x_guess[k].unsqueeze(0), 
    #                 self.u_guess[k].unsqueeze(0), 
    #                 self.dt
    #             ).squeeze(0)
                
    #     A_seq, B_seq = self.sys.rk4_jacobians(self.x_guess[:-1], self.u_guess, self.dt)
    #     A_qp, l_qp, u_qp = self._build_kkt(x_current, A_seq, B_seq, self.x_guess)
        
    #     if not self.solver_initialized:
    #         self.solver.setup(P=self.P, q=self.q, A=A_qp, l=l_qp, u=u_qp, warm_start=True, verbose=False)
    #         self.solver_initialized = True
    #     else:
    #         self.solver.update(l=l_qp, u=u_qp)
    #         self.solver.update(Ax=A_qp.data)
            
    #     res = self.solver.solve()
        
    #     if res.info.status_val != 1:
    #         return self.u_guess[0].clone()
            
    #     z_opt = res.x
    #     for k in range(self.N):
    #         idx_x = k * (self.nx + self.nu)
    #         idx_u = idx_x + self.nx
    #         self.x_guess[k] = torch.tensor(z_opt[idx_x:idx_x+self.nx], dtype=torch.float32)
    #         self.u_guess[k] = torch.tensor(z_opt[idx_u:idx_u+self.nu], dtype=torch.float32)
    #     self.x_guess[self.N] = torch.tensor(z_opt[-self.nx:], dtype=torch.float32)
        
    #     return self.u_guess[0].clone()
    def step(self, x_current):
        with torch.no_grad():
            self._shift_horizon()
            self.x_guess[0] = x_current
            for k in range(self.N):
                self.x_guess[k+1] = self.sys.rk4_step(
                    self.x_guess[k].unsqueeze(0), 
                    self.u_guess[k].unsqueeze(0), 
                    self.dt
                ).squeeze(0)
                
        A_seq, B_seq = self.sys.rk4_jacobians(self.x_guess[:-1], self.u_guess, self.dt)
        A_qp, l_qp, u_qp = self._build_kkt(x_current, A_seq, B_seq, self.x_guess)
        
        if not self.solver_initialized:
            self.solver.setup(P=self.P, q=self.q, A=A_qp, l=l_qp, u=u_qp, warm_start=True, verbose=False)
            self.solver_initialized = True
        else:
            self.solver.update(l=l_qp, u=u_qp)
            try:
                # Attempt a fast hot-start update
                self.solver.update(Ax=A_qp.data)
            except ValueError:
                # Sparsity pattern changed! OSQP memory out of bounds.
                # Re-instantiate and setup the solver to allocate the new size.
                self.solver = osqp.OSQP()
                self.solver.setup(P=self.P, q=self.q, A=A_qp, l=l_qp, u=u_qp, warm_start=True, verbose=False)
            
        res = self.solver.solve()
        
        if res.info.status_val != 1:
            return self.u_guess[0].clone()
            
        z_opt = res.x
        for k in range(self.N):
            idx_x = k * (self.nx + self.nu)
            idx_u = idx_x + self.nx
            self.x_guess[k] = torch.tensor(z_opt[idx_x:idx_x+self.nx], dtype=torch.float32)
            self.u_guess[k] = torch.tensor(z_opt[idx_u:idx_u+self.nu], dtype=torch.float32)
        self.x_guess[self.N] = torch.tensor(z_opt[-self.nx:], dtype=torch.float32)
        
        return self.u_guess[0].clone()