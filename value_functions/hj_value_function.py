import hj_reachability as hj
from hj_reachability import dynamics
from hj_reachability import sets
import numpy as np
import torch
import jax

class HJValueFunction():
    def __init__(self,
                 dyn_sys,
                 grid_min = np.array([-2., -2., 0.]),
                 grid_max = np.array([2., 2., 2 * np.pi]),
                 num_cells = (51, 51, 51)):
        self.dyn_sys = dyn_sys
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_cells = num_cells
        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(grid_min, grid_max),
                                                               num_cells,
                                                               periodic_dims=2)
        self.grid_res = np.array([self.grid_max[0] - self.grid_min[0], 
                                  self.grid_max[1] - self.grid_min[1], 
                                  self.grid_max[2] - self.grid_min[2]]) / np.array([self.num_cells[0] - 1, 
                                                                                    self.num_cells[1] - 1, 
                                                                                    self.num_cells[2]])

        self.solver_settings = hj.SolverSettings.with_accuracy("very_high",
                                                  hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)

    def compute_V(self, failure_lx, time = 0., target_time = -2.8):
        self.val_grid = hj.step(self.solver_settings, self.dyn_sys, self.grid, time, failure_lx, target_time)

    def V(self, state):
        state = state.detach().clone()
        #state[..., 2] -= np.pi/2
        state[..., 2] = torch.where(state[..., 2] > 2*np.pi, state[..., 2] - 2*np.pi, state[..., 2])
        state[..., 2] = torch.where(state[..., 2] < 0, state[..., 2] + 2*np.pi, state[..., 2])

        x_idx = (state[..., 0] - self.grid_min[0]) / self.grid_res[0]
        y_idx = (state[..., 1] - self.grid_min[1]) / self.grid_res[1]
        z_idx = (state[..., 2] - self.grid_min[2]) / self.grid_res[2]

        x_high, y_high, z_high = torch.ceil(x_idx), torch.ceil(y_idx), torch.ceil(z_idx)
        x_low, y_low, z_low = torch.floor(x_idx), torch.floor(y_idx), torch.floor(z_idx)
        x_weight = (x_idx - x_low).numpy()
        y_weight = (y_idx - y_low).numpy()
        z_weight = (z_idx - z_low).numpy()

        # Clamp indices to be within the grid bounds
        x_low = torch.clamp(x_low, 0, self.grid.shape[0] - 1).to(int).item()
        y_low = torch.clamp(y_low, 0, self.grid.shape[1] - 1).to(int).item()
        z_low = torch.clamp(z_low, 0, self.grid.shape[2] - 1).to(int).item()
        x_high = torch.clamp(x_high, 0, self.grid.shape[0] - 1).to(int).item()
        y_high = torch.clamp(y_high, 0, self.grid.shape[1] - 1).to(int).item()
        z_high = torch.clamp(z_high, 0, self.grid.shape[2] - 1).to(int).item()        

        # Perform trilinear interpolation
        c000 = self.val_grid[x_low, y_low, z_low]
        c100 = self.val_grid[x_high, y_low, z_low]
        c010 = self.val_grid[x_low, y_high, z_low]
        c001 = self.val_grid[x_low, y_low, z_high]
        c101 = self.val_grid[x_high, y_low, z_high]
        c011 = self.val_grid[x_low, y_high, z_high]
        c110 = self.val_grid[x_high, y_high, z_low]
        c111 = self.val_grid[x_high, y_high, z_high]

        c00 = c000 * (1 - x_weight) + c100 * x_weight
        c01 = c001 * (1 - x_weight) + c101 * x_weight
        c10 = c010 * (1 - x_weight) + c110 * x_weight
        c11 = c011 * (1 - x_weight) + c111 * x_weight

        c0 = c00 * (1 - y_weight) + c10 * y_weight
        c1 = c01 * (1 - y_weight) + c11 * y_weight

        interpolated_value = c0 * (1 - z_weight) + c1 * z_weight
        return float(interpolated_value)    

    def theta_to_grid(self, theta):
        return int(np.floor((self.num_cells[2] - 1) * np.clip((theta - self.grid_min[2]) / (self.grid_max[2] - self.grid_min[2]), 0, 1)))




                            