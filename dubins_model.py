import jax.numpy as jnp
import numpy as np

from hj_reachability import dynamics
from hj_reachability import sets
import torch

class Dubins3D(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(self,
                 max_turn_rate=1.,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None,
                 speed = 0.5):
        self.speed = speed
        if control_space is None:
            control_space = sets.Box(jnp.array([-max_turn_rate]), jnp.array([max_turn_rate]))
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array([0, 0]), jnp.array([0, 0]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        _, _, psi = state
        v = self.speed
        return jnp.array([v * jnp.cos(psi), v * jnp.sin(psi), 0.])

    def control_jacobian(self, state, time):
        x, y, _ = state
        return jnp.array([
            [0],
            [0],
            [1],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [1., 0.],
            [0., 1.],
            [0., 0.],
        ])

    def continuous_dynamics(self, state, action):
        """Compute the derivatives [dx/dt, dy/dt, dtheta/dt]."""
        x_dot = self.speed*torch.cos(state[:, 2])
        y_dot = self.speed*torch.sin(state[:, 2])
        theta_dot = action[:, 0]
        return torch.stack([x_dot, y_dot, theta_dot], dim=1)

    def discrete_rk4(self, current_state: torch.Tensor, action: torch.Tensor, dt: float) \
        -> torch.Tensor:
        # k1
        k1 = self.continuous_dynamics(current_state, action)
        # k2
        mid_state_k2 = current_state + 0.5 * dt * k1
        k2 = self.continuous_dynamics(mid_state_k2, action)
        # k3
        mid_state_k3 = current_state + 0.5 * dt * k2
        k3 = self.continuous_dynamics(mid_state_k3, action)
        # k4
        end_state_k4 = current_state + dt * k3
        k4 = self.continuous_dynamics(end_state_k4, action)
        # Combine k1, k2, k3, k4 to compute the next state
        next_state = current_state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # next_state[..., -1] = next_state[..., -1] % (2 * np.pi)
        return next_state

    def state_error(self, state_1, state_2):
        error = state_1 - state_2
        error[2] = error[2] % (2 * np.pi)
        if error[2] > np.pi:
            error[2] -= 2*np.pi
        return error
