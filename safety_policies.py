from v_func_utils import get_spatial_grad, compute_jacobian
import torch
import osqp
from scipy import sparse
import numpy as np

def safety_filter(action, vfunc, vgrad, state, u_max, eps):
    safety_V = vfunc(state)
    if safety_V <= eps:
        _, _, pz = vgrad(state)
        if pz < 0:
            action = -u_max
        else:
            action = u_max
    return action

def qp_filter(nominal_action, vfunc, vgrad, state, u_max, eps, dubins, dt):
    A = compute_jacobian(lambda state_in: dubins.discrete_dynamics(state_in, torch.zeros((1,1)), dt), state)
    B = compute_jacobian(lambda action: dubins.discrete_dynamics(state, action, dt), torch.zeros((1,1)))
    
    # Linearize the value function around the state
    dV_dx = vgrad(state.squeeze())

    # Approximation is V(f(x, u)) = V(f(x, 0) + Bu) = V(f(x, 0)) + dV_dxBu >= eps
    # Cost is (u - u_nom)'(u - u_nom) = u'u - 2*u'u_nom
    P = sparse.csc_matrix([[1.0]], dtype=float)
    q = -np.array([nominal_action])
    A = sparse.csc_matrix(np.vstack([dV_dx@B, np.array([[1.0]])]))
    l = np.array([eps - vfunc(dubins.discrete_dynamics(state, torch.zeros((1, 1)), dt).squeeze()), -u_max])
    u = np.array([1e6, u_max])

    prob = osqp.OSQP()
    prob.setup(P, q, A, l, u, verbose = False, eps_abs=1e-6, eps_rel=0)
    res = prob.solve()
    return res.x[0], res.info.status

def sampling_filter(nominal_action, vfunc, vgrad, state, u_max, eps, dubins, dt, num_samples):
    safety_V = vfunc(state)
    if safety_V <= eps:
        safe_action = safety_filter(nominal_action, vfunc, vgrad, state, u_max, eps)
        possible_actions = torch.linspace(nominal_action, safe_action, num_samples).unsqueeze(1)

        next_states = dubins.discrete_dynamics(state.repeat(num_samples, 1), possible_actions, dt)



        vs = []
        for i in range(len(possible_actions)):
            if vfunc(next_states[i, :]) > eps*1.1:
                return possible_actions[i, 0], vfunc(next_states[i, :])
    else:
        return nominal_action, safety_V
