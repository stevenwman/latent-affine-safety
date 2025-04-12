import torch

# Compute grad of V_func using finite diff
def get_spatial_grad(v_func, state, eps = 0.05):
    state_grad = torch.zeros_like(state)
    for i in range(len(state_grad)):
        delta = torch.zeros_like(state)
        delta[i] = eps
        state_grad[i] = (v_func(state + delta) - v_func(state - delta)) / (2*eps)
    return state_grad