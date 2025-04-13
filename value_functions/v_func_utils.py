import torch

# Compute grad of V_func using finite diff
def get_spatial_grad(v_func, state, eps = 0.05):
    state_grad = torch.zeros_like(state)
    for i in range(len(state_grad)):
        delta = torch.zeros_like(state)
        delta[i] = eps
        state_grad[i] = (v_func(state + delta) - v_func(state - delta)) / (2*eps)
    return state_grad

# Compute jacobian of a function using finite diff
def compute_jacobian(func, x, eps = 0.05):
    output = func(x)
    J = torch.zeros((output.shape[1], x.shape[1]))
    for i in range(x.shape[1]):
        delta = torch.zeros_like(x)

        delta[:, i] = eps
        x + delta

        ((func(x + delta) - func(x - delta)) / (2 * eps)).squeeze()
        J[:, i] = ((func(x + delta) - func(x - delta)) / (2 * eps))
    return J