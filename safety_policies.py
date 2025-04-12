
def safety_filter(action, vfunc, vgrad, state, u_max, eps):
    safety_V = vfunc(state)
    if safety_V <= eps:
        _, _, pz = vgrad(state)
        if pz < 0:
            action = -u_max
        else:
            action = u_max
    return action