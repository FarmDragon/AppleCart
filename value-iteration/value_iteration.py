import numpy as np


def compute_quadratic_cost(Q, data):
    """
    Computes the cost of each sample data

    Q is of size (num_states x num_states)
    data is of size (num_states x num_samples)
    return a cost of size (num_samples,)
    """
    assert Q.shape[0] == data.shape[0]
    if len(data.shape) != 2:
        data = data.reshape(-1, 1)
    return np.einsum("xs,xx,xs->s", data, Q, data)


def compute_state_cost(Q, target_state, data):
    """
    Compute the state cost of each sample in state

    Q is of size (num_states x num_states)
    target_state is of size (num_states x 1)
    state is of size (num_states x num_samples)
    return is of size (num_samples,)
    """
    if len(data.shape) != 2:
        data = data.reshape(-1, 1)
    return compute_quadratic_cost(Q, data - target_state)


def compute_u_star(R_diag, dJdX, dstate_dynamics_du):
    """
    Compute optimal control given R, dJdX, and dXdU

    R_diag is an array of size num_inputs that is the diagonal entries of R
    dJdX is of shape (num_states x num_samples)
    dstate_dynamics_du are (num_states x num_inputs x num_samples)
    return u_star of shape (num_inputs x num_samples)
    """
    R_inverse = np.diag(1 / R_diag)
    return -1 / 2 * np.einsum("uu,xut,xt->ut", R_inverse, dstate_dynamics_du, dJdX)
