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
