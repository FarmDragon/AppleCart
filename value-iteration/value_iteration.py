import numpy as np
from IPython.display import HTML, display
from pydrake.all import (
    RandomGenerator,
    LeafSystem,
)
from underactuated.optimizers import Adam
import altair as alt
import pandas as pd
import seaborn as sns
from IPython.display import display, clear_output


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


def ContinuousFittedValueIteration(
    plant,
    plant_context,
    value_mlp,
    state_cost_function,
    compute_u_star,
    R_diag,
    state_samples,
    time_step=0.01,
    discount_factor=1.0,
    lr=0.001,
    minibatch=None,
    epochs=1000,
    optimization_steps_per_epoch=25,
    input_limits=None,
    target_state=None,
):
    if "get_actuation_input_port" in dir(plant):
        input_port = plant.get_actuation_input_port()
    else:
        input_port = plant.get_input_port(0)
    num_states = plant.num_continuous_states()
    num_inputs = input_port.size()
    if target_state is not None:
        np.append(state_samples, target_state)

    N = state_samples.shape[1]

    # perform some checks to make sure the inputs to the function make sense
    assert plant_context.has_only_continuous_state()
    assert value_mlp.get_input_port().size() == num_states
    assert value_mlp.layers()[-1] == 1
    assert R_diag.shape == (num_inputs,)
    assert state_samples.shape[0] == num_states
    assert time_step > 0.0
    assert discount_factor > 0.0 and discount_factor <= 1.0
    if input_limits is not None:
        assert (
            num_inputs == 1
        ), "Input limits are only supported for scalar inputs (for now)"
        assert len(input_limits) == 2

    # random initialization of our Neural Network weights
    mlp_context = value_mlp.CreateDefaultContext()
    generator = RandomGenerator(123)
    value_mlp.SetRandomContext(mlp_context, generator)

    state_cost = state_cost_function(state_samples)
    state_dynamics_x = np.empty((N, num_states))
    dstate_dynamics_du = np.empty((num_states, num_inputs, N))
    state = plant_context.get_mutable_continuous_state_vector()

    # Precompute dynamics of zero-order hold and cost.
    for i in range(N):
        u = np.zeros(num_inputs)
        input_port.FixValue(plant_context, u)
        state.SetFromVector(state_samples[:, i])
        state_dynamics_x[i] = plant.EvalTimeDerivatives(plant_context).CopyToVector()
        for j in range(num_inputs):
            u[j] = 1
            input_port.FixValue(plant_context, u)
            dstate_dynamics_du[:, j, i] = (
                plant.EvalTimeDerivatives(plant_context).CopyToVector()
                - state_dynamics_x[i]
            )
            u[j] = 0

    optimizer = Adam(value_mlp.GetMutableParameters(mlp_context), lr=lr)

    if minibatch and target_state is not None:
        M = minibatch + 1
    elif minibatch:
        M = minibatch
    else:
        M = N

    J = np.zeros((1, M))
    Jnext = np.zeros((1, M))
    Jd = np.zeros((1, M))
    dJdX = np.asfortranarray(np.zeros((num_states, M)))
    dloss_dparams = np.zeros(value_mlp.num_parameters())

    last_loss = np.inf
    for epoch in range(epochs):
        if minibatch:
            batch = np.random.randint(0, N, minibatch)
            # always include the target state in the batch
            if target_state is not None:
                batch = np.append(batch, -1)
        else:
            batch = range(N)

        # Compute dJdX
        value_mlp.BatchOutput(mlp_context, state_samples[:, batch], J, dJdX)

        # compute the next input
        u_star = compute_u_star(R_diag, dJdX, dstate_dynamics_du[:, :, batch])

        # clamp to input limits
        if input_limits is not None:
            u_star = np.clip(u_star, input_limits[0], input_limits[1])

        # compute Xnext
        Xprev = state_samples[:, batch]
        f1 = state_dynamics_x[batch, :].T
        f2_u = np.einsum("xun,un->xn", dstate_dynamics_du[:, :, batch], u_star)
        Xnext = Xprev + time_step * (f1 + f2_u)

        # compute cost
        G = state_cost[batch] + compute_quadratic_cost(np.diag(R_diag), u_star)

        value_mlp.BatchOutput(mlp_context, Xnext, Jnext)

        # Create the target network
        Jd[:] = time_step * G + discount_factor * Jnext
        loss_over_time = []
        for i in range(optimization_steps_per_epoch):
            # low pass filter target network
            if (i + 1) % 50:
                alpha = 5e-4
                Jd[:] = (1 - alpha) * Jd[:] + alpha * Jnext[:]

            # This does back prop
            loss = value_mlp.BackpropagationMeanSquaredError(
                mlp_context, state_samples[:, batch], Jd, dloss_dparams
            )
            loss_over_time.append(loss)
            optimizer.step(loss, dloss_dparams)
        if not minibatch and np.linalg.norm(last_loss - loss) < 1e-8:
            break
        last_loss = loss
        clear_output(wait=True)
        display("loss: {:.6} epoch: {:}/{:}".format(last_loss, epoch, epochs))

    return (mlp_context, loss_over_time)


class ContinuousFittedValueIterationPolicy(LeafSystem):
    """A Drake system to wire our controller to the Drake simulator"""

    def __init__(
        self,
        plant,
        value_mlp,
        value_mlp_context,
        R_diag,
        compute_u_star,
        input_limits=None,
    ):
        LeafSystem.__init__(self)

        self.num_plant_states = value_mlp.get_input_port().size()
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()

        self.value_mlp = value_mlp
        self.value_mlp_context = value_mlp_context
        self.J = np.zeros((1, 1))
        self.dJdX = np.asfortranarray(np.zeros((self.num_plant_states, 1)))

        self.compute_u_star = compute_u_star

        self.R_inverse = 1 / R_diag
        self.R_diag = R_diag
        self.input_limits = input_limits
        self.DeclareVectorInputPort("plant_state", self.num_plant_states)
        if "get_actuation_input_port" in dir(self._plant):
            self._plant_input_port = self._plant.get_actuation_input_port()
        else:
            self._plant_input_port = self._plant.get_input_port(0)
        self.DeclareVectorOutputPort(
            "output", self._plant_input_port.size(), self.CalcOutput
        )

    def CalcOutput(self, context, output):
        num_inputs = self._plant_input_port.size()
        u = np.zeros(num_inputs)
        plant_state = self.get_input_port().Eval(context)

        self.value_mlp.BatchOutput(
            self.value_mlp_context, np.atleast_2d(plant_state).T, self.J, self.dJdX
        )

        self._plant_context.SetContinuousState(plant_state)
        self._plant_input_port.FixValue(self._plant_context, u)
        state_dynamics_x = self._plant.EvalTimeDerivatives(
            self._plant_context
        ).CopyToVector()

        dstate_dynamics_du = np.empty((self.num_plant_states, num_inputs, 1))
        for i in range(num_inputs):
            u[i] = 1
            self._plant_input_port.FixValue(self._plant_context, u)
            dstate_dynamics_du[:, :, i] = (
                self._plant.EvalTimeDerivatives(self._plant_context).CopyToVector()
                - state_dynamics_x
            ).reshape(-1, 1)
            u[i] = 0

        u_star = self.compute_u_star(self.R_diag, self.dJdX, dstate_dynamics_du)[:, 0]
        if self.input_limits is not None:
            u_star = np.clip(u_star, self.input_limits[0], self.input_limits[1])
        for i in range(num_inputs):
            output.SetAtIndex(i, u_star[i])


def simulate_and_animate(starting_state, visualizer, simulator, sim_time=5):
    """
    Simulates the system and produce a video
    """

    visualizer.start_recording()

    context = simulator.get_mutable_context()
    context.SetTime(0.0)
    context.SetContinuousState(starting_state)
    simulator.Initialize()
    simulator.AdvanceTo(sim_time)

    visualizer.stop_recording()

    ani = visualizer.get_recording_as_animation()
    display(HTML(ani.to_jshtml()))
    visualizer.reset_recording()


def plot_loss(loss_over_time):
    losses = pd.DataFrame({"epoch": range(len(loss_over_time)), "loss": loss_over_time})
    return (
        alt.Chart(losses)
        .mark_line()
        .encode(
            x="epoch",
            y="loss",
        )
    )


def plot_J(data, x, y, color="turbo"):
    data = data.pivot(y["name"], x["name"], "J")

    def fmt(s):
        try:
            n = "{:.2f}".format(float(s))
        except:
            n = ""
        return n

    def fmt_angle(s):
        try:
            n = "{:.2f} τ".format(float(s) / 2 / np.pi)
        except:
            n = ""
        return n

    ax = sns.heatmap(data, cmap=color)

    ax.invert_yaxis()
    _ = ax.set_xticklabels(
        [
            fmt_angle(label.get_text()) if x["is_angle"] else fmt(label.get_text())
            for label in ax.get_xticklabels()
        ]
    )
    _ = ax.set_yticklabels(
        [
            fmt_angle(label.get_text()) if y["is_angle"] else fmt(label.get_text())
            for label in ax.get_yticklabels()
        ],
        rotation=0,
        horizontalalignment="right",
    )
