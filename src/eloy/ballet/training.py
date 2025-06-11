import numpy as np
from tqdm import tqdm

try:
    from eloy.ballet.model import CNN
    import jax
    import jax.numpy as jnp
    from flax.training import train_state
    import optax
except ImportError:
    pass


class Moffat2D:
    """
    Moffat 2D generator
    """

    def __init__(self, cutout_size=21, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.train_history = None
        self.cutout_size = cutout_size
        self.x, self.y = np.indices((cutout_size, cutout_size))

    def moffat2D_model(self, a, x0, y0, sx, sy, theta, b, beta):
        # https://pixinsight.com/doc/tools/DynamicPSF/DynamicPSF.html
        dx_ = self.x - x0
        dy_ = self.y - y0
        dx = dx_ * np.cos(theta) + dy_ * np.sin(theta)
        dy = -dx_ * np.sin(theta) + dy_ * np.cos(theta)

        return b + a / np.power(1 + (dx / sx) ** 2 + (dy / sy) ** 2, beta)

    def sigma_to_fwhm(self, beta):
        return 2 * np.sqrt(np.power(2, 1 / beta) - 1)

    def random_model_label(self, N=10000, flatten=False, return_all=False, sigma=1.0):

        images = []
        labels = []

        a = np.ones(N)
        b = np.zeros(N)
        x0, y0 = np.random.normal(self.cutout_size / 2, sigma, (2, N))
        theta = np.random.uniform(0, np.pi / 8, size=N)
        beta = np.random.uniform(1, 8, size=N)
        sx = np.array(
            [np.random.uniform(1.5, 20.5) / self.sigma_to_fwhm(_beta) for _beta in beta]
        )
        sy = np.random.uniform(0.5, 1.5, size=N) * sx
        noise = np.random.uniform(0, 0.1, size=N)

        for i in range(N):
            _noise = np.random.rand(self.cutout_size, self.cutout_size) * noise[i]
            data = (
                self.moffat2D_model(
                    a[i], x0[i], y0[i], sx[i], sy[i], theta[i], b[i], beta[i]
                )
                + _noise
            )

            images.append(data)

        images = np.array(images)[:, :, :, None]

        if not return_all:
            labels = np.array([x0, y0]).T
        else:
            labels = np.array([a, x0, y0, sx, sy, theta, b, beta, noise]).T

        if N == 1 and flatten:
            return (np.array(images[0]), np.array(labels[0]))
        else:
            return (np.array(images), np.array(labels))


# --- Training utilities ---
class TrainState(train_state.TrainState):
    pass


def compute_loss(params, batch):
    x, y = batch
    preds = model.apply({"params": params}, x)
    return jnp.mean(optax.l2_loss(preds, y))


# def compute_loss(params, batch, delta=1.0):
#     """Robust loss that's less sensitive to outliers than MSE."""
#     x, y = batch
#     preds = model.apply({"params": params}, x)
#     diff = jnp.abs(preds - y)
#     loss = jnp.where(diff < delta, 0.5 * diff**2, delta * diff - 0.5 * delta**2)
#     return jnp.mean(loss)


@jax.jit
def train_step(state, batch):
    grad_fn = jax.value_and_grad(compute_loss)
    loss, grads = grad_fn(state.params, batch)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


@jax.jit
def eval_step(params, batch):
    x, y = batch
    preds = model.apply({"params": params}, x)
    return jnp.sqrt(jnp.mean((preds - y) ** 2))  # RMSE


def params_to_flat_dict(params):
    weights = {}
    for k, v in params.items():
        weights.update(
            {f"{k}_bias": np.array(v["bias"]), f"{k}_kernel": np.array(v["kernel"])}
        )
    return weights


# --- Data batching ---
def get_batches(X, y, batch_size):
    indices = np.random.permutation(len(X))
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield jnp.array(X[batch_idx]), jnp.array(y[batch_idx])


if __name__ == "__main__":
    from datetime import datetime

    size = 15
    rng = jax.random.PRNGKey(0)
    model = CNN()
    params = model.init(rng, jnp.ones([1, size, size, 1]))["params"]
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adamw(5e-5))
    moffat_gen = Moffat2D(size)

    train_size = 5000
    test_size = 10000
    batch_size = 100

    print("Generate test/train samples")
    X_train, y_train = moffat_gen.random_model_label(train_size)
    X_test, y_test = moffat_gen.random_model_label(test_size)

    learning_rate = 1e-3

    state = TrainState.create(
        apply_fn=model.apply, params=state.params, tx=optax.adamw(learning_rate)
    )

    print("Start model training")
    # --- Training loop ---
    for epoch in range(300):
        for batch in get_batches(X_train, y_train, batch_size):
            state, loss = train_step(state, batch)
        if epoch == 100:
            learning_rate = 1e-4
        elif epoch == 150:
            learning_rate = 1e-5
        if epoch % 10 == 0:
            test_rmse = eval_step(state.params, (jnp.array(X_test), jnp.array(y_test)))
            print(
                f"Epoch {epoch}: Loss = {loss:.4f}, Test RMSE = {test_rmse:.4f}, LR = {learning_rate:.1e}"
            )
            X_train, y_train = moffat_gen.random_model_label(train_size)
            state = TrainState.create(
                apply_fn=model.apply, params=state.params, tx=optax.adamw(learning_rate)
            )
        if test_rmse < 0.01:
            break

    print("Adjusting model")
    X_train, y_train = moffat_gen.random_model_label(20000)
    adjust_params = []
    adjust_mean = []
    for i in range(10):
        for batch in get_batches(X_train, y_train, batch_size):
            state, loss = train_step(state, batch)

        predictions = model.apply({"params": state.params}, X_train)
        adjust_params.append(np.mean(predictions - y_train, 0))
        adjust_mean.append(state.params)
        print(f"{i} - (x,y) = {adjust_params[i]}")

    j = np.argmin([np.max(np.abs(d)) for d in adjust_params])
    final_model = adjust_mean[j]
    print(f"Best model: {j} - (x,y) = {adjust_params[j]}")

    print("Saving model file")
    now = datetime.now().isoformat()
    file_name = f"{size}x{size}_{now}.npz"
    np.savez(file_name, **params_to_flat_dict(final_model))
    print("Model saved")
