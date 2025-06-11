try:
    from flax import linen as nn
    import jax.numpy as jnp
except:
    pass

import numpy as np
from pathlib import Path
import requests


class CNN(nn.Module):
    params: None = None

    @nn.compact
    def __call__(self, x):
        x = x - jnp.min(x, axis=(1, 2, 3), keepdims=True)  # Center input
        x = x / jnp.max(x, axis=(1, 2, 3), keepdims=True)  # Normalize input
        x = nn.Conv(64, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2), padding="SAME")
        x = nn.Conv(128, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2), padding="SAME")
        x = nn.Conv(256, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(2048)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(512)(x)
        x = nn.sigmoid(x)
        return nn.Dense(2)(x)


def load_weights_file(file):

    weights = np.load(file)
    layers = np.unique(
        [key.replace("_bias", "").replace("_kernel", "") for key in weights.keys()]
    )

    return {
        layer: {
            "kernel": weights[f"{layer}_kernel"],
            "bias": weights[f"{layer}_bias"],
        }
        for layer in layers
    }


def download_weights():
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id="lgrcia/ballet", filename="centroid_15x15.npz")


class Ballet:
    cnn: None = None
    params: None = None

    def __init__(self, model_file=None):
        if model_file is None:
            model_file = download_weights()

        self.cnn = CNN()
        self.params = load_weights_file(model_file)

    def centroid(self, x):
        return self.cnn.apply({"params": self.params}, x[..., None])[:, ::-1]
