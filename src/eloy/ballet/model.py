try:
    from flax import linen as nn
    import jax.numpy as jnp
    from huggingface_hub import hf_hub_download
except ImportError:
    raise ImportError(
        'jax-related packages are not installed. Use pip insall "eloy[jax]"'
    )

import numpy as np


class CNN(nn.Module):
    """
    Convolutional Neural Network for centroid regression.

    Attributes
    ----------
    params : None
        Placeholder for model parameters.
    """

    params: None = None

    @nn.compact
    def __call__(self, x):
        """
        Forward pass of the CNN.

        Parameters
        ----------
        x : jax.numpy.ndarray
            Input image batch of shape (batch, height, width, channels).

        Returns
        -------
        jax.numpy.ndarray
            Output predictions of shape (batch, 2).
        """
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
    """
    Load model weights from a .npz file.

    Parameters
    ----------
    file : str or Path
        Path to the .npz weights file.

    Returns
    -------
    dict
        Dictionary mapping layer names to their kernel and bias arrays.
    """

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
    """
    Download pretrained weights from HuggingFace Hub.

    Returns
    -------
    str
        Path to the downloaded weights file.
    """

    return hf_hub_download(repo_id="lgrcia/ballet", filename="centroid_15x15.npz")


class Ballet:
    """
    Ballet interface for centroid prediction using a pretrained CNN.

    Attributes
    ----------
    cnn : CNN
        The CNN model instance.
    params : dict
        Model parameters loaded from file.
    """

    cnn: None = None
    params: None = None

    def __init__(self, model_file=None):
        """
        Initialize the Ballet model.

        Parameters
        ----------
        model_file : str or Path, optional
            Path to the model weights file. If None, downloads default weights.
        """

        if model_file is None:
            model_file = download_weights()

        self.cnn = CNN()
        self.params = load_weights_file(model_file)

    def centroid(self, x):
        """
        Predict centroids for input images.

        Parameters
        ----------
        x : numpy.ndarray
            Input images of shape (batch, height, width).

        Returns
        -------
        numpy.ndarray
            Predicted centroids of shape (batch, 2), with coordinates (y, x).
        """

        return self.cnn.apply({"params": self.params}, x[..., None])[:, ::-1]
