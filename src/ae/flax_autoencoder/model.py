from typing import Sequence

from flax import linen as nn


class Encoder(nn.Module):
    """A simple MLP encoder"""

    latents: Sequence[int]
    nonlinearity: str = "relu"

    @nn.compact
    def __call__(self, x):
        for i, n in enumerate(self.latents):
            x = nn.Dense(features=n)(x)
            if i != len(self.latents) - 1:
                x = nn.relu(x) if self.nonlinearity == "relu" else x
        return x


class Decoder(nn.Module):
    """A simple MLP decoder"""

    latents: Sequence[int]
    nonlinearity: str = "relu"

    @nn.compact
    def __call__(self, x):
        for i, n in enumerate(self.latents):
            x = nn.relu(x) if self.nonlinearity == "relu" else x
            x = nn.Dense(features=n)(x)
            if i != len(self.latents) - 1:
                x = nn.relu(x) if self.nonlinearity == "relu" else x
        return x


class Autoencoder(nn.Module):
    """Full autoencoder model"""

    latents: Sequence[int]
    nonlinearity: str = "relu"

    def setup(self):
        input = self.latents[0]
        latents = self.latents[1:]
        self.encoder = Encoder(latents, self.nonlinearity)
        self.decoder = Decoder(([input] + list(latents[:-1]))[::-1], self.nonlinearity)

    def __call__(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)
