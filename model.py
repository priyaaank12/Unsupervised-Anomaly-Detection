"""
model.py
--------
A Convolutional Autoencoder (CAE).

WHY AN AUTOENCODER?
An autoencoder has two parts:
  1. Encoder: squeezes the image down into a small "bottleneck" representation
              (it is forced to keep only the most important information).
  2. Decoder: tries to rebuild the original image from that squeezed representation.

If we train it ONLY on normal/good images, it becomes really good at rebuilding
normal patterns (smooth surfaces, expected textures, correct shapes).

When we later feed it a DEFECTIVE image, the decoder has never learned how to
reconstruct a scratch/dent/crack — so that region comes out blurry/wrong.
The difference between the input image and the reconstructed image
(called "reconstruction error") is highest exactly where the defect is.

That reconstruction error map IS our anomaly detector.
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()

        # ---------------- ENCODER ----------------
        # Each block halves the image size and increases feature depth.
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),           # 64 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),          # 32 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),         # 16 -> 8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # ---------------- DECODER ----------------
        # Mirror of the encoder — each block doubles the image size back up.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 16 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 32 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),  # 64 -> 128
            nn.Sigmoid(),  # pixel values squashed to [0, 1] to match normalized input
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


if __name__ == "__main__":
    # quick sanity check: does a fake image pass through correctly?
    model = ConvAutoencoder()
    dummy = torch.randn(2, 3, 128, 128)
    out = model(dummy)
    print("Input shape :", dummy.shape)
    print("Output shape:", out.shape)  # should match input shape
