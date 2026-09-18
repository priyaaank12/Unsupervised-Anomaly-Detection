"""
train.py
--------
Trains the Convolutional Autoencoder using ONLY normal ("good") images.

Loss function: Mean Squared Error (MSE) between input image and its
reconstruction. We simply want the output to look as close as possible
to the input, pixel by pixel.

Run:
    python src/train.py --data_dir data/train/good --epochs 50
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ConvAutoencoder
from dataset import FolderImageDataset


def train(data_dir, epochs, batch_size, lr, save_path, device):
    dataset = FolderImageDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    best_loss = float("inf")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for images, _ in pbar:
            images = images.to(device)

            optimizer.zero_grad()
            reconstructions = model(images)
            loss = criterion(reconstructions, images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch}: avg reconstruction loss = {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> saved new best model to {save_path}")

    print("Training complete. Best loss:", best_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/train/good")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="models/autoencoder_best.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.save_path, device)
